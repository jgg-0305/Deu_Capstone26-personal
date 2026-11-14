"""원본 MP4를 전체 프레임으로 분석하고 추적·행동 모델 진단을 저장한다."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from backend.worker import limit_cpu_threads
from core.detector import HumanDetector, ObjectDetector
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from core.rgb_behavior_classifier import RgbBehaviorClassifier
from core.theft_monitor import TheftMonitor
from core.tracker import HumanTracker
from core.tracking_geometry import match_keypoints

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(args):
    selected = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    pose = HumanDetector(str(ROOT / args.pose), args.confidence, imgsz=args.resolution)
    objects = ObjectDetector(str(ROOT / 'model/objects.pt'), .25, imgsz=args.resolution)
    for detector in (pose, objects):
        detector.model.add_callback('on_predict_start', limit_cpu_threads)
    rgb = RgbBehaviorClassifier(str(ROOT / 'model_final.pth')) if args.rgb else None
    reports = []
    for selected_video in selected:
        if 'path' in selected_video:
            source = Path(selected_video['path'])
        elif args.data_root:
            source = Path(args.data_root) / selected_video['file']
        else:
            raise ValueError('영상 파일을 찾으려면 --data-root로 원본 영상 폴더를 지정하세요.')
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f'영상 열기 실패: {source.name}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        tracker = HumanTracker()
        classifiers = {
            'v2': TrackBehaviorBuffer(str(ROOT / 'train/1_behavior/output/kpt_behavior.pth')),
            'v1': TrackBehaviorBuffer(str(ROOT / 'kpt_behavior.pth')),
        }
        if rgb:
            rgb.features.clear()
            rgb.inference_count = 0
        rule = TheftMonitor()
        rows = []
        ids, frame_count, empty, visible, predicted = set(), 0, 0, 0, 0
        distributions = {name: Counter() for name in ('v2', 'v1', 'rgb')}
        target_frames = {name: [] for name in distributions}
        cached = None
        if args.cache:
            # 캐시는 동일 포즈 모델·해상도에서 얻은 검출만 재사용한다.
            if args.pose != 'yolo11s-pose.pt' or args.resolution != 480 or args.confidence < .25:
                raise ValueError('이 캐시는 YOLO11s 480, 검출 기준 25% 이상에서만 사용할 수 있습니다.')
            cached = json.loads((Path(args.cache) / f"{selected_video['id']}-detections.json").read_text())
        started = time.monotonic()
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                if cached is not None:
                    record = cached[frame_count]
                    detections = [{**d, 'keypoints': np.array(d['keypoints'], dtype=np.float32) if d['keypoints'] is not None else None}
                                  for d in record['detections'] if d['conf'] >= args.confidence]
                    detected_objects = record['objects']
                else:
                    detections, detected_objects = pose.detect(frame), objects.detect(frame)
                tracks = tracker.update(detections, frame)
                points = match_keypoints(tracks, detections)
                observed = [t for t in tracks if t['observed']]
                ids.update(t['track_id'] for t in observed)
                empty += not detections
                visible += bool(observed)
                predicted += bool(tracks) and not observed
                results = {name: classifier.update(tracks, points) for name, classifier in classifiers.items()}
                if rgb:
                    result = rgb.update(frame)
                    results['rgb'] = {0: result} if result is not None else {}
                for name, predictions in results.items():
                    distributions[name].update(str(value[0]) for value in predictions.values())
                    if any(label == 6 and score >= .6 for label, score in predictions.values()):
                        target_frames[name].append(round(timestamp))
                suspects = rule.update(observed, points, detected_objects, source_seconds=timestamp / 1000, include_lost=True)
                rows.append({'frame': frame_count, 'time_ms': round(timestamp),
                             'tracks': [{'id': t['track_id'], 'observed': t['observed'], 'box': t['bbox'],
                                         'buffer': classifiers['v2'].progress(t['track_id'])} for t in tracks],
                             'predictions': results, 'rule_suspects': list(suspects)})
                frame_count += 1
        cap.release()
        with source.open('rb') as source_file:
            digest = hashlib.file_digest(source_file, 'sha256').hexdigest()
        report = {
            'id': selected_video['id'], 'file': source.name,
            'sha256': digest,
            'source_fps': fps, 'frames': frame_count, 'input_window_seconds': 30 / fps,
            'empty_detection_frames': empty, 'observed_frames': visible,
            'predicted_only_frames': predicted, 'track_ids': len(ids),
            'inference_calls': {name: model.inference_count for name, model in classifiers.items()},
            'buffer_resets': classifiers['v2'].reset_count,
            'class_observations': distributions,
            'theft_over_60_percent': {name: {'frames': len(values), 'first_ms': min(values) if values else None,
                                           'last_ms': max(values) if values else None} for name, values in target_frames.items()},
            'rule_frames': sum(bool(row['rule_suspects']) for row in rows),
            'evaluation_seconds': round(time.monotonic() - started, 2),
        }
        if rgb:
            report['inference_calls']['rgb'] = rgb.inference_count
        reports.append(report)
        write_json(output / f"{selected_video['id']}-trace.json", rows)
        write_json(output / 'results.json', {'settings': vars(args), 'videos': reports,
                                            'scope': '정답 구간 XML이 없는 진단 결과이며, 클래스별 정확도나 오탐률이 아닙니다.'})
        print(json.dumps(report, ensure_ascii=False), flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='id와 file 또는 path가 담긴 영상 목록 JSON')
    parser.add_argument('--data-root', help='목록의 파일 이름을 찾을 원본 영상 폴더')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pose', choices=['yolo11n-pose.pt', 'yolo11s-pose.pt'], default='yolo11s-pose.pt')
    parser.add_argument('--resolution', type=int, choices=[384, 480, 640], default=480)
    parser.add_argument('--confidence', type=float, default=.4)
    parser.add_argument('--rgb', action='store_true', help='장면 모델도 함께 검증한다.')
    parser.add_argument('--cache', help='이미 추출한 YOLO11s 480 검출 진단 폴더')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
