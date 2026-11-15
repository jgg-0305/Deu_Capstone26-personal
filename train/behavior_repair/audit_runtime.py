"""Compare original GUI and web inference using identical video detections.

The upstream directory must be a reviewed checkout of the original project.
This does not run its GUI, change deployed models, or use predicted labels as truth.
"""
import argparse
import ast
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.detector import HumanDetector, ObjectDetector
from core.kpt_behavior_classifier import TrackBehaviorBuffer, _encode
from core.labels import label_en
from core.theft_monitor import TheftMonitor
from core.tracker import HumanTracker
from core.tracking_geometry import match_keypoints


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_matching(path):
    # Import only these reviewed pure functions, not Qt / notification handlers.
    tree = ast.parse(path.read_text('utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {'_iou', '_match_keypoints'}]
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope['_match_keypoints']


def plain_predictions(predictions):
    return {str(tid): [int(label), float(score)] for tid, (label, score) in predictions.items()}


def summarize(rows, section, start, end, actor):
    selected = [row for row in rows if start - 1e-4 <= row['seconds'] <= end + 1e-4]
    counts = Counter()
    classified = 0
    for row in selected:
        predictions = row['predictions'][section]
        values = list(predictions.values()) if actor is None else (
            [predictions[str(actor)]] if str(actor) in predictions else [])
        classified += bool(values)
        counts.update(label_en(p[0]).lower() for p in values)
    return {'source_frames': len(selected), 'frames_with_prediction': classified,
            'prediction_counts': dict(counts)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--pose', type=Path, default=ROOT/'yolo11n-pose.pt')
    parser.add_argument('--old-model', type=Path, default=ROOT/'train/1_behavior/output/kpt_behavior.pth')
    parser.add_argument('--new-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=480)
    parser.add_argument('--confidence', type=float, default=.4)
    parser.add_argument('--event-start', type=float)
    parser.add_argument('--event-end', type=float)
    parser.add_argument('--actor', type=int)
    args = parser.parse_args()
    if (args.event_start is None) != (args.event_end is None):
        parser.error('Both event start and end must be supplied.')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    from ultralytics.utils import LOGGER
    LOGGER.setLevel('ERROR')
    upstream_files = {name: args.upstream/'core'/f'{name}.py' for name in
                      ['kpt_behavior_classifier', 'tracker', 'theft_monitor', 'video_thread']}
    original_behavior = load_module(upstream_files['kpt_behavior_classifier'], 'original_behavior')
    original_tracker = load_module(upstream_files['tracker'], 'original_tracker')
    original_rule = load_module(upstream_files['theft_monitor'], 'original_rule')
    original_match = load_matching(upstream_files['video_thread'])
    pose = HumanDetector(str(args.pose), args.confidence, imgsz=args.resolution)
    objects = ObjectDetector(str(ROOT/'model/objects.pt'), .25, imgsz=args.resolution)
    web_tracker, gui_tracker = HumanTracker(), original_tracker.HumanTracker()
    paths = {'old_v2': args.old_model, 'new': args.new_model}
    buffers = {f'{pipeline}_{name}': cls(str(path)) for pipeline, cls in
               [('original', original_behavior.TrackBehaviorBuffer), ('web', TrackBehaviorBuffer)]
               for name, path in paths.items()}
    # Control: same weights, same actual web features, original buffer algorithm.
    buffers['original_buffer_web_input_new'] = original_behavior.TrackBehaviorBuffer(str(args.new_model))
    gui_rule, web_rule = original_rule.TheftMonitor(), TheftMonitor()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError('Cannot open video')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video needs an explicit valid source FPS')
    rows = []
    feature_error = []
    probability_error = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            index = len(rows)
            seconds = index/fps
            detections = pose.detect(frame)
            found_objects = objects.detect(frame)
            web_tracks = web_tracker.update(detections, frame)
            gui_tracks = gui_tracker.update(detections, frame)
            web_points = match_keypoints(web_tracks, detections)
            gui_points = original_match(gui_tracks, detections)
            predictions = {}
            raw_top = {}
            for name, buffer in buffers.items():
                if name.startswith('web_'):
                    prediction = buffer.update(web_tracks, web_points, source_seconds=seconds)
                elif name == 'original_buffer_web_input_new':
                    prediction = buffer.update(web_tracks, web_points)
                else:
                    prediction = buffer.update(gui_tracks, gui_points)
                predictions[name] = plain_predictions(prediction)
                raw_top[name] = {str(t): [{'label': label_en(int(i)).lower(), 'score': float(p[i])}
                                        for i in np.argsort(-p)[:3]] for t, p in buffer._probs.items()}
            # Forward parity isolates model/label implementation from tracking and history.
            new_buffer = buffers['web_new']
            for track in web_tracks:
                tid = track['track_id']
                current_vec = _encode(web_points.get(tid), track['bbox'])
                original_vec = original_behavior._encode(web_points.get(tid), track['bbox'])
                feature_error.append(float(np.max(np.abs(current_vec-original_vec))))
                if len(new_buffer._bufs.get(tid, ())) == 30:
                    sequence = torch.from_numpy(np.stack(new_buffer._bufs[tid])).unsqueeze(0).to(new_buffer.device)
                    a = torch.softmax(new_buffer.model(sequence), 1)
                    b = torch.softmax(buffers['original_new'].model(sequence), 1)
                    probability_error.append(float((a-b).abs().max()))
            # Original GUI uses wall time. Replay on the source clock to model real-time playback.
            original_rule.time = SimpleNamespace(time=lambda: seconds)
            gui_suspects = gui_rule.update(gui_tracks, gui_points, found_objects)
            web_suspects = web_rule.update([t for t in web_tracks if t['observed']], web_points,
                                           found_objects, source_seconds=seconds, include_lost=True)
            for model in paths:
                shown = dict(predictions['original_'+model])
                for tid in gui_suspects:
                    if shown.get(str(tid), [-1, 0])[0] in (-1, 0):
                        shown[str(tid)] = [6, .9]
                predictions['original_display_'+model] = shown
                web_shown = dict(predictions['web_'+model])
                for track in web_tracks:
                    tid = track['track_id']
                    if buffers['web_'+model].progress(tid)['state'] == 'uncertain':
                        web_shown.pop(str(tid), None)
                    if tid in web_suspects and not web_shown.get(str(tid), [None, None])[0]:
                        web_shown[str(tid)] = [6, None]
                predictions['web_display_'+model] = web_shown
            rows.append({'frame': index, 'seconds': seconds, 'predictions': predictions,
                         'ema_top3': raw_top,
                         'web_progress': {str(t['track_id']): new_buffer.progress(t['track_id']) for t in web_tracks},
                         'objects': dict(Counter(str(o['cls']) for o in found_objects)),
                         'original_rule': list(gui_suspects), 'web_rule': list(web_suspects)})
            if len(rows) % 60 == 0:
                print(f'Compared {len(rows)} frames', flush=True)
    finally:
        capture.release()
    windows = {'whole_clip': (0, len(rows)/fps)}
    if args.event_start is not None:
        windows['annotated_event'] = (args.event_start, args.event_end)
    result = {'source': args.video.name, 'source_sha256': sha256(args.video.read_bytes()).hexdigest(),
              'source_fps': fps, 'frames': len(rows),
              'settings': {'resolution': args.resolution, 'confidence': args.confidence, 'interval': 1},
              'weights': {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
              'pose_sha256': sha256(args.pose.read_bytes()).hexdigest(),
              'upstream_file_sha256': {name: sha256(path.read_bytes()).hexdigest() for name, path in upstream_files.items()},
              'forward_probability_max_abs_difference': max(probability_error, default=None),
              'encoding_max_abs_difference': max(feature_error, default=None),
              'rule_frames': {'original': sum(bool(r['original_rule']) for r in rows),
                              'web': sum(bool(r['web_rule']) for r in rows)},
              'summary': {window: {name: summarize(rows, name, start, end, args.actor)
                                  for name in rows[0]['predictions']} for window, (start, end) in windows.items()},
              'note': 'Source-clock replay; same YOLO detections, separately run trackers. No DB event evaluation. '
                      'Original new-model path intentionally uses original per-frame sampling. '
                      'Counts are predictions, not accuracy unless an independently annotated interval is supplied.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output.with_suffix('.frames.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
