"""Reproducible full-clip pose and streaming behavior regression.

Uses existing project modules; does not change deployed weights or labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from zipfile import ZipFile

import numpy as np
import torch

from config import ROOT, OUT, DATA, PREPARED, PREVIOUS, POSE_N, POSE_S
STAGING = ROOT
sys.path.insert(0, str(ROOT))
from aihub_labels import index_archives, read_annotation, split_clips, remove_cross_split_duplicates, LABELS
from prepare_aihub import extract, labeled_windows, match_actor
from core.detector import HumanDetector
from core.tracker import HumanTracker
from core.kpt_behavior_classifier import TrackBehaviorBuffer, _encode
from ultralytics.utils import LOGGER
LOGGER.setLevel('ERROR')

CASE = 'C_3_12_41_BU_DYA_07-29_15-20-29_CE_RGB_DF2_F2'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def get_groups():
    clips, _ = index_archives(DATA)
    examined = json.loads((STAGING / 'docs/design/raw-video-test-set.json').read_text('utf-8'))
    groups = split_clips(clips, [v['file'] for v in examined])
    remove_cross_split_duplicates(groups)
    return clips, groups


def cached_extract(clip, detector, tracker_factory, pose_name):
    # Include code, weights and settings in cache identity.
    code = b''.join((STAGING / f).read_bytes() for f in (
        'core/tracker.py', 'core/detector.py', 'core/kpt_behavior_classifier.py',
        'train/behavior_repair/prepare_aihub.py'))
    pose_path = POSE_S if pose_name == 'yolo11s-pose.pt' else POSE_N
    config = hashlib.sha256(code + pose_path.read_bytes() + b'480,.4,batch16').hexdigest()[:16]
    target = OUT / 'pose-cache' / config / (clip.name + '.npz')
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(clip.label_zip) as z:
        expected, events = read_annotation(z, clip)
    if target.exists():
        with np.load(target, allow_pickle=False) as d:
            tracks = {}
            for tid, frame, box, vec in zip(d['ids'], d['frames'], d['boxes'], d['vectors']):
                tracks.setdefault(int(tid), {})[int(frame)] = (box, vec)
    else:
        with tempfile.TemporaryDirectory(dir=OUT) as scratch:
            video = Path(scratch) / 'clip.mp4'
            with ZipFile(clip.video_zip) as z, z.open(clip.video) as src, video.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            tracks = extract(video, detector, expected, tracker_factory, _encode)
        rows = [(tid, f, b, v) for tid, positions in tracks.items() for f, (b, v) in positions.items()]
        np.savez_compressed(target, ids=np.array([r[0] for r in rows], dtype=np.int64),
                            frames=np.array([r[1] for r in rows], dtype=np.int64),
                            boxes=np.array([r[2] for r in rows]),
                            vectors=np.array([r[3] for r in rows], dtype=np.float32))
    return tracks, expected, events, str(target)


def replay(tracks, count, checkpoint):
    buffer = TrackBehaviorBuffer(str(checkpoint))
    rows = []
    # Invert _encode with a unit bbox; preserves the exact cached 34 features.
    for frame in range(count):
        observed, points = [], {}
        for tid, positions in tracks.items():
            if frame in positions:
                vec = positions[frame][1].reshape(17, 2)
                observed.append({'track_id': tid, 'bbox': [0, 0, 1, 1], 'observed': True})
                points[tid] = np.column_stack((vec, np.any(vec != 0, axis=1).astype(float)))
        predictions = buffer.update(observed, points, source_seconds=frame / 3)
        states = {str(t['track_id']): buffer.progress(t['track_id'])['state'] for t in observed}
        rows.append({'frame': frame, 'seconds': frame / 3,
                     'predictions': {str(t): [int(v[0]), float(v[1])] for t, v in predictions.items()},
                     'states': states})
    return rows


def summarize(rows, tracks, events, label):
    targets = [(match_actor(tracks, s), s[0], e[0]) for s, e in events]
    counts, pocket = Counter(), Counter()
    visible_counts = Counter()
    event_uncertain = event_missing = 0
    event_total = event_classified = 0
    normal_total=normal_classified=0
    normal_counts=Counter()
    normal_before=events[0][0][0]-6
    actors=set(t[0] for t in targets)
    for row in rows:
        f = row['frame']
        if f < normal_before:
            for actor in actors:
                normal_total+=1
                p=row['predictions'].get(str(actor))
                if p:
                    normal_classified+=1
                    normal_counts[LABELS[p[0]]]+=1
        for actor, start, end in targets:
            if start <= f <= end:
                event_total += 1
                pred = row['predictions'].get(str(actor))
                if pred:
                    event_classified += 1
                    counts[LABELS[pred[0]]] += 1
                    if row.get('states', {}).get(str(actor)) == 'uncertain':
                        event_uncertain += 1
                    else:
                        visible_counts[LABELS[pred[0]]] += 1
                else:
                    event_missing += 1
        if 134 <= f <= 143:
            pocket.update(LABELS[p[0]] for p in row['predictions'].values())
    return {'event_annotated_frames': event_total, 'event_classified_frames': event_classified,
            'normal_proxy_frames':normal_total,'normal_proxy_classified_frames':normal_classified,
            'normal_proxy_predictions':dict(normal_counts),
            'event_predictions': dict(counts), 'annotated_events': targets,
            'event_visible_predictions': dict(visible_counts),
            'event_uncertain_frames': event_uncertain,
            'event_missing_or_collecting_frames': event_missing,
            'pocket_44_67_to_47_67': dict(pocket),
            'whole_clip_predictions': dict(Counter(LABELS[p[0]] for r in rows for p in r['predictions'].values()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pose', default='yolo11s-pose.pt')
    p.add_argument('--model', type=Path, default=PREVIOUS)
    p.add_argument('--name', default='baseline')
    p.add_argument('--smoke-count', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    clips, groups = get_groups()
    selected = [next(c for c in clips if c.name == CASE)]
    selected += [c for c in groups['val'] if c.label == 4][:a.smoke_count]
    detector = HumanDetector(str(POSE_S if a.pose == 'yolo11s-pose.pt' else POSE_N), .4, imgsz=480)
    embedder = HumanTracker().tracker.embedder
    reports = []
    for c in selected:
        print('Extract/replay', a.pose, c.name, flush=True)
        tracks, expected, events, cache = cached_extract(c, detector, lambda: HumanTracker(embedder=embedder), a.pose)
        rows = replay(tracks, expected[0], a.model)
        report = {'clip': c.name, 'pose': a.pose, 'cache': cache, 'model_sha256': hashlib.sha256(a.model.read_bytes()).hexdigest(),
                  **summarize(rows, tracks, events, c.label)}
        reports.append(report)
        dump(OUT / f'{a.name}-{a.pose}-{c.name}-frames.json', rows)
        dump(OUT / f'{a.name}-{a.pose}-summary.json', reports)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
