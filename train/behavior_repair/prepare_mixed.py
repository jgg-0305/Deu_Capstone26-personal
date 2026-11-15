"""Add independently grouped YOLO11s training inputs to the full YOLO11n set."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import time
import numpy as np
import torch
import cv2
from diagnose import (get_groups, cached_extract, labeled_windows, HumanDetector,
                      HumanTracker, STAGING, OUT, LABELS, dump)
from config import PREPARED, POSE_S


def select(clips, per_class):
    chosen = []
    for label in range(1, 8):
        groups = defaultdict(list)
        for c in clips:
            if c.label == label:
                groups[c.group].append(c)
        # Round robin across filming groups, deterministic and no result-based selection.
        pools = [sorted(v, key=lambda c: hashlib.sha256(c.name.encode()).digest())
                 for k, v in sorted(groups.items())]
        n = 0
        while any(pools) and n < per_class:
            for pool in pools:
                if pool and n < per_class:
                    chosen.append(pool.pop())
                    n += 1
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-per-class', type=int, default=64)
    p.add_argument('--val-per-class', type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    _, groups = get_groups()
    detector = HumanDetector(str(POSE_S), .4, imgsz=480)
    embedder = HumanTracker().tracker.embedder
    target = OUT/'mixed-dataset'
    target.mkdir(exist_ok=True)
    stats = {}
    started = time.time()
    for split, cap in [('train', a.train_per_class), ('val', a.val_per_class)]:
        clips = select(groups[split], cap)
        arrays = {key: [] for key in ['x', 'y', 'groups', 'clips', 'frames']}
        review = {}
        for i, clip in enumerate(clips, 1):
            try:
                tracks, expected, events, _ = cached_extract(clip, detector, lambda: HumanTracker(embedder=embedder), 'yolo11s-pose.pt')
                windows, unmatched = labeled_windows(tracks, events, clip.label)
                review[clip.name] = {'windows': len(windows), 'unmatched': unmatched,
                                     'positive_windows': sum(r[2] == clip.label for r in windows)}
            except (ValueError, TypeError) as error:
                windows = []
                review[clip.name] = {'windows': 0, 'reason': str(error)}
                print('Annotation excluded', clip.name, str(error), flush=True)
            if windows:
                for key, value in zip(arrays, [np.stack([r[1] for r in windows]), np.array([r[2] for r in windows]),
                        np.repeat(clip.group, len(windows)), np.repeat(clip.name, len(windows)),
                        np.array([r[0] for r in windows])]):
                    arrays[key].append(value)
            status = {'stage': 'pose_extraction', 'split': split, 'completed': i, 'total': len(clips),
                      'elapsed_seconds': time.time()-started, 'clip': clip.name}
            dump(OUT/'progress.json', status)
            if i % 5 == 0:
                print(json.dumps(status), flush=True)
        arrays = {k: np.concatenate(v) for k,v in arrays.items()}
        np.savez_compressed(target / f'{split}-s.npz', **arrays)
        with np.load(PREPARED/f'{split}.npz', allow_pickle=False) as d:
            combined = {k: np.concatenate([d[k], arrays[k]]) for k in arrays}
        np.savez_compressed(target/f'{split}.npz', **combined)
        stats[split] = {'s_clips': len(clips), 's_windows': len(arrays['y']),
                        'combined_windows': len(combined['y']), 'review': review}
        dump(target/'preparation-report.json', stats)
    meta = json.loads((PREPARED/'dataset.json').read_text('utf-8'))
    meta.update({'pose': ['yolo11n-pose.pt', 'yolo11s-pose.pt'],
                 'additional_pose_sha256': hashlib.sha256(POSE_S.read_bytes()).hexdigest(),
                 'source_dataset': str(PREPARED),
                 'selection': 'round robin filming groups, SHA256 deterministic camera ordering',
                 'additional_train_clips': a.train_per_class*7, 'additional_val_clips': a.val_per_class*7,
                 'combined_window_counts': {s: v['combined_windows'] for s,v in stats.items()},
                 's_pose_window_counts': {s: v['s_windows'] for s,v in stats.items()}})
    dump(target/'dataset.json', meta)
    print('Completed mixed training/validation dataset', flush=True)


if __name__ == '__main__':
    main()
