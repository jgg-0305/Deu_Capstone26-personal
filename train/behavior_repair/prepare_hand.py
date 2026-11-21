"""Prepare synchronized RGB wrist crops and pose traces from AI-Hub 71550."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from zipfile import ZipFile

import cv2
import numpy as np
import torch

from aihub_labels import index_archives, read_annotation, split_clips, remove_cross_split_duplicates
from prepare_aihub import detect_batch, embed_batch, iou, match_actor
from config import ROOT

sys.path.insert(0, str(ROOT))
from core.detector import HumanDetector
from core.hand_behavior import encode_context
from core.kpt_behavior_classifier import _encode
from core.tracker import HumanTracker
from ultralytics.utils import LOGGER

CASE = 'C_3_12_41_BU_DYA_07-29_15-20-29_CE_RGB_DF2_F2'


def select(clips, limit):
    selected = []
    for label in range(1, 8):
        groups = set()
        for clip in sorted((c for c in clips if c.label == label),
                           key=lambda c: sha256(('hand-v1:' + c.name).encode()).digest()):
            if clip.group not in groups and len(groups) < limit:
                selected.append(clip)
                groups.add(clip.group)
    return selected


def extract(scratch, expected, events, detector, embedder):
    tracker = HumanTracker(embedder=embedder)
    tracks, detections_at = defaultdict(dict), {}
    markers = {marker[0] for pair in events for marker in pair}
    cwd = Path.cwd()
    os.chdir(scratch)
    try:
        cap = cv2.VideoCapture('clip.mp4')
    finally:
        os.chdir(cwd)
    count, width, height = expected
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 3) > .01:
            raise ValueError('Missing or invalid 3 FPS video')
        for begin in range(0, count, 16):
            frames = []
            for i in range(begin, min(begin + 16, count)):
                ok, frame = cap.read()
                if not ok or frame.shape[:2] != (height, width):
                    raise ValueError(f'Video/XML mismatch at frame {i}')
                frames.append(frame)
            batches = detect_batch(detector, frames)
            embeddings = embed_batch(tracker, frames, batches)
            for i, (frame, detections, embed) in enumerate(zip(frames, batches, embeddings), begin):
                if i in markers:
                    detections_at[i] = detections
                for track in tracker.update(detections, frame, embeds=embed):
                    if track['observed']:
                        box, points = track['bbox'], track['keypoints']
                        vector = encode_context(frame, points, box, _encode(points, box))
                        tracks[track['track_id']][i] = (box, vector.astype(np.float16))
        if cap.read()[0]:
            raise ValueError('Video exceeds XML frame count')
    finally:
        cap.release()
    actors, marker_scores = [], []
    for start, end in events:
        actor = match_actor(tracks, start)
        # End match is for auditing drift and for excluding ambiguous training labels.
        end_actor = match_actor(tracks, end)
        actors.append((-1 if actor is None else actor, start[0], end[0],
                       int(end_actor is not None and end_actor != actor)))
        for frame, box in (start, end):
            marker_scores.append(max((iou(box, d['bbox']) for d in detections_at[frame]), default=0))
    rows = [(tid, frame, vector) for tid, positions in tracks.items()
            for frame, (_, vector) in positions.items()]
    if not rows:
        from core.hand_behavior import INPUT_SIZE
        return (np.empty((0, 2), np.int32), np.empty((0, INPUT_SIZE), np.float16),
                np.array(actors), np.array(marker_scores))
    return (np.array([(tid, frame) for tid, frame, _ in rows], np.int32),
            np.stack([vector for _, _, vector in rows]), np.array(actors), np.array(marker_scores))


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--train-per-class', type=int, default=24)
    parser.add_argument('--val-per-class', type=int, default=5)
    parser.add_argument('--test-per-class', type=int, default=10)
    parser.add_argument('--prefetch', action='store_true', help='Populate the same clip cache from the test end.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_spec = {'data': str(args.data.resolve()), 'fps': 3, 'resolution': 480,
                  'confidence': .4, 'features': 'hand_fusion_v1',
                  'yolo_sha256': sha256((ROOT/'yolo11n-pose.pt').read_bytes()).hexdigest()}
    cache_spec_path = args.out/'cache-spec.json'
    if cache_spec_path.exists() and json.loads(cache_spec_path.read_text('utf-8')) != cache_spec:
        raise ValueError('This output directory contains a different source or extraction specification.')
    if not cache_spec_path.exists():
        atomic_json(cache_spec_path, cache_spec)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    LOGGER.setLevel('ERROR')
    clips, _ = index_archives(args.data)
    groups = split_clips(clips, [CASE])
    remove_cross_split_duplicates(groups)
    selected = {split: select(groups[split], limit) for split, limit in
                [('train', args.train_per_class), ('val', args.val_per_class), ('test', args.test_per_class)]}
    selected['diagnostic'] = [c for c in groups['diagnostic'] if c.name == CASE]
    assert len(selected['diagnostic']) == 1
    group_sets = [set(c.group for c in subset) for subset in selected.values()]
    assert sum(map(len, group_sets)) == len(set.union(*group_sets))
    unpacked = {p.stem: p for p in args.data.rglob('*.mp4')}
    detector = HumanDetector(str(ROOT/'yolo11n-pose.pt'), .4, imgsz=480)
    embedder = HumanTracker().tracker.embedder
    total = sum(map(len, selected.values()))
    manifest = []
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        scratch = Path(temp)
        subsets = list(selected.items())
        if args.prefetch:
            subsets.reverse()
        for split, subset in subsets:
            folder = args.out/split
            folder.mkdir(exist_ok=True)
            for clip in reversed(subset) if args.prefetch else subset:
                destination = folder/(clip.name + '.npz')
                row = {'split': split, 'name': clip.name, 'group': clip.group, 'label': clip.label,
                       'path': str(destination.relative_to(args.out))}
                try:
                    if not destination.exists():
                        with ZipFile(clip.label_zip) as archive:
                            expected, events = read_annotation(archive, clip)
                        with ZipFile(clip.video_zip) as archive, archive.open(clip.video) as video:
                            local = unpacked.get(clip.name)
                            # Verify the unpacked copy against the official ZIP entry before reuse.
                            if local and local.stat().st_size == clip.video_identity[1]:
                                import zlib
                                with local.open('rb') as stream:
                                    crc = 0
                                    while block := stream.read(1 << 20):
                                        crc = zlib.crc32(block, crc)
                                if crc != clip.video_identity[0]:
                                    local = None
                            else:
                                local = None
                            with (scratch/'clip.mp4').open('wb') as target:
                                if local:
                                    with local.open('rb') as stream:
                                        shutil.copyfileobj(stream, target)
                                else:
                                    shutil.copyfileobj(video, target)
                        ids, features, actors, ious = extract(scratch, expected, events, detector, embedder)
                        temp_cache = destination.with_suffix(f'.{os.getpid()}.tmp')
                        with temp_cache.open('wb') as file:
                            np.savez_compressed(file, ids=ids, x=features, events=actors,
                                                detector_ious=ious, frames=expected[0])
                        temp_cache.replace(destination)
                except ValueError as exc:
                    row['error'] = str(exc)
                manifest.append(row)
                if not args.prefetch:
                    atomic_json(args.out/'manifest.json', manifest)
                    atomic_json(args.out/'progress.json', {'stage': 'extract', 'done': len(manifest), 'total': total})
                print(f'{len(manifest)}/{total} {split} {clip.name}', row.get('error', ''), flush=True)


if __name__ == '__main__':
    main()
