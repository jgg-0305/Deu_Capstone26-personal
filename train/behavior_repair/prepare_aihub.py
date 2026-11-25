"""AI-Hub 원본과 주석으로 학습용 관절 시퀀스를 만든다. 촬영 그룹을 분리하고 사건의 인물과 추적 대상이 맞는 구간을 선택하며 추출 캐시를
재사용한다.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from zipfile import ZipFile, ZipInfo

from aihub_labels import (
    LABELS, index_archives, inventory, read_annotation, remove_cross_split_duplicates, split_clips,
)

ROOT = Path(__file__).resolve().parents[2]
SEQ_LEN = 30
STRIDE = 5
FPS = 3.0
NORMAL_MARGIN = 6
MAX_GAP = 3
SPLITS = ("train", "val", "test", "diagnostic")


def audit_labels(groups):
    results = {}
    for split, clips in groups.items():
        counts = Counter()
        skipped = {}
        with ExitStack() as stack:
            labels = {path: stack.enter_context(ZipFile(path)) for path in {c.label_zip for c in clips}}
            for clip in clips:
                try:
                    _, events = read_annotation(labels[clip.label_zip], clip)
                except (ValueError, TypeError) as error:
                    skipped[clip.name] = str(error)
                    continue
                counts[LABELS[clip.label]] += len(events)
        results[split] = {"events": dict(counts), "skipped": skipped}
    return results


def iou(left, right):
    x = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    y = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - x * y)
    return x * y / union if union > 0 else 0


def match_actor(tracks, marker):
    frame, box = marker
    candidates = sorted(
        ((iou(box, positions[frame][0]), tid) for tid, positions in tracks.items() if frame in positions),
        reverse=True,
    )
    if not candidates or candidates[0][0] < 0.5:
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
        return None
    return candidates[0][1]


def inference_windows(positions):
    import numpy as np

    empty = np.zeros(34, dtype=np.float32)
    first = None
    previous = None
    for frame, (_, vector) in sorted(positions.items()):
        if not vector.any():
            continue
        if previous is None or frame - previous - 1 > MAX_GAP:
            first = frame
        previous = frame
        age = frame - first + 1
        if age >= SEQ_LEN and age % STRIDE == 0:
            window = np.stack([positions.get(i, (None, empty))[1]
                               for i in range(frame - SEQ_LEN + 1, frame + 1)])
            if np.count_nonzero(np.any(window, axis=1)) >= SEQ_LEN * 0.5:
                yield frame, window


def labeled_windows(tracks, events, label):
    from collections import defaultdict

    intervals_by_actor = defaultdict(list)
    skipped = 0
    for start, end in events:
        start_actor, end_actor = match_actor(tracks, start), match_actor(tracks, end)
        if start_actor is None or (end_actor is not None and end_actor != start_actor):
            skipped += 1
            continue
        intervals_by_actor[start_actor].append((start[0], end[0]))
    positive = []
    normal = []
    normal_before = events[0][0][0] - NORMAL_MARGIN
    for actor, intervals in intervals_by_actor.items():
        for frame, window in inference_windows(tracks[actor]):
            if any(start <= frame <= end for start, end in intervals):
                positive.append((frame, window, label))
            elif frame < normal_before:
                normal.append((frame, window, 0))
    return normal + positive, skipped


def extract(video, detector, expected, tracker_factory, encode):
    from collections import defaultdict
    import os
    import cv2

    tracks = defaultdict(dict)
    tracker = tracker_factory()
    # Windows의 영상 캡처는 한글 절대 경로에서 실패할 수 있어 파일명에 ASCII 문자를 사용한다.
    cwd = Path.cwd()
    os.chdir(video.parent)
    try:
        cap = cv2.VideoCapture(video.name)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or abs(fps - FPS) > 0.01:
                raise ValueError(f"Expected {FPS} FPS, got {fps}")
            count, width, height = expected
            batch_size = 16 if detector.device != "cpu" else 1
            for first in range(0, count, batch_size):
                frames = []
                for i in range(first, min(first + batch_size, count)):
                    success, frame = cap.read()
                    if not success or frame.shape[:2] != (height, width):
                        raise ValueError(f"Video dimensions or XML frame count differ at frame {i}")
                    frames.append(frame)
                batches = detect_batch(detector, frames)
                embeddings = embed_batch(tracker, frames, batches)
                for i, (frame, detections, embeds) in enumerate(zip(frames, batches, embeddings), first):
                    for track in tracker.update(detections, frame, embeds=embeds):
                        if track["observed"]:
                            box = track["bbox"]
                            tracks[track["track_id"]][i] = (box, encode(track["keypoints"], box))
            if cap.read()[0]:
                raise ValueError("Video has more frames than the XML")
        finally:
            cap.release()
    finally:
        os.chdir(cwd)
    return tracks


def detect_batch(detector, frames):
    results = detector.model(
        frames, conf=detector.conf_threshold, imgsz=detector.imgsz,
        device=detector.device, half=detector.half, verbose=False,
    )
    batches = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        batches.append([
            {"bbox": list(map(int, box)), "conf": float(score), "keypoints": points}
            for box, score, points in zip(boxes, scores, keypoints)
            if int(box[2]) > int(box[0]) and int(box[3]) > int(box[1])
        ])
    return batches


def embed_batch(tracker, frames, batches):
    from itertools import accumulate

    crops = []
    for frame, detections in zip(frames, batches):
        raw = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            raw.append(([x1, y1, x2 - x1, y2 - y1], detection["conf"], 0))
        frame_crops, _ = tracker.tracker.crop_bb(frame, raw)
        crops.extend(frame_crops)
    embeds = tracker.tracker.embedder.predict(crops) if crops else []
    edges = list(accumulate((len(detections) for detections in batches), initial=0))
    return [embeds[first:last] for first, last in zip(edges, edges[1:])]


def write_npz(path, **arrays):
    import numpy as np

    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def cache_key(raw: ZipInfo, label: ZipInfo, config_key):
    return sha256(f"{config_key}:{raw.filename}:{raw.CRC}:{raw.file_size}:{label.CRC}:"
                  f"{label.file_size}".encode()).hexdigest()


def prepare_clip(clip, archives, detector, tracker_cls, encode, scratch, cache, config_key):
    import numpy as np

    raw, labels = archives[clip.video_zip], archives[clip.label_zip]
    entry = raw.getinfo(clip.video)
    path = cache / (cache_key(entry, labels.getinfo(clip.xml), config_key) + ".npz")
    if not path.is_file():
        reason = ""
        windows = []
        skipped = 0
        try:
            expected, events = read_annotation(labels, clip)
            video = scratch / "clip.mp4"
            with raw.open(entry) as source, video.open("wb") as target:
                shutil.copyfileobj(source, target, length=2**20)
            tracks = extract(video, detector, expected, tracker_cls, encode)
            windows, skipped = labeled_windows(tracks, events, clip.label)
            if not any(label == clip.label for _, _, label in windows):
                reason = "No positive sequence: actor matching, tracking gap or short interval"
        except (ValueError, TypeError) as error:
            reason = str(error)
        x = np.stack([row[1] for row in windows]) if windows else np.empty((0, SEQ_LEN, 34), np.float32)
        write_npz(path, x=x, y=np.array([row[2] for row in windows], dtype=np.int64),
                  frame=np.array([row[0] for row in windows], dtype=np.int64),
                  error=reason, unmatched=skipped)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("x", "y", "frame", "error", "unmatched")}


def prepare(args, assigned, audit, duplicates):
    import cv2
    import numpy as np
    import torch

    from core.detector import HumanDetector
    from core.kpt_behavior_classifier import _encode
    from core.tracker import HumanTracker

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA unavailable. Install CUDA-enabled PyTorch or pass --allow-cpu.")
    detector = HumanDetector(str(args.pose), conf_threshold=args.confidence, imgsz=args.resolution)
    if not detector.is_pose:
        raise ValueError("--pose must point to a YOLO pose checkpoint")
    config = {"labels": LABELS, "fps": FPS, "pose": args.pose.name,
              "pose_sha256": sha256(args.pose.read_bytes()).hexdigest(),
              "resolution": args.resolution, "confidence": args.confidence,
              "sequence_length": SEQ_LEN, "stride": STRIDE, "max_gap": MAX_GAP,
              "normal_margin_frames": NORMAL_MARGIN, "device": str(detector.device),
              "dependencies": {name: version(name) for name in (
                  "torch", "torchvision", "opencv-python", "ultralytics", "deep-sort-realtime")},
              "clip_cap_per_class": args.limit_per_class}
    code = sha256()
    for file in [Path(__file__), Path(__file__).with_name("aihub_labels.py"),
                 *(ROOT / "core" / name for name in (
                     "detector.py", "tracker.py", "kpt_behavior_classifier.py"))]:
        code.update(file.read_bytes())
    config["preprocessing_sha256"] = code.hexdigest()
    cache_config = {key: value for key, value in config.items() if key != "clip_cap_per_class"}
    config_key = sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()
    cache = args.cache / config_key
    cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    stats, rejected = {}, {}
    embedder = HumanTracker().tracker.embedder

    def new_tracker():
        return HumanTracker(embedder=embedder)

    with ExitStack() as stack:
        scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="aihub-", dir=args.cache)))
        paths = {path for clips in assigned.values() for clip in clips
                 for path in (clip.label_zip, clip.video_zip)}
        archives = {path: stack.enter_context(ZipFile(path)) for path in paths}
        for split, clips in assigned.items():
            if not clips:
                continue
            arrays = {key: [] for key in ("x", "y", "groups", "clips", "frames")}
            kept = Counter()
            for i, clip in enumerate(clips, 1):
                if split != "diagnostic" and args.limit_per_class and kept[clip.label] >= args.limit_per_class:
                    continue
                data = prepare_clip(clip, archives, detector, new_tracker, _encode, scratch, cache, config_key)
                if data["error"] or data["unmatched"]:
                    rejected[clip.name] = {"reason": str(data["error"]), "events_without_actor": int(data["unmatched"])}
                if (data["y"] == clip.label).any():
                    kept[clip.label] += 1
                count = len(data["y"])
                for key, value in zip(arrays, (data["x"], data["y"], np.repeat(clip.group, count),
                                               np.repeat(clip.name, count), data["frame"])):
                    arrays[key].append(value)
                if i % 20 == 0:
                    print(f"{split}: through clip {i}/{len(clips)}", flush=True)
            arrays = {key: np.concatenate(values) for key, values in arrays.items()}
            counts = Counter(LABELS[index] for index in arrays["y"])
            stats[split] = {"windows": dict(counts), "positive_clips": {
                LABELS[index]: count for index, count in kept.items()}}
            print(f"{split}: {sum(kept.values())} clips with positive windows", flush=True)
            write_npz(args.output / f"{split}.npz", **arrays)
    (args.output / "preparation-report.json").write_text(json.dumps({
        "archive_pairs": audit, "splits": stats, "review": rejected, "duplicate_groups_excluded": duplicates,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    for split in SPLITS[:3]:
        missing = set(LABELS) - stats.get(split, {}).get("windows", {}).keys()
        if missing:
            raise ValueError(f"{split}: no sequences for {sorted(missing)}. See preparation-report.json.")
    (args.output / "dataset.json").write_text(json.dumps({
        **config, "counts": stats, "normal_labels": "pre-event proxies",
        "group_strategy": "scenario, filming location, date",
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=Path("pose-cache"))
    parser.add_argument("--pose", type=Path, default=ROOT / "yolo11n-pose.pt")
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.limit_per_class is not None and args.limit_per_class < 1:
        parser.error("--limit-per-class must be positive")
    if args.resolution < 1 or not 0 < args.confidence <= 1:
        parser.error("Invalid resolution or confidence")
    args.data_root = args.data_root.absolute()
    if not args.data_root.is_dir():
        parser.error(f"Data directory not found: {args.data_root}")
    args.pose = args.pose.absolute()
    if not args.audit_only and not args.pose.is_file():
        parser.error(f"Pose checkpoint not found: {args.pose}")
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    clips, audit = index_archives(args.data_root)
    examined = json.loads((ROOT / "docs/design/raw-video-test-set.json").read_text(encoding="utf-8"))
    assigned = split_clips(clips, [item["file"] for item in examined])
    duplicates = remove_cross_split_duplicates(assigned)
    print(json.dumps(inventory(assigned), ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "archives": audit, "assigned_clips": inventory(assigned), "annotations": audit_labels(assigned),
            "duplicate_groups_excluded": duplicates,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        if (args.output / "dataset.json").exists():
            parser.error("Dataset already completed. Choose a new output folder.")
        sys.path.insert(0, str(ROOT))
        prepare(args, assigned, audit, duplicates)


if __name__ == "__main__":
    main()
