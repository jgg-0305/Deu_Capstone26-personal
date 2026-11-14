"""실행 환경과 같은 포즈·추적 입력으로 라벨 구간의 시퀀스를 만든다."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.detector import HumanDetector
from core.kpt_behavior_classifier import _encode
from core.tracker import HumanTracker

SEQ_LEN = 30
STRIDE = 5
SPLITS = ("train", "val", "test")
LABELS = ("normal", "fall", "vandalism", "arson", "smoking", "abandon", "theft", "assault")


def extract(video, detector):
    tracks = defaultdict(dict)
    tracker = HumanTracker()
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"영상 또는 FPS를 읽을 수 없습니다. {video}")
        index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            for track in tracker.update(detector.detect(frame), frame):
                if track["observed"]:
                    box = track["bbox"]
                    tracks[track["track_id"]][index] = (box, _encode(track["keypoints"], box))
            index += 1
    finally:
        cap.release()
    if not index:
        raise ValueError(f"영상 프레임을 읽을 수 없습니다. {video}")
    return fps, index, tracks


def iou(left, right):
    x = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    y = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    union = (left[2]-left[0]) * (left[3]-left[1]) + (right[2]-right[0]) * (right[3]-right[1]) - x*y
    return x*y / union if union > 0 else 0


def segment_windows(tracks, row, frame_count):
    first, stop, anchor = (int(row[key]) for key in ("start", "stop", "anchor"))
    context = int(row.get("context_start") or first)
    box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
    if not 0 <= context <= first <= anchor < stop <= frame_count or not all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"프레임 범위 또는 대상 상자를 확인하세요. {row}")
    matches = sorted(((iou(box, frames[anchor][0]), tid) for tid, frames in tracks.items()
                      if anchor in frames), reverse=True)
    if not matches or matches[0][0] < .5 or (len(matches) > 1 and matches[0][0] - matches[1][0] < .1):
        raise ValueError(f"대상 인물을 확정할 수 없습니다. 앵커 프레임과 상자를 확인하세요. {row}")
    frames = tracks[matches[0][1]]
    empty = np.zeros(34, dtype=np.float32)
    for end in range(max(first, context + SEQ_LEN - 1), stop, STRIDE):
        window = np.stack([frames.get(i, (None, empty))[1] for i in range(end - SEQ_LEN + 1, end + 1)])
        observed = np.any(window != 0, axis=1)
        if observed[-1] and observed.sum() >= 15 and "0000" not in "".join(str(int(v)) for v in observed):
            yield window


def prepare(args, detector):
    videos = defaultdict(list)
    ownership = {}
    file_splits = {}
    labels_by_split = defaultdict(set)
    with args.labels.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split, group, label = row["split"], row["group"].strip(), row["label"]
            if split not in SPLITS or not group or label not in LABELS:
                raise ValueError(f"분할, 촬영 그룹 또는 행동 라벨을 확인하세요. {row}")
            path = (args.data_root / row["video"]).resolve(strict=True)
            if not path.is_relative_to(args.data_root.resolve()):
                raise ValueError(f"데이터 폴더 밖 영상입니다. {path}")
            if ownership.setdefault(group, split) != split or file_splits.setdefault(path, split) != split:
                raise ValueError(f"한 촬영 그룹 또는 영상이 여러 분할에 포함됐습니다. {group}")
            labels_by_split[split].add(label)
            videos[path].append(row)
    for split in SPLITS:
        missing = set(LABELS) - labels_by_split[split]
        if missing:
            raise ValueError(f"{split}에 필요한 라벨이 없습니다. {', '.join(sorted(missing))}")
    hashes = {}
    for video, rows in videos.items():
        with video.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in hashes:
            raise ValueError(f"영상 내용이 같은 복사본이 있습니다. {video} / {hashes[digest]}")
        hashes[digest] = video
        intervals = sorted((int(row["start"]), int(row["stop"])) for row in rows)
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError(f"한 영상의 라벨 구간이 겹칩니다. {video}")

    samples = defaultdict(list)
    targets = defaultdict(list)
    groups = defaultdict(list)
    rates = set()
    for video, rows in videos.items():
        fps, total, tracks = extract(video, detector)
        rates.add(round(fps, 3))
        for row in rows:
            windows = list(segment_windows(tracks, row, total))
            if not windows:
                raise ValueError(f"유효한 30프레임 구간이 없습니다. {row}")
            split = row["split"]
            samples[split].extend(windows)
            targets[split].extend([LABELS.index(row["label"])] * len(windows))
            groups[split].extend([row["group"]] * len(windows))
        print(video.name, fps, total, flush=True)
    if len(rates) != 1:
        raise ValueError(f"학습·검증 영상의 FPS를 통일하세요. 발견한 값 {sorted(rates)}")
    for split in SPLITS:
        if set(targets[split]) != set(range(len(LABELS))):
            raise ValueError(f"{split}에 8개 행동 클래스의 유효한 시퀀스가 필요합니다. {Counter(targets[split])}")
    args.output.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        np.savez_compressed(args.output / f"{split}.npz", x=np.stack(samples[split]),
                            y=np.asarray(targets[split], dtype=np.int64), groups=np.asarray(groups[split]))
    metadata = {"labels": LABELS, "fps": rates.pop(), "pose": args.pose.name,
                "resolution": 480, "confidence": .4, "sequence_length": SEQ_LEN,
                "counts": {split: dict(Counter(targets[split])) for split in SPLITS}}
    (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="라벨 없이 포즈·추적 입력만 검사할 원본 영상")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.video and (not args.labels or not args.data_root or not args.output):
        parser.error("학습 입력에는 --labels, --data-root, --output이 필요합니다.")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    if not args.pose.is_file():
        parser.error("프로젝트의 포즈 가중치 경로를 확인하세요.")
    detector = HumanDetector(str(args.pose), conf_threshold=.4, imgsz=480)
    def limit_threads(predictor):
        if predictor.device.type == "cpu":
            torch.set_num_threads(2)
    detector.model.add_callback("on_predict_start", limit_threads)
    if args.video:
        fps, total, tracks = extract(args.video, detector)
        print(json.dumps({"fps": fps, "frames": total,
                          "observations_per_track": {tid: len(frames) for tid, frames in tracks.items()}}))
    else:
        prepare(args, detector)


if __name__ == "__main__":
    main()
