"""YOLO11m과 혼합 포즈 입력을 사용하는 손 융합 분류기를 미세조정한다. 포즈 시퀀스와 손 영역을 결합한 자료로 재학습한다.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aihub_labels import LABELS
from config import ROOT
from prepare_aihub import inference_windows
from prepare_hand import atomic_json
from train_hand import windows as legacy_windows

sys.path.insert(0, str(ROOT))
from core.hand_behavior import INPUT_SIZE
from core.kpt_behavior_classifier import TrackBehaviorBuffer


def load_features(folder, row):
    with np.load(folder / row["path"], allow_pickle=False) as cached:
        tracks = defaultdict(dict)
        for (frame, tid), box, vector in zip(cached["ids"], cached["boxes"], cached["x"]):
            if vector[:34].any():
                tracks[int(tid)][int(frame)] = (box, vector)
        return tracks, cached["events"].tolist()


def actor_intervals(events, tracks):
    intervals = defaultdict(list)
    for actor, start, end, end_actor in events:
        if actor >= 0:
            intervals[actor].append((start, end))
        elif end_actor >= 0 and len(tracks[end_actor]) >= 30:
            intervals[end_actor].append((start, end))
    return intervals


def samples_for_track(positions, stride):
    first = previous = None
    for frame in sorted(positions):
        if previous is None or frame - previous > 4:
            first = frame
        previous = frame
        age = frame - first + 1
        if age < 30 or age % stride:
            continue
        indexes = range(frame - 29, frame + 1)
        if sum(i in positions for i in indexes) < 18:
            continue
        yield frame


def sequence(positions, frame):
    empty = np.zeros(INPUT_SIZE, np.float16)
    return np.stack([positions[i][1] if i in positions else empty for i in range(frame - 29, frame + 1)])


def distant_bystander(frame, positions, actor_positions):
    comparable = []
    for i in range(frame - 29, frame + 1):
        if i not in positions or i not in actor_positions:
            continue
        b = positions[i][0]
        actor = actor_positions[i][0]
        center_offset = (b[:2] + b[2:] - actor[:2] - actor[2:]) / 2
        comparable.append(np.linalg.norm(center_offset) >= .90 * max(1., actor[3] - actor[1]))
    return len(comparable) >= 24 and all(comparable)


def training_windows(folder, rows):
    inputs, labels = [], []
    negative_counts = Counter()
    for row in rows:
        tracks, events = load_features(folder, row)
        aligned = actor_intervals(events, tracks)
        normal_end = min(e[1] for e in events) - 9
        possible_actors = {e[i] for e in events for i in (0, 3) if e[i] >= 0}
        for tid, positions in tracks.items():
            stride = 3 if tid in aligned else 12
            for frame in samples_for_track(positions, stride):
                if any(begin <= frame <= end for begin, end in aligned.get(tid, ())):
                    label = row["label"]
                elif frame < normal_end and (tid in aligned or tid not in possible_actors):
                    label = 0
                    negative_counts["before_event"] += 1
                elif row["label"] in (1, 4, 5, 6) and tid not in possible_actors and len(aligned) == 1:
                    actor = next(iter(aligned))
                    if not any(begin <= frame <= end for begin, end in aligned[actor]):
                        continue
                    if not distant_bystander(frame, positions, tracks[actor]):
                        continue
                    label = 0
                    negative_counts["distant_bystander"] += 1
                else:
                    continue
                inputs.append(sequence(positions, frame))
                labels.append(label)
    return np.stack(inputs), np.array(labels), negative_counts


def validation_windows(folder, rows):
    inputs, index = [], []
    expected = []
    for cid, row in enumerate(rows):
        tracks, events = load_features(folder, row)
        expected.append((row["label"], events))
        for tid, positions in tracks.items():
            poses = {i: (None, vector[:34]) for i, (_, vector) in positions.items()}
            for i, _ in inference_windows(poses):
                inputs.append(sequence(positions, i))
                index.append((cid, tid, i))
    return torch.from_numpy(np.stack(inputs)), index, expected


@torch.inference_mode()
def validation_metrics(model, validation, temperatures):
    model.eval()
    inputs, index, expected = validation
    logits = torch.cat([model(batch.cuda().float()).cpu() for batch in inputs.split(96)])
    result = []
    for temperature in temperatures:
        probabilities = (logits / temperature).softmax(1).numpy()
        ema = {}
        detected = set()
        incorrect = set()
        total = Counter(label for label, _ in expected)
        for (clip, tid, frame), score in zip(index, probabilities):
            key = clip, tid
            prev = ema.get(key)
            if prev and frame - prev[0] <= 8:
                score = .6 * prev[1] + .4 * score
            ema[key] = (frame, score)
            label = int(score.argmax())
            if label == 0 or score[label] < .90:
                continue
            truth, events = expected[clip]
            if label == truth and any(tid == actor and begin <= frame <= end for actor, begin, end, _ in events):
                detected.add((clip, label))
            else:
                incorrect.add((clip, label))
        true_counts = Counter(label for _, label in detected)
        false_counts = Counter(label for _, label in incorrect)
        class_f1 = {label: 2 * true_counts[label] / (total[label] + true_counts[label] + false_counts[label])
                    for label in total}
        macro = sum(class_f1.values()) / len(class_f1)
        result.append({"temperature": temperature, "macro_f1": macro, "detected": len(detected),
                       "class_f1": class_f1, "false_clip_by_class": dict(false_counts),
                       "detected_by_class": dict(true_counts)})
    return result


def save_calibrated(model, metadata, temperature, target):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    for name in ("pose.head.3", "residual.1"):
        for suffix in ("weight", "bias"):
            state[f"{name}.{suffix}"].div_(temperature)
    partial = target.with_suffix(".part")
    torch.save({"state_dict": state, "input_config": metadata}, partial)
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--legacy-cache", required=True, type=Path)
    parser.add_argument("--medium-cache", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--focus", action="store_true", help="Add weight to destruction and abandonment.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rows = json.loads((args.medium_cache / "manifest-train.json").read_text("utf8"))
    train, val = ([row for row in rows if row["split"] == name] for name in ("train", "val"))
    held_out = {row["group"] for row in val}
    old_rows = json.loads((args.legacy_cache / "manifest.json").read_text("utf8"))
    old_train = [row for row in old_rows if row["split"] == "train" and row["group"] not in held_out]
    x_m, y_m, negatives = training_windows(args.medium_cache, train)
    old = legacy_windows(args.legacy_cache, old_train)
    inputs = torch.cat((torch.from_numpy(x_m), old.tensors[0]))
    labels = torch.cat((torch.from_numpy(y_m), old.tensors[1]))
    counts = labels.bincount(minlength=8)
    if not torch.all(counts > 0) or set(row["label"] for row in val) != set(range(1, 8)):
        raise ValueError("Training and calibration must cover all seven abnormal behaviors.")
    train_loader = DataLoader(TensorDataset(inputs, labels), batch_size=96, shuffle=True)
    val_data = validation_windows(args.medium_cache, val)
    model = TrackBehaviorBuffer._build_and_load(args.base).cuda()
    metadata = {**model.behavior_input_config, "pose_models": ["yolo11n-pose.pt", "yolo11m-pose.pt"],
                "training": "mixed_pose_and_bystanders", "resolution": 480, "detection_confidence": .60}
    weights = (counts.float().median()/counts.float()).sqrt().cuda()
    if args.focus:
        weights[[2, 5]] *= 1.30
        metadata["focused_classes"] = ["vandalism", "abandon"]
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=.015)
    optimizer = torch.optim.AdamW([
        {"params": model.pose.parameters(), "lr": 1.5e-4},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("pose.")], "lr": 4e-4},
    ], weight_decay=.01)
    baseline = validation_metrics(model, val_data, (1.,))[0]
    best = -1.
    epochs = []
    name = "hand_fusion_medium_focused_3fps" if args.focus else "hand_fusion_medium_3fps"
    out = args.medium_cache / (name + ".pth")
    report_path = args.medium_cache / ("training-focused.json" if args.focus else "training.json")
    report = {"baseline": baseline, "medium_training_clips": len(train), "legacy_training_clips": len(old_train),
              "medium_validation_clips": len(val), "windows_by_class": dict(zip(LABELS, counts.tolist())),
              "negative_windows": dict(negatives), "epochs": epochs}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda().float(), y.cuda()
            x[torch.rand(len(x), device=x.device) < .08, :, 34:] = 0
            optimizer.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        metrics = validation_metrics(model, val_data, (.90, 1., 1.15))
        def rank(metric):
            score = metric["macro_f1"] - .002 * sum(metric["false_clip_by_class"].values())
            if args.focus:
                score += .06 * (metric["class_f1"][2] + metric["class_f1"][5])
            return score

        choice = max(metrics, key=rank)
        epochs.append({"epoch": epoch, "calibration": metrics})
        ranking = rank(choice)
        if ranking > best:
            best = ranking
            save_calibrated(model, metadata, choice["temperature"], out)
            report["selection"] = {"epoch": epoch, **choice}
        atomic_json(report_path, report)
        print(epoch, "/", args.epochs, choice, flush=True)
    artifact = subprocess.run(
        [sys.executable, "-m", "backend.behavior_upload", str(out.resolve())],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    report["upload_kind"] = artifact.stdout.strip()
    atomic_json(report_path, report)

if __name__ == "__main__":
    main()
