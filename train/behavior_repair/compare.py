"""기존 행동 모델과 개선 가중치를 동일한 평가 시퀀스에서 비교한다. 추론 설정을 고정하고 예측 점수와 사건 구간의 차이를 집계한다.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch

from aihub_labels import LABELS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from backend.behavior_upload import inspect_checkpoint


@torch.inference_mode()
def predict(path, sequences):
    model = TrackBehaviorBuffer._build_and_load(path).eval()
    return np.concatenate([model(torch.from_numpy(batch)).argmax(dim=1).numpy()
                           for batch in np.array_split(sequences, (len(sequences) + 127) // 128)])


def scores(y, prediction):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(matrix, (y, prediction), 1)
    metrics = {}
    for i, label in enumerate(LABELS):
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        hits = int(matrix[i, i])
        metrics[label] = {
            "support": support,
            "precision": hits / predicted if predicted else None,
            "recall": hits / support if support else None,
            "f1": 2 * hits / (support + predicted) if support else None,
        }
    present = [row["f1"] for row in metrics.values() if row["f1"] is not None]
    return {"accuracy": float(matrix.trace() / matrix.sum()), "macro_f1": sum(present) / len(present),
            "confusion_matrix": matrix.tolist(), "per_class": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.parent.name == "aihub-rtx3070" and (ROOT.parent / "skip-full-comparison").is_file():
        print("Baseline comparison canceled.", flush=True)
        return
    print("Web checkpoint validated:", inspect_checkpoint(args.model), flush=True)
    behavior_dir = ROOT / "train/1_behavior"
    checkpoints = {
        "previous_v2": behavior_dir / "output/kpt_behavior.pth",
        "previous_v1": behavior_dir / "kpt_behavior.pth",
        "retrained": args.model,
    }
    torch.set_num_threads(2)
    report = {"labels": LABELS, "unit": "30-frame window", "checkpoint_sha256": {
        name: sha256(path.read_bytes()).hexdigest() for name, path in checkpoints.items()
    }}
    for split in ("test", "diagnostic"):
        path = args.dataset / f"{split}.npz"
        if not path.is_file() and split == "diagnostic":
            continue
        with np.load(path, allow_pickle=False) as archive:
            x = archive["x"].astype(np.float32)
            y, clips = archive["y"], archive["clips"]
        if not len(x):
            continue
        predictions = {name: predict(path, x) for name, path in checkpoints.items()}
        report[split] = {"clips": len(set(clips)), **{
            name: scores(y, pred) for name, pred in predictions.items()
        }}
        if split == "diagnostic":
            # 절도 주석 구간 안에서 끝나는 추론 창만 집계한다.
            report[split]["theft_event_windows_by_clip"] = {
                clip: {name: dict(Counter(LABELS[i] for i in prediction[(clips == clip) & (y == 6)]))
                       for name, prediction in predictions.items()}
                for clip in sorted(set(clips))
            }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    test = report["test"]
    for name in checkpoints:
        print(f"{name} test macro-F1: {test[name]['macro_f1']:.4f}")
        for label in ("fall", "smoking", "theft"):
            print(label, test[name]["per_class"][label])


if __name__ == "__main__":
    main()
