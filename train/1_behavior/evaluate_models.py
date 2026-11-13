# -*- coding: utf-8 -*-
"""두 행동분류 모델 성능 평가 + 논문용 성능평가표(docx) 생성.

평가 대상:
  1) kpt_behavior.pth  — KeypointLSTMv2 (키포인트 30프레임 시퀀스)  / kpt_val.npz 검증셋
  2) model_final.pth   — LSTM_NIA (ResNet50+LSTM, RGB 3프레임)      / test.csv 테스트셋

산출 지표(실측):
  - Accuracy, Macro/Weighted Precision·Recall·F1  (혼동행렬 기반, numpy 직접 계산)
  - 평균 추론 지연(ms/sample) 및 처리량(samples/s)  — GPU / CPU 각각

실행:
    .\.venv\Scripts\python.exe train\1_behavior\evaluate_models.py
출력:
    AI_성능평가표.docx  (프로젝트 루트)
"""

import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model_kpt import KeypointLSTM, KeypointLSTMv2
from dataset_kpt import KptBehaviorDataset, LABEL_NAMES

NUM_CLASSES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────
# 지표 계산 (혼동행렬 기반, sklearn 불필요)
# ──────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    """혼동행렬로부터 accuracy / per-class P,R,F1 / macro·weighted 평균 계산."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    support = cm.sum(axis=1)                       # 클래스별 실제 표본 수
    accuracy = np.trace(cm) / cm.sum() if cm.sum() else 0.0

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0.0)

    # macro: 등장한(support>0) 클래스에 대해서만 단순 평균 (sklearn 기본과 동일하게 전체 라벨 평균도 병기)
    present = support > 0
    macro = {
        "precision": float(precision[present].mean()) if present.any() else 0.0,
        "recall": float(recall[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
    }
    total = support.sum()
    weighted = {
        "precision": float((precision * support).sum() / total) if total else 0.0,
        "recall": float((recall * support).sum() / total) if total else 0.0,
        "f1": float((f1 * support).sum() / total) if total else 0.0,
    }
    return {
        "accuracy": float(accuracy),
        "macro": macro,
        "weighted": weighted,
        "per_class": {"precision": precision, "recall": recall, "f1": f1,
                      "support": support},
        "confusion": cm,
        "n_samples": int(total),
        "n_present_classes": int(present.sum()),
    }


# ──────────────────────────────────────────────────────────────────────────
# 추론 지연 측정 (모델 forward 순수 시간)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_latency(model, sample_input, device, warmup=10, iters=50):
    """배치=1 단일 추론의 평균 지연(ms)과 처리량(samples/s) 측정."""
    model = model.to(device).eval()
    x = sample_input.to(device)
    is_cuda = device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
    for _ in range(warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / iters * 1000.0
    return ms, 1000.0 / ms


# ──────────────────────────────────────────────────────────────────────────
# 1) 키포인트 모델 평가
# ──────────────────────────────────────────────────────────────────────────
def load_kpt_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt if not isinstance(ckpt, dict) or "state_dict" not in ckpt else ckpt["state_dict"]
    from collections import OrderedDict
    state = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                        for k, v in state.items())
    model = KeypointLSTMv2() if "attn.weight" in state else KeypointLSTM()
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, type(model).__name__


@torch.no_grad()
def evaluate_kpt(weights, npz_path):
    print(f"\n[1] 키포인트 모델 평가: {os.path.basename(weights)}")
    model, n_params, arch = load_kpt_model(weights)
    model = model.to(DEVICE).eval()
    ds = KptBehaviorDataset(npz_path, augment=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    print(f"    아키텍처 {arch}  파라미터 {n_params:,}  검증표본 {len(ds)}")

    y_true, y_pred = [], []
    for seq, label in loader:
        out = model(seq.to(DEVICE))
        y_pred.extend(out.argmax(dim=1).cpu().tolist())
        y_true.extend(label.tolist() if torch.is_tensor(label) else list(label))
    m = compute_metrics(y_true, y_pred)

    sample = ds[0][0].unsqueeze(0)  # (1, T, 34)
    gpu_ms = gpu_tp = None
    if torch.cuda.is_available():
        gpu_ms, gpu_tp = measure_latency(model, sample, "cuda")
    cpu_ms, cpu_tp = measure_latency(model, sample, "cpu")

    m.update({"arch": arch, "n_params": n_params, "model_file": weights,
              "eval_set": os.path.basename(npz_path), "epoch": 80,
              "gpu_ms": gpu_ms, "gpu_tp": gpu_tp, "cpu_ms": cpu_ms, "cpu_tp": cpu_tp})
    return m


# ──────────────────────────────────────────────────────────────────────────
# 2) 데모 영상 온셋 감지 평가 — 앱과 동일한 추론 경로(헤드리스)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_onset(kpt_weights, object_weights, demo_onset, conf=0.4, imgsz=480):
    """각 데모 영상에서 목표 이상행동이 처음 감지되는 프레임을 측정해
    요청 온셋과 비교한다. 앱(VideoThread)과 동일한 컴포넌트를 직접 구동."""
    from core.detector import HumanDetector, ObjectDetector
    from core.tracker import HumanTracker
    from core.kpt_behavior_classifier import TrackBehaviorBuffer
    from core.theft_monitor import TheftMonitor
    from core.video_thread import _match_keypoints, _summarize_behavior
    import cv2

    rows = []
    for fname, (target, onset) in demo_onset.items():
        path = os.path.join(_ROOT, fname)
        if not os.path.isfile(path):
            rows.append({"file": fname, "target": target, "onset_req": onset,
                         "onset_det": None, "note": "파일 없음"})
            continue

        detector = HumanDetector("yolo11n-pose.pt", conf, imgsz=imgsz)
        tracker = HumanTracker(30)
        buf = TrackBehaviorBuffer(kpt_weights)
        obj_det = theft = None
        if object_weights and os.path.isfile(object_weights):
            obj_det = ObjectDetector(object_weights, conf_threshold=0.25, imgsz=imgsz)
            if 1 in getattr(obj_det, "names", {}):
                theft = TheftMonitor()

        cap = cv2.VideoCapture(path)
        fi, detected = 0, None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            tracks = tracker.update(dets, frame)
            kpt_map = _match_keypoints(tracks, dets)
            bmap = buf.update(tracks, kpt_map)
            if theft is not None:
                objs = obj_det.detect(frame)
                for tid in theft.update(tracks, kpt_map, objs):
                    if bmap.get(tid, (-1, 0))[0] in (-1, 0):
                        bmap[tid] = (6, 0.9)
            cid, _ = _summarize_behavior(bmap)
            if detected is None and cid == target:
                detected = fi
            fi += 1
        cap.release()
        delta = (detected - onset) if detected is not None else None
        rows.append({"file": fname, "target": target, "onset_req": onset,
                     "onset_det": detected, "total": fi,
                     "delta": delta, "note": ""})
        ds = "미감지" if detected is None else f"{detected}f (Δ{delta:+d})"
        print(f"  [{fname}] 목표 {LABEL_NAMES[target]}  요청 {onset}f → 감지 {ds}")
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 콘솔 요약 출력
# ──────────────────────────────────────────────────────────────────────────
def print_summary(name, m):
    print(f"\n===== {name} =====")
    print(f"  표본 {m['n_samples']}  (등장 클래스 {m['n_present_classes']}/{NUM_CLASSES})")
    print(f"  Accuracy        : {m['accuracy']*100:.2f}%")
    print(f"  Macro    P/R/F1 : {m['macro']['precision']*100:.2f} / "
          f"{m['macro']['recall']*100:.2f} / {m['macro']['f1']*100:.2f}")
    print(f"  Weighted P/R/F1 : {m['weighted']['precision']*100:.2f} / "
          f"{m['weighted']['recall']*100:.2f} / {m['weighted']['f1']*100:.2f}")
    if m["gpu_ms"]:
        print(f"  추론(GPU)        : {m['gpu_ms']:.2f} ms/sample  ({m['gpu_tp']:.1f} samples/s)")
    print(f"  추론(CPU)        : {m['cpu_ms']:.2f} ms/sample  ({m['cpu_tp']:.1f} samples/s)")
    pc = m["per_class"]
    print("  클래스별 F1:")
    for i, nm in enumerate(LABEL_NAMES):
        if pc["support"][i] > 0:
            print(f"    {nm:>4}: P {pc['precision'][i]*100:5.1f}  "
                  f"R {pc['recall'][i]*100:5.1f}  F1 {pc['f1'][i]*100:5.1f}  "
                  f"(n={int(pc['support'][i])})")


def main():
    print(f"device: {DEVICE}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    kpt_weights = os.path.join(_HERE, "output", "kpt_behavior.pth")
    object_weights = os.path.join(_ROOT, "model", "objects.pt")

    # 1) 키포인트 행동 분류 지표 (held-out kpt_val.npz)
    res_kpt = evaluate_kpt(kpt_weights, os.path.join(_HERE, "split", "kpt_val.npz"))
    print_summary("KeypointLSTMv2 (kpt_behavior.pth)", res_kpt)

    # 2) 데모 영상 온셋 감지 평가 (앱 추론 경로)
    from demo_onset import DEMO_ONSET
    print("\n[2] 데모 영상 온셋 감지 평가")
    onset_rows = evaluate_onset(kpt_weights, object_weights, DEMO_ONSET)

    # docx 생성
    from eval_report import build_report
    out = os.path.join(_ROOT, "AI_성능평가표.docx")
    build_report(out, res_kpt, onset_rows, DEVICE, LABEL_NAMES)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
