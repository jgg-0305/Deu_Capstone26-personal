# -*- coding: utf-8 -*-
"""키포인트 학습 파이프라인 공용 유틸.

추출(extract_kpts_video) · 온셋 학습(demo_onset) · 평가(evaluate_models)가
공유하는 상수/함수를 한 곳에 모은다. (이전에는 스크립트마다 중복 정의되어 있었음)
"""

import os

import cv2
import numpy as np

# ── 공용 상수 ────────────────────────────────────────────────────────────────
SEQ_LEN = 30            # 슬라이딩 윈도우 길이(프레임)
STRIDE = 10             # 윈도우 이동 간격
KPT_CONF_THR = 0.3      # 이 값 미만 관절 → (0, 0)
MIN_VALID_FRAC = 0.5    # 윈도우 내 인물 검출(비-제로) 프레임 비율 최소값

LABEL_NAMES = ["정상", "전도", "파손", "방화", "흡연", "유기", "절도", "폭행"]

# split/data 폴더 접두 → 라벨. 01/04/05 는 모두 정상(0).
FOLDER_LABEL_MAP = {
    "01": 0, "04": 0, "05": 0,
    "07": 1, "08": 2, "09": 3, "10": 4, "11": 5, "12": 6, "13": 7,
}


# ── 키포인트 인코딩 ──────────────────────────────────────────────────────────
def encode_kpts(kpts_xy, bbox) -> np.ndarray:
    """(17, 2) 좌표 + bbox[x1,y1,x2,y2] → (34,) bbox 상대 정규화 벡터."""
    x1, y1, x2, y2 = bbox
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    vec = np.zeros(34, dtype=np.float32)
    for i, (kx, ky) in enumerate(kpts_xy):
        vec[i * 2] = (kx - x1) / bw
        vec[i * 2 + 1] = (ky - y1) / bh
    return vec


def run_yolo_on_frame(model, img_bgr):
    """YOLO pose 추론 → 가장 넓은 bbox 의 (keypoints_xy, bbox). 없으면 None."""
    results = model(img_bgr, verbose=False)
    best_area, best = -1, None
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                kpts = r.keypoints[i].data[0].cpu().numpy()  # (17, 3)
                kpts_xy = np.array(
                    [[k[0], k[1]] if k[2] >= KPT_CONF_THR else [0.0, 0.0]
                     for k in kpts],
                    dtype=np.float32,
                )
                best_area = area
                best = (kpts_xy, [x1, y1, x2, y2])
    return best


def video_to_vecs(video_path, yolo_model, progress=False):
    """영상 모든 프레임 → (T, 34) 벡터 리스트. 사람 미검출 프레임은 0 벡터."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [오류] 열기 실패: {video_path}")
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vecs, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        r = run_yolo_on_frame(yolo_model, frame)
        vecs.append(encode_kpts(*r) if r else np.zeros(34, dtype=np.float32))
        idx += 1
        if progress and idx % 200 == 0:
            print(f"    {idx}/{total} 프레임...", end="\r", flush=True)
    cap.release()
    return vecs


def window_valid(window) -> bool:
    """윈도우 내 인물 검출(비-제로) 프레임 비율이 기준 이상인지."""
    return np.mean([not np.all(f == 0) for f in window]) >= MIN_VALID_FRAC


def make_windows(vecs, label, seq_len=SEQ_LEN, stride=STRIDE):
    """벡터 리스트 → 품질 통과 윈도우와 라벨. (단일 라벨 부여)"""
    seqs, labels = [], []
    for s in range(0, len(vecs) - seq_len + 1, stride):
        win = np.stack(vecs[s:s + seq_len])
        if window_valid(win):
            seqs.append(win)
            labels.append(label)
    return seqs, labels


def merge_npz(path, new_seqs, new_labels):
    """기존 npz 에 시퀀스를 병합 저장하고 총 개수를 반환."""
    if len(new_seqs) == 0:
        return len(np.load(path)["sequences"]) if os.path.exists(path) else 0
    new_arr = np.stack(new_seqs).astype(np.float32)
    new_lbl = np.array(new_labels, dtype=np.int64)
    if os.path.exists(path):
        d = np.load(path)
        new_arr = np.concatenate([d["sequences"], new_arr], axis=0)
        new_lbl = np.concatenate([d["labels"], new_lbl], axis=0)
    np.savez_compressed(path, sequences=new_arr, labels=new_lbl)
    return len(new_arr)


def load_yolo(model_path="yolo11n-pose.pt"):
    from ultralytics import YOLO
    return YOLO(model_path)
