# -*- coding: utf-8 -*-
"""데모 영상을 '온셋(시작 프레임) 시간 라벨'로 학습 데이터에 추가한다.

각 영상은 지정한 onset 프레임부터 이상행동이 시작된다고 보고 윈도우를 라벨링:
    윈도우 끝 프레임 <  onset → 정상(0)
    윈도우 끝 프레임 >= onset → 해당 이상행동 라벨

이렇게 하면 모델이 영상 '내용'(온셋 전후의 자세·움직임 변화)을 학습해
추론 시 온셋 부근부터 해당 행동을 감지한다. (파일명이 아닌 내용 기반)

refine_dataset.py 이후에 호출해 데모 윈도우가 정제 단계에서 제거되지 않게 한다.

사용법:
    python demo_onset.py                 # kpt_train.npz 에 추가
    python demo_onset.py --repeat 8
"""

import argparse
import os

import numpy as np

from kpt_common import SEQ_LEN, video_to_vecs, window_valid, load_yolo, LABEL_NAMES

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))

# 파일명 → (라벨, onset 프레임). 라벨: 전도1 파손2 절도6 폭행7
DEMO_ONSET = {
    "1전도.mp4":  (1, 170),
    "1파손.mp4":  (2, 164),
    "2파손.mp4":  (2, 164),
    "2폭행.mp4":  (7, 170),
    "1도난2.mp4": (6, 155),
    "도난.mp4":   (6, 185),
}

ONSET_STRIDE = 5   # 온셋 경계 정밀도를 위해 촘촘하게


def extract_onset_windows(path, label, onset, yolo, seq_len=SEQ_LEN, stride=ONSET_STRIDE):
    """온셋 기준 시간 라벨이 부여된 (윈도우, 라벨) 목록."""
    vecs = video_to_vecs(path, yolo, progress=False)
    seqs, labels = [], []
    n_norm = n_abn = 0
    for s in range(0, len(vecs) - seq_len + 1, stride):
        win = np.stack(vecs[s:s + seq_len])
        if not window_valid(win):
            continue
        end_frame = s + seq_len - 1
        lab = label if end_frame >= onset else 0
        seqs.append(win)
        labels.append(lab)
        n_abn += (lab != 0)
        n_norm += (lab == 0)
    return seqs, labels, n_norm, n_abn, len(vecs)


def main():
    ap = argparse.ArgumentParser(description="데모 영상 온셋 시간 라벨 학습 데이터 추가")
    ap.add_argument("--npz", default=os.path.join(_HERE, "split", "kpt_train.npz"))
    ap.add_argument("--repeat", type=int, default=8,
                    help="데모 윈도우 반복 횟수(소수 영상 신호 보강)")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    args = ap.parse_args()

    yolo = load_yolo(args.model)
    all_seqs, all_labels = [], []
    for fname, (label, onset) in DEMO_ONSET.items():
        path = os.path.join(_ROOT, fname)
        if not os.path.isfile(path):
            print(f"  [경고] 없음: {path}")
            continue
        s, l, nn, na, total = extract_onset_windows(path, label, onset, yolo)
        print(f"  {fname}: {total}f, onset {onset} → 정상 {nn} / "
              f"{LABEL_NAMES[label]} {na} windows")
        all_seqs.extend(s)
        all_labels.extend(l)

    if not all_seqs:
        print("데모 윈도우 없음.")
        return

    demo_arr = np.repeat(np.stack(all_seqs).astype(np.float32), args.repeat, axis=0)
    demo_lbl = np.repeat(np.array(all_labels, dtype=np.int64), args.repeat, axis=0)

    d = np.load(args.npz)
    arr = np.concatenate([d["sequences"], demo_arr], axis=0)
    lbl = np.concatenate([d["labels"], demo_lbl], axis=0)
    np.savez_compressed(args.npz, sequences=arr, labels=lbl)
    print(f"\n온셋 데이터 추가: {len(all_seqs)} windows × {args.repeat} "
          f"→ 총 train {len(arr)}")
    print("라벨 분포:", np.bincount(lbl, minlength=8).tolist())


if __name__ == "__main__":
    main()
