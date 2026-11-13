# -*- coding: utf-8 -*-
"""키포인트 학습 데이터셋(npz) 정제 — 정확도 향상용.

약지도(영상 전체 단일 라벨) + 중복 윈도우(stride 겹침)로 인한 과적합·라벨노이즈를
완화하기 위해 다음을 적용한다:

  1) 품질 필터   — 미검출 좌표 비율(zero-fraction)이 높은 윈도우 제거
  2) 중복 제거   — 같은 라벨에서 직전 보존 윈도우와 거의 동일한(겹친) 윈도우 제거
  3) 정적 노이즈 — 움직임이 본질인 클래스(전도·폭행)에서 모션 하위 p% 윈도우 제거
                   (해당 구간은 사건이 일어나지 않는 '정상' 구간일 가능성이 큼)

train 만 정제한다(val 은 동일 벤치마크 비교를 위해 원본 유지).

사용법:
    python refine_dataset.py
    python refine_dataset.py --max-zero 0.5 --dedup-pct 25 --motion-drop-pct 20
출력:
    split/kpt_train_refined.npz
"""

import argparse
import os
import numpy as np

from kpt_common import LABEL_NAMES as LBL

_HERE = os.path.dirname(os.path.abspath(__file__))
MOTION_CLASSES = {1, 7}   # 전도, 폭행 — 움직임이 본질


def motion_of(seq):
    """(N,T,34) → 윈도우별 평균 프레임간 변위(유효 관절만)."""
    nz = (seq != 0).astype(np.float32)
    vel = (seq[:, 1:] - seq[:, :-1]) * nz[:, 1:] * nz[:, :-1]
    denom = nz[:, 1:].reshape(len(seq), -1).sum(1) + 1e-6
    return np.sqrt((vel ** 2).reshape(len(seq), -1).sum(1) / denom)


def zero_fraction(seq):
    nz = (seq != 0).astype(np.float32)
    return 1.0 - nz.reshape(len(seq), -1).mean(1)


def refine(seq, lab, max_zero, dedup_pct, motion_drop_pct):
    keep = np.ones(len(seq), dtype=bool)

    # 1) 품질 필터
    zf = zero_fraction(seq)
    keep &= (zf <= max_zero)

    # 2) 중복 제거 — 클래스별로 직전 보존 윈도우와 L2 거리가 작은 윈도우 제거
    #    (윈도우는 영상 순서대로 쌓여 있어 겹친 윈도우가 인접함)
    flat = seq.reshape(len(seq), -1)
    for c in range(8):
        idx = np.where((lab == c) & keep)[0]
        if len(idx) < 3:
            continue
        # 인접 L2 거리 분포로 임계값 결정
        dists = np.linalg.norm(flat[idx[1:]] - flat[idx[:-1]], axis=1)
        tau = np.percentile(dists, dedup_pct)
        prev = flat[idx[0]]
        for j in idx[1:]:
            if np.linalg.norm(flat[j] - prev) < tau:
                keep[j] = False
            else:
                prev = flat[j]

    # 3) 정적 노이즈 제거 — 모션 클래스에서 모션 하위 p% 제거
    if motion_drop_pct > 0:
        mot = motion_of(seq)
        for c in MOTION_CLASSES:
            idx = np.where((lab == c) & keep)[0]
            if len(idx) < 10:
                continue
            thr = np.percentile(mot[idx], motion_drop_pct)
            drop = idx[mot[idx] < thr]
            keep[drop] = False

    return keep


def main():
    ap = argparse.ArgumentParser(description="키포인트 데이터셋 정제")
    ap.add_argument("--in-npz", default=os.path.join(_HERE, "split", "kpt_train.npz"))
    ap.add_argument("--out-npz", default=os.path.join(_HERE, "split", "kpt_train_refined.npz"))
    ap.add_argument("--max-zero", type=float, default=0.5)
    ap.add_argument("--dedup-pct", type=float, default=25.0)
    ap.add_argument("--motion-drop-pct", type=float, default=20.0)
    args = ap.parse_args()

    d = np.load(args.in_npz)
    seq, lab = d["sequences"].astype(np.float32), d["labels"].astype(np.int64)
    print(f"입력: {len(seq)} windows")

    keep = refine(seq, lab, args.max_zero, args.dedup_pct, args.motion_drop_pct)
    seq2, lab2 = seq[keep], lab[keep]

    print(f"보존: {len(seq2)} / 제거: {len(seq)-len(seq2)}")
    print(f"{'클래스':<6}{'before':>8}{'after':>8}")
    for c in range(8):
        print(f"{LBL[c]:<6}{int((lab==c).sum()):>8}{int((lab2==c).sum()):>8}")

    np.savez_compressed(args.out_npz, sequences=seq2, labels=lab2)
    print(f"\n저장: {args.out_npz}")


if __name__ == "__main__":
    main()
