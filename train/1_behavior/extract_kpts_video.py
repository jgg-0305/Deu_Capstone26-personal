# -*- coding: utf-8 -*-
"""split/data/ 하위 클래스 폴더의 mp4 영상에서 키포인트를 추출해 kpt_*.npz 생성.

폴더 → 라벨 (kpt_common.FOLDER_LABEL_MAP):
    01.매장이동/04.구매/05.반품 → 0(정상)
    07.전도1 08.파손2 09.방화3 10.흡연4 11.유기5 12.절도6 13.폭행7
    14.교통약자 → 스킵(8클래스 모델 외)

영상 파일 단위로 train/val 분할(윈도우 단위 분할의 누수 제거, seed 고정).
※ 데모 영상의 온셋(시작 프레임) 시간 라벨은 demo_onset.py 가 별도로 추가한다.

사용법:
    python extract_kpts_video.py
    python extract_kpts_video.py --train-ratio 0.8 --model yolo11n-pose.pt
"""

import argparse
import os
import random
from collections import Counter

import numpy as np

from kpt_common import (
    SEQ_LEN, STRIDE, FOLDER_LABEL_MAP, LABEL_NAMES,
    video_to_vecs, make_windows, merge_npz, load_yolo,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_SEED = 42


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(_HERE, p)


def main():
    parser = argparse.ArgumentParser(description="클래스 폴더 mp4 → 키포인트 npz")
    parser.add_argument("--data-dir", default=os.path.join(_HERE, "split", "data"))
    parser.add_argument("--split-dir", default=os.path.join(_HERE, "split"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    yolo = load_yolo(args.model)
    data_dir, split_dir = _abs(args.data_dir), _abs(args.split_dir)

    train_seqs, train_labels = [], []
    val_seqs, val_labels = [], []

    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        prefix = folder.split(".")[0]
        if prefix not in FOLDER_LABEL_MAP:
            print(f"스킵: {folder}")
            continue
        label = FOLDER_LABEL_MAP[prefix]

        mp4s = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".mp4"))
        if not mp4s:
            print(f"스킵: {folder} (mp4 없음)")
            continue

        # 영상 파일 단위 train/val 분할 (재현성 위해 라벨별 seed 고정)
        rng = random.Random(SPLIT_SEED + label)
        rng.shuffle(mp4s)
        n_train = max(1, int(len(mp4s) * args.train_ratio))
        val_files = set(mp4s[n_train:])

        print(f"\n[{folder}] → label {label} ({LABEL_NAMES[label]}), "
              f"영상 {len(mp4s)}개 (train {n_train} / val {len(val_files)})")
        for name in mp4s:
            split = "val" if name in val_files else "train"
            vecs = video_to_vecs(os.path.join(folder_path, name), yolo, progress=True)
            seqs, labels = make_windows(vecs, label, args.seq_len, args.stride)
            print(f"  [{split}] {name}: {len(seqs)} windows")
            if split == "val":
                val_seqs.extend(seqs); val_labels.extend(labels)
            else:
                train_seqs.extend(seqs); train_labels.extend(labels)

    if not train_seqs:
        print("\n추출 시퀀스 없음. 데이터 디렉터리를 확인하세요.")
        return

    train_npz = os.path.join(split_dir, "kpt_train.npz")
    val_npz = os.path.join(split_dir, "kpt_val.npz")
    n_tr = merge_npz(train_npz, train_seqs, train_labels)
    n_va = merge_npz(val_npz, val_seqs, val_labels)

    print(f"\n저장 완료: train {n_tr} / val {n_va}")
    for split_name, npz_path in [("train", train_npz), ("val", val_npz)]:
        c = Counter(np.load(npz_path)["labels"].tolist())
        dist = "  ".join(f"{LABEL_NAMES[i]}:{c.get(i, 0)}"
                         for i in range(8) if c.get(i, 0))
        print(f"  {split_name}: {dist}")


if __name__ == "__main__":
    main()
