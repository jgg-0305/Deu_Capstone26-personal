"""이상행동 분류 데이터셋.

source/1. 이상행동 분류/ConvLSTM/lib/dataset/NIADataset.py 를 정리·독립화.

CSV 형식 (헤더 없음):
    frame1_rel_path,frame2_rel_path,frame3_rel_path,label

    - 경로는 DATA_ROOT 기준 상대경로
    - label: 0=정상, 1=전도, 2=파손, 3=방화, 4=흡연, 5=유기, 6=절도, 7=폭행

전처리 주의:
    학습 원본 코드가 cv2.imread(BGR) 를 RGB 변환 없이 ToTensor+Normalize 했으므로,
    추론 측(core/behavior_classifier.py)과 동일하게 BGR 채널 순서를 유지한다.
"""

import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

LABEL_NAMES = ["정상", "전도", "파손", "방화", "흡연", "유기", "절도", "폭행"]


class BehaviorDataset(Dataset):
    """3-프레임 시퀀스 이미지 → 이상행동 레이블."""

    def __init__(self, root: str, csv_path: str, seq_len: int = 3, transform=None):
        self.root = root
        self.seq_len = seq_len
        self.transform = transform
        self.samples = self._load_csv(csv_path)

    def _load_csv(self, csv_path: str):
        samples = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                row = [r.strip() for r in row]
                if not row or len(row) < self.seq_len + 1:
                    continue
                paths = [r.replace("\\", os.sep) for r in row[:self.seq_len]]
                label = int(row[self.seq_len])
                # 존재 확인: 루트 결합 또는 절대경로 그대로
                full_paths = []
                for p in paths:
                    fp = p if os.path.isabs(p) else os.path.join(self.root, p)
                    full_paths.append(fp)
                # 첫 프레임만 확인 (나머지는 데이터 로더 단계에서 처리)
                if os.path.exists(full_paths[0]):
                    samples.append((full_paths, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        imgs = torch.zeros(self.seq_len, 3, 224, 224, dtype=torch.float32)
        for i, fp in enumerate(paths):
            img = cv2.imread(fp, cv2.IMREAD_COLOR)  # BGR 유지 (학습과 동일)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            if self.transform:
                img = self.transform(img)
            imgs[i] = img
        return imgs, label

    def class_weights(self) -> torch.Tensor:
        """불균형 클래스 보정 가중치 (CrossEntropyLoss weight 파라미터용)."""
        counts = [0] * 8
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts)
        weights = [1.0 - (c / total) if total > 0 else 1.0 for c in counts]
        return torch.FloatTensor(weights)


def build_transform():
    """학습 원본과 동일한 전처리 (BGR → ToTensor → ImageNet 정규화)."""
    return T.Compose([
        T.ToTensor(),
        # ImageNet 정규화 (mean/std 는 RGB 기준이지만 학습 원본도 BGR로 그대로 적용함)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python dataset.py <DATA_ROOT> <csv_path>")
        sys.exit(1)
    root, csv_path = sys.argv[1], sys.argv[2]
    ds = BehaviorDataset(root, csv_path, transform=build_transform())
    print(f"샘플 수: {len(ds)}")
    imgs, label = ds[0]
    print(f"imgs shape: {imgs.shape}, label: {label} ({LABEL_NAMES[label]})")
    print(f"클래스 가중치: {ds.class_weights().tolist()}")
