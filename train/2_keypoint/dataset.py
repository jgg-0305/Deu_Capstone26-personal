"""UniPose 키포인트 데이터셋.

source/2. 키포인트 객체 인식/UniPose/lib/dataset/nia.py 를 독립·단순화.

JSON 주석 형식 (split/train.json, split/val.json):
[
  {
    "image": "relative/path/to/image.png",   // DATA_ROOT 기준 상대경로
    "center": [cx, cy],                       // 사람 중심 (픽셀)
    "scale": 1.0,                             // (bbox 폭 or 높이) / 200
    "joints": [[x0,y0], [x1,y1], ...],        // 17관절 원본 좌표
    "joints_vis": [1, 1, 0, ...]              // 1=가시, 0=불가시
  },
  ...
]
"""

import json
import os
import math
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import config as C


def _get_affine_transform(center, scale, rot, output_size, inv=False):
    """center + scale → affine transform 행렬."""
    scale_px = scale * 200.0
    src_w = scale_px
    dst_w, dst_h = output_size

    rot_rad = math.pi * rot / 180.0

    def rotate(vec, rad):
        s, c = math.sin(rad), math.cos(rad)
        return np.array([-vec[1] * s + vec[0] * c,
                          vec[1] * c + vec[0] * s])

    src_dir = rotate(np.array([0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0, dst_w * -0.5])

    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + src_dir
    dst[0] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1] = dst[0] + dst_dir

    # 3번째 점 (외적으로 결정)
    def _3rd(a, b):
        d = a - b
        return b + np.array([-d[1], d[0]])

    src[2] = _3rd(src[0], src[1])
    dst[2] = _3rd(dst[0], dst[1])

    if inv:
        mat = cv2.getAffineTransform(dst.astype(np.float32), src.astype(np.float32))
    else:
        mat = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
    return mat


def _affine_pt(pt, mat):
    v = np.array([pt[0], pt[1], 1.0])
    return mat @ v


def _make_heatmap(joints, joints_vis, hm_h, hm_w, sigma):
    """(17, hm_h, hm_w) 가우시안 히트맵 + (17,) target_weight."""
    num_joints = len(joints)
    hm = np.zeros((num_joints, hm_h, hm_w), dtype=np.float32)
    weight = np.ones((num_joints, 1), dtype=np.float32)

    size = 6 * sigma + 1
    x0 = y0 = size // 2
    g = np.exp(-((np.arange(size) - x0) ** 2) / (2 * sigma ** 2))
    gaussian = np.outer(g, g)

    for i, (jt, vis) in enumerate(zip(joints, joints_vis)):
        if vis == 0:
            weight[i] = 0
            continue
        px, py = int(jt[0] + 0.5), int(jt[1] + 0.5)
        ul = [px - x0, py - y0]
        br = [px + x0, py + y0]

        # 범위 클리핑
        img_x = max(0, ul[0]), min(hm_w, br[0] + 1)
        img_y = max(0, ul[1]), min(hm_h, br[1] + 1)
        g_x = max(0, -ul[0]), min(size, hm_w - ul[0])
        g_y = max(0, -ul[1]), min(size, hm_h - ul[1])

        if img_x[1] > img_x[0] and img_y[1] > img_y[0] and \
                g_x[1] > g_x[0] and g_y[1] > g_y[0]:
            hm[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                gaussian[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return hm, weight


class KeypointDataset(Dataset):
    """NIA 키포인트 데이터셋."""

    def __init__(self, root: str, json_path: str, is_train: bool = True,
                 transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        with open(json_path, encoding="utf-8") as f:
            self.db = json.load(f)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        ann = self.db[idx]
        img_path = ann["image"].replace("\\", os.sep)
        img_path = img_path if os.path.isabs(img_path) else \
            os.path.join(self.root, img_path)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((C.IMAGE_H, C.IMAGE_W, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # UniPose: RGB 입력

        center = np.array(ann["center"], dtype=np.float64)
        scale = float(ann["scale"])
        joints = np.array(ann["joints"], dtype=np.float64)    # (17, 2)
        joints_vis = np.array(ann["joints_vis"], dtype=int)    # (17,)

        # 중심/스케일 미세 조정 (source 와 동일)
        center[1] += 15 * scale
        scale *= 1.25

        rot = 0.0
        if self.is_train:
            # 스케일 지터
            scale *= np.clip(np.random.randn() * C.SCALE_FACTOR + 1,
                             1 - C.SCALE_FACTOR, 1 + C.SCALE_FACTOR)
            # 회전
            if random.random() < 0.6:
                rot = np.clip(np.random.randn() * C.ROT_FACTOR,
                              -C.ROT_FACTOR * 2, C.ROT_FACTOR * 2)
            # 수평 뒤집기
            if C.FLIP and random.random() > 0.5:
                img = img[:, ::-1, :].copy()
                w = img.shape[1]
                joints[:, 0] = w - 1 - joints[:, 0]
                for lp, rp in C.FLIP_PAIRS:
                    joints[[lp, rp]] = joints[[rp, lp]]
                    joints_vis[[lp, rp]] = joints_vis[[rp, lp]]
                center[0] = w - 1 - center[0]

        # Affine crop → (IMAGE_H, IMAGE_W)
        mat = _get_affine_transform(center, scale, rot,
                                    (C.IMAGE_W, C.IMAGE_H))
        img = cv2.warpAffine(img, mat, (C.IMAGE_W, C.IMAGE_H),
                             flags=cv2.INTER_LINEAR)

        # 관절 좌표를 히트맵 공간으로 변환
        mat_hm = _get_affine_transform(center, scale, rot,
                                       (C.HEATMAP_W, C.HEATMAP_H))
        joints_hm = np.zeros_like(joints)
        for i, (jt, vis) in enumerate(zip(joints, joints_vis)):
            if vis:
                joints_hm[i] = _affine_pt(jt, mat_hm)

        heatmap, target_weight = _make_heatmap(
            joints_hm, joints_vis, C.HEATMAP_H, C.HEATMAP_W, C.SIGMA
        )

        if self.transform:
            img = self.transform(img)

        return (img,
                torch.from_numpy(heatmap),
                torch.from_numpy(target_weight))


def build_transform(is_train=True):
    """UniPose 전처리: RGB → ToTensor → ImageNet 정규화."""
    ops = [T.ToTensor(),
           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    if is_train:
        # 가벼운 색상 지터
        ops.insert(0, T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
    return T.Compose(ops)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python dataset.py <DATA_ROOT> <json_path>")
        sys.exit(1)
    ds = KeypointDataset(sys.argv[1], sys.argv[2], transform=build_transform(True))
    print(f"샘플 수: {len(ds)}")
    img, hm, tw = ds[0]
    print(f"img {img.shape}  heatmap {hm.shape}  weight {tw.shape}")
