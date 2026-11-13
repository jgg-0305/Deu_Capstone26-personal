"""UniPose 키포인트 학습 설정."""

import os

# ─── 경로 ─────────────────────────────────────────────────────────────────────────
DATA_ROOT = r"C:\data\NIA"          # 이미지 루트
SPLIT_DIR = "split"                 # split/train.json, split/val.json 위치
OUTPUT_DIR = "output"               # 체크포인트 저장

# ResNet 사전학습 가중치 — model/2. 키포인트 객체 인식/UniPose/weight/ 에 이미 있음
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_HERE, "..", "..")
BACKBONE_PRETRAINED = os.path.normpath(
    os.path.join(_PROJECT_ROOT, "model", "2. 키포인트 객체 인식",
                 "UniPose", "weight", "resnet50-19c8e357.pth")
)

# ─── 모델 ─────────────────────────────────────────────────────────────────────────
BACKBONE = "resnet50"
OUTPUT_STRIDE = 8
NUM_JOINTS = 17

# ─── 입력 크기 ────────────────────────────────────────────────────────────────────
IMAGE_W = 192          # crop 후 입력 너비
IMAGE_H = 256          # crop 후 입력 높이
HEATMAP_W = IMAGE_W // OUTPUT_STRIDE   # 24
HEATMAP_H = IMAGE_H // OUTPUT_STRIDE   # 32
SIGMA = 2              # Gaussian 히트맵 표준편차

# ─── 학습 ─────────────────────────────────────────────────────────────────────────
BEGIN_EPOCH = 0
END_EPOCH = 15
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
LR_END = 1e-5
LR_STEP = [7, 10, 13]   # MultiStepLR

# 데이터 증강
FLIP = True
ROT_FACTOR = 45
SCALE_FACTOR = 0.35

GPUS = [0]
SAVE_INTERVAL = 5

# ─── 관절 정의 (NIA 17관절 — read_data_from_nia.py 순서) ──────────────────────────
# idx:  0         1          2          3        4          5
JOINT_NAMES = [
    "Right foot", "Right knee", "Right hip", "Left hip", "Left knee", "Left foot",
    "Pelvis", "Spine naval", "Spine chest", "Neck base", "Center head",
    "Right hand", "Right elbow", "Right shoulder",
    "Left shoulder", "Left elbow", "Left hand",
]

# 좌우 대칭 쌍 (flip augmentation)
FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [11, 16], [12, 15], [13, 14]]
