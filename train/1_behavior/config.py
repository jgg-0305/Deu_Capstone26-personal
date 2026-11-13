"""이상행동 분류 학습 설정.

경로만 수정하면 바로 학습을 시작할 수 있다.
"""

import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))

# ─── 데이터 경로 ────────────────────────────────────────────────────────────────
# 이미지 루트 디렉터리 (CSV 경로들의 기준점) — config.py 위치 기준 절대경로
DATA_ROOT = _os.path.join(_HERE, "split", "data")

# CSV 파일 위치 — split/ 폴더에 train.csv, val.csv 를 준비해야 한다
# 형식: frame1_path,frame2_path,frame3_path,label  (헤더 없음)
# 경로는 DATA_ROOT 상대경로 또는 절대경로 둘 다 허용
SPLIT_DIR = "split"  # 이 config.py 기준 상대 경로

# ─── 모델/출력 ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = _os.path.join(_HERE, "output")  # 체크포인트 저장 위치 (자동 생성)
NUM_CLASSES = 8             # 0 정상 / 1 전도 / 2 파손 / 3 방화 / 4 흡연 / 5 유기 / 6 절도 / 7 폭행
SEQ_LEN = 3                 # 시퀀스 길이 (학습 데이터 FPS 와 맞춤)
HIDDEN_SIZE = 256
NUM_LAYERS = 1
BACKBONE = "resnet50"       # torchvision ResNet 종류

# ─── 학습 하이퍼파라미터 ─────────────────────────────────────────────────────────
BEGIN_EPOCH = 0
END_EPOCH = 50
BATCH_SIZE = 4
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
LR_MIN_RATIO = 0.1          # CosineAnnealingLR eta_min = LR * LR_MIN_RATIO
USE_CLASS_WEIGHT = True     # 클래스 불균형 보정용 가중 CrossEntropyLoss

# ─── GPU ─────────────────────────────────────────────────────────────────────────
# DataParallel 사용할 GPU 인덱스 목록. 단일 GPU면 [0]
GPUS = [0]

# ─── 저장 포맷 (앱과 호환) ────────────────────────────────────────────────────────
# 앱의 BehaviorClassifier._load_weights() 가 'state_dict' 키 + 'module.' 접두사를 처리함.
# DataParallel 사용 시 model.state_dict() 키에 'module.' 이 붙으며, 앱이 이를 자동 제거한다.
SAVE_INTERVAL = 5           # N 에포크마다 체크포인트 저장
