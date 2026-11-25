"""재학습 단계에서 공유할 데이터와 출력 경로를 결정한다. 실행 도구가 넘긴 환경값을 우선하며 필요한 항목에는 프로젝트 기본 경로를 사용한다.
"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('BEHAVIOR_OUTPUT', ROOT/'train/behavior_repair/output')).resolve()
DATA=Path(os.environ.get('BEHAVIOR_DATA_ROOT', ROOT/'data/aihub')).resolve()
PREPARED=Path(os.environ.get('BEHAVIOR_PREPARED', ROOT/'train/behavior_repair/prepared')).resolve()
PREVIOUS=Path(os.environ.get('BEHAVIOR_PREVIOUS', ROOT/'train/1_behavior/kpt_behavior_local.pth')).resolve()
LEGACY_V1=Path(os.environ['BEHAVIOR_V1']).resolve() if os.environ.get('BEHAVIOR_V1') else None
LEGACY_V2=Path(os.environ.get('BEHAVIOR_V2', ROOT/'train/1_behavior/output/kpt_behavior.pth')).resolve()
POSE_N=Path(os.environ.get('BEHAVIOR_POSE_N', ROOT/'yolo11n-pose.pt')).resolve()
POSE_S=Path(os.environ.get('BEHAVIOR_POSE_S', ROOT/'yolo11s-pose.pt')).resolve()
