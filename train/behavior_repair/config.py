"""Paths supplied by run.py, with project-local defaults."""
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
