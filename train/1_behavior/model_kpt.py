# -*- coding: utf-8 -*-
"""키포인트 행동 분류 모델 정의 — 앱(core)과 단일 소스 공유.

모델 구조의 정본은 core/kpt_behavior_classifier.py 한 곳에만 둔다.
학습 스크립트는 이 모듈을 통해 동일 클래스를 import 한다(중복 정의 제거).

저장 포맷: plain state_dict() → torch.save(model.state_dict(), 'kpt_behavior.pth')
"""

import os
import sys

# core 패키지 import 를 위해 프로젝트 루트를 path 에 추가
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.kpt_behavior_classifier import KeypointLSTM, KeypointLSTMv2  # noqa: E402

__all__ = ["KeypointLSTM", "KeypointLSTMv2"]


if __name__ == "__main__":
    import torch
    for cls in (KeypointLSTM, KeypointLSTMv2):
        m = cls()
        x = torch.rand(4, 30, 34)
        n = sum(p.numel() for p in m.parameters())
        print(f"{cls.__name__}: {x.shape} → {m(x).shape}, params {n:,}")
