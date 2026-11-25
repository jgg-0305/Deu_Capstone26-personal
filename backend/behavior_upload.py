# 첨부한 관절 행동 모델을 등록하기 전에 검사한다. 압축 내부 크기를 제한하고 텐서 전용 로더로 가중치를 연 뒤, 예제 입력이 유한한 8개
# 클래스 점수를 만드는지 확인한다. 검사는 API에서 별도 프로세스로 실행한다.
"""원본 GUI가 지원하는 키포인트 행동 가중치의 첨부 검증."""

import json
import sys
from functools import lru_cache
from zipfile import ZipFile

from backend.samples import ROOT

MODEL_DIR = ROOT / ".local-models"
MAX_MODEL_BYTES = 8 * 1024 * 1024
UPLOAD_PREFIX = "upload_"
HAND_FUSION_NAME = "KeypointLSTM v2 + hand RGB and motion"
HAND_FUSION_FEATURES = 34 + 16 + 2 * 3 * 16 * 16


@lru_cache(maxsize=1)
def bundled_behavior_artifact():
    manifest = json.loads((ROOT / "docs/design/model-manifest.json").read_text(encoding="utf-8"))
    return next(model for model in manifest["models"] if model["id"] == "behavior")


def normalize_behavior_model(name):
    # 이미 첨부했던 기본 가중치의 선택값도 같은 파일을 가리킨다.
    if name and name.startswith(UPLOAD_PREFIX) and name == UPLOAD_PREFIX + bundled_behavior_artifact()["sha256"]:
        return "behavior"
    return name


def bundled_rgb_artifact():
    manifest = json.loads((ROOT / "docs/design/model-manifest.json").read_text(encoding="utf-8"))
    return next(model for model in manifest["models"] if model["id"] == "behavior_rgb")


def inspect_checkpoint(path):
    # 모델 객체 대신 제한된 로더로 읽을 수 있는 텐서 가중치만 등록한다.
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > 128 or sum(info.file_size for info in entries) > MAX_MODEL_BYTES:
            raise ValueError("가중치 내부 데이터가 너무 큽니다.")

    import torch

    from core.kpt_behavior_classifier import KeypointLSTMv2, TrackBehaviorBuffer
    from core.hand_behavior import HandFusion, INPUT_SIZE

    torch.set_num_threads(1)
    model = TrackBehaviorBuffer._build_and_load(path).eval()
    if not all(torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()):
        raise ValueError("가중치에 유효하지 않은 값이 있습니다.")
    with torch.inference_mode():
        logits = model(torch.zeros(1, 30, INPUT_SIZE if isinstance(model, HandFusion) else 34))
    if logits.shape != (1, 8) or not torch.isfinite(logits).all():
        raise ValueError("8개 행동 클래스 출력이 올바르지 않습니다.")
    if isinstance(model, HandFusion):
        if INPUT_SIZE != HAND_FUSION_FEATURES:
            raise ValueError("행동 모델 입력 규격이 등록 정보와 다릅니다.")
        return HAND_FUSION_NAME
    return "KeypointLSTM v2" if isinstance(model, KeypointLSTMv2) else "KeypointLSTM v1"


if __name__ == "__main__":
    print(inspect_checkpoint(sys.argv[1]))
