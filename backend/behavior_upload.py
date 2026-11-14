"""원본 GUI가 지원하는 키포인트 행동 가중치의 첨부 검증."""

import json
import sys
from zipfile import ZipFile

from backend.samples import ROOT

MODEL_DIR = ROOT / ".local-models"
MAX_MODEL_BYTES = 8 * 1024 * 1024
UPLOAD_PREFIX = "upload_"


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

    torch.set_num_threads(1)
    model = TrackBehaviorBuffer._build_and_load(path).eval()
    if not all(torch.isfinite(tensor).all().item() for tensor in model.state_dict().values()):
        raise ValueError("가중치에 유효하지 않은 값이 있습니다.")
    with torch.inference_mode():
        logits = model(torch.zeros(1, 30, 34))
    if logits.shape != (1, 8) or not torch.isfinite(logits).all():
        raise ValueError("8개 행동 클래스 출력이 올바르지 않습니다.")
    return "KeypointLSTM v2" if isinstance(model, KeypointLSTMv2) else "KeypointLSTM v1"


if __name__ == "__main__":
    print(inspect_checkpoint(sys.argv[1]))
