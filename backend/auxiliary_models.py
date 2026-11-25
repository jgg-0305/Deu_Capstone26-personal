"""Mac과 Windows의 분석 워커에서 공통으로 사용하는 상품·가방 가중치 경로."""

import os

from backend.behavior_upload import MODEL_DIR
from backend.samples import ROOT


BAG_MODEL_PATH = MODEL_DIR / "yolo11s-bag.pt"


def article_model_path():
    """명시한 경로를 우선하고, 기본 파일명과 이전에 설치한 파일명을 차례로 찾는다."""
    configured = os.getenv("CCTV_ARTICLE_MODEL", "").strip()
    candidates = (ROOT / configured,) if configured else (
        MODEL_DIR / "article.pt", MODEL_DIR / "article-item.pt",
    )
    return next((path for path in candidates if path.is_file()), None)
