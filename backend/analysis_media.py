"""웹캠 입력과 분석 미리보기의 최신 프레임을 로컬에서 교환한다."""

import json
import os
from tempfile import NamedTemporaryFile

from backend.samples import UPLOAD_DIR

FRAME_LIMIT = 256 * 1024


def frame_path(camera_id, kind):
    return UPLOAD_DIR / f"{camera_id}.{kind}"


def replace_frame(camera_id, kind, data):
    UPLOAD_DIR.mkdir(mode=0o700, exist_ok=True)
    path = frame_path(camera_id, kind)
    with NamedTemporaryFile(dir=UPLOAD_DIR, suffix=".part", delete=False) as output:
        temp = output.name
        try:
            output.write(data)
            output.close()
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def read_snapshot(camera_id, run_id):
    try:
        snapshot = json.loads(frame_path(camera_id, "analysis.json").read_bytes())
    except (FileNotFoundError, ValueError):
        return None
    return snapshot if snapshot["run_id"] == str(run_id) else None
