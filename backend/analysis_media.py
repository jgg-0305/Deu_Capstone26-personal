"""웹캠 입력과 분석 미리보기의 최신 프레임을 로컬에서 교환한다."""

import json
import os
from tempfile import NamedTemporaryFile
from time import sleep

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
            # Windows can deny replacement while a reader briefly has the
            # previous snapshot open, including a reader in the worker process.
            for delay in (.01, .02, .04, .08, .16):
                try:
                    os.replace(temp, path)
                    break
                except PermissionError:
                    if os.name != "nt":
                        raise
                    sleep(delay)
            else:
                os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def read_snapshot(camera_id, run_id):
    path = frame_path(camera_id, "analysis.json")
    for delay in (.01, .02, .04):
        try:
            snapshot = json.loads(path.read_bytes())
            return snapshot if snapshot["run_id"] == str(run_id) else None
        except (FileNotFoundError, ValueError):
            return None
        except PermissionError:
            if os.name != "nt":
                raise
            sleep(delay)
    return None
