# 웹캠 입력과 분석 미리보기의 최신 프레임을 교환한다. 임시 파일에 기록한 뒤 목적지로 교체해 다른 프로세스가 불완전한 프레임을 읽지 않게
# 한다. 스냅샷은 현재 분석 실행의 결과인지 확인해서 반환한다.
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
            # Windows에서는 이전 스냅샷이 열려 있으면 파일 교체가 잠시 거부될 수 있다.
            # 같은 워커 안에서 읽는 경우도 있어 교체를 재시도한다.
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
