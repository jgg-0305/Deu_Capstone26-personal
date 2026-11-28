"""로컬 관제 서버와 워커를 함께 실행한다. 웹 빌드와 DB 준비를 확인하고 API 준비 응답을 받은 뒤 분석 워커와 선택적 카카오 발송 워커를
시작한다. 사진용 ngrok 자동 실행을 켜면 서명된 사건 사진만 제공하는 게이트웨이를 함께 운영한다.
"""

import argparse
import os
import secrets
import shutil
import subprocess
import sys
from threading import Thread
import time
from urllib.error import URLError
from urllib.request import urlopen

from backend.manage import migrate, register_models
from backend.kakao_image_gateway import photo_gateway
from backend.kakao_images import settings as image_settings
from backend.kakao_notify import delivery_link
from backend.samples import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not (ROOT / "frontend/dist/index.html").is_file():
        parser.error("Build the frontend with npm --prefix frontend run build.")
    migrate()
    register_models()
    env = os.environ.copy()
    env.setdefault("CCTV_WORKER_TOKEN", secrets.token_urlsafe(32))
    if len(env["CCTV_WORKER_TOKEN"]) < 32:
        parser.error("CCTV_WORKER_TOKEN must contain at least 32 characters.")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    address = f"http://127.0.0.1:{args.port}"
    photo_config = image_settings()
    auto_ngrok = env.get("CCTV_KAKAO_NGROK") == "1"
    ngrok = shutil.which("ngrok") if auto_ngrok else None
    if auto_ngrok and (not ngrok or photo_config is None):
        print("사진 게이트웨이 설정을 확인하세요. ngrok 실행 파일, HTTPS 주소와 서명 키가 필요합니다.", flush=True)
        ngrok = None

    def start(*command):
        return subprocess.Popen([sys.executable, "-m", *command], cwd=ROOT, env=env)

    api = start("uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", str(args.port), "--no-access-log")
    worker = None
    notifier = None
    notifier_retry = 0.0
    gateway = None
    tunnel = None
    tunnel_retry = 0.0
    kakao_link = delivery_link()
    if env.get("CCTV_KAKAO_LINK_URL") and not kakao_link:
        print("카카오톡 발송 주소가 올바르지 않아 알림 워커를 시작하지 않습니다.", flush=True)
    try:
        for _ in range(50):
            if api.poll() is not None:
                raise RuntimeError("Web API exited before startup.")
            try:
                with urlopen(address + "/health/ready", timeout=1):
                    break
            except (URLError, TimeoutError):
                time.sleep(.2)
        else:
            raise RuntimeError("Web API did not become ready.")
        if ngrok:
            # 운영체제가 할당한 전용 포트의 게이트웨이만 공개 대상으로 사용한다.
            gateway = photo_gateway(0, args.port)
            Thread(target=gateway.serve_forever, daemon=True).start()
            print(f"카카오톡 사진 게이트웨이: 127.0.0.1:{gateway.server_port}", flush=True)
            ngrok_env = {key: value for key, value in env.items() if not key.startswith("CCTV_")}
        worker_retry = 0.0
        while api.poll() is None:
            now = time.monotonic()
            if tunnel is not None and tunnel.poll() is not None:
                print(f"사진용 ngrok 종료({tunnel.returncode}). 10초 뒤 다시 시작합니다.", flush=True)
                tunnel, tunnel_retry = None, now + 10
            if gateway and tunnel is None and now >= tunnel_retry:
                target = f"http://127.0.0.1:{gateway.server_port}"
                try:
                    tunnel = subprocess.Popen([ngrok, "http", target, "--url", photo_config[0],
                                               "--inspect=false", "--log", "stdout", "--log-level", "warn"],
                                              cwd=ROOT, env=ngrok_env)
                    print("카카오톡 사진용 ngrok을 시작했습니다.", flush=True)
                except OSError:
                    print("사진용 ngrok을 시작하지 못했습니다. 10초 뒤 다시 시도합니다.", flush=True)
                    tunnel_retry = now + 10
            if worker is not None and worker.poll() is not None:
                print(f"분석 워커 종료({worker.returncode}). 2초 뒤 다시 시작합니다.", flush=True)
                worker, worker_retry = None, now + 2
            if worker is None and now >= worker_retry:
                worker = start("backend.worker", "--serve", "--api-url", address)
            if notifier is not None and notifier.poll() is not None:
                print(f"카카오톡 워커 종료({notifier.returncode}). 10초 뒤 다시 시작합니다.", flush=True)
                notifier, notifier_retry = None, now + 10
            if kakao_link and notifier is None and now >= notifier_retry:
                notifier = start("backend.kakao_notify", "serve")
            time.sleep(.5)
        raise SystemExit(api.returncode)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (notifier, worker, tunnel, api):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if gateway:
            gateway.shutdown()
            gateway.server_close()


if __name__ == "__main__":
    main()
