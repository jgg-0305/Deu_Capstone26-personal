"""Start the web API and the analysis worker together on the local PC."""

import argparse
import os
import secrets
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from backend.manage import migrate, register_models
from backend.samples import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not (ROOT / "frontend/dist/index.html").is_file():
        parser.error("Build the frontend with npm --prefix frontend run build.")
    migrate()
    register_models(allow_retrained_behavior=True)
    env = os.environ.copy()
    env.setdefault("CCTV_WORKER_TOKEN", secrets.token_urlsafe(32))
    if len(env["CCTV_WORKER_TOKEN"]) < 32:
        parser.error("CCTV_WORKER_TOKEN must contain at least 32 characters.")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    address = f"http://127.0.0.1:{args.port}"

    def start(*command):
        return subprocess.Popen([sys.executable, "-m", *command], cwd=ROOT, env=env)

    api = start("uvicorn", "backend.api:app", "--host", "127.0.0.1", "--port", str(args.port), "--no-access-log")
    worker = None
    notifier = None
    notifier_retry = 0.0
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
        worker_retry = 0.0
        while api.poll() is None:
            now = time.monotonic()
            if worker is not None and worker.poll() is not None:
                print(f"분석 워커 종료({worker.returncode}). 2초 뒤 다시 시작합니다.", flush=True)
                worker, worker_retry = None, now + 2
            if worker is None and now >= worker_retry:
                worker = start("backend.worker", "--serve", "--api-url", address)
            if notifier is not None and notifier.poll() is not None:
                print(f"카카오톡 워커 종료({notifier.returncode}). 10초 뒤 다시 시작합니다.", flush=True)
                notifier, notifier_retry = None, now + 10
            if env.get("CCTV_KAKAO_LINK_URL") and notifier is None and now >= notifier_retry:
                notifier = start("backend.kakao_notify", "serve")
            time.sleep(.5)
        raise SystemExit(api.returncode)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (notifier, worker, api):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
