# 이전 웹 실행 명령과 호환되는 진입점이다. backend.api에서 정의한 같은 FastAPI 앱을 노출해 두 진입점의 동작이 분리되지 않게
# 한다.
"""기존 uvicorn core.api_server:app 명령과 호환되는 진입점."""

from backend.api import app as app
