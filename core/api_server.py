"""기존 uvicorn core.api_server:app 명령과 호환되는 진입점."""

from backend.api import app as app
