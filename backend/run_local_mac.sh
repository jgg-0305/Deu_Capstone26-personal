#!/usr/bin/env bash
# macOS의 프로젝트 전용 PostgreSQL과 웹 관제를 시작한다. 기존 DB 데이터를 Unix 소켓으로 열고, 저장된 접속 설정으로
# 연결을 확인한 뒤 React를 빌드하여 통합 실행기에 넘긴다.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f .local-db/web.env || ! -f .local-db/data/PG_VERSION ]]; then
  echo "기존 Mac 로컬 DB 설정(.local-db/web.env 및 .local-db/data)을 찾을 수 없습니다." >&2
  exit 1
fi

pg_version=$(<.local-db/data/PG_VERSION)
pg_ctl="$(brew --prefix "postgresql@$pg_version")/bin/pg_ctl"
if [[ ! -x "$pg_ctl" ]]; then
  echo "Homebrew PostgreSQL $pg_version의 pg_ctl을 찾을 수 없습니다." >&2
  exit 1
fi

socket_dir=/private/tmp/deu-cctv-local-db
if ! "$pg_ctl" -D .local-db/data status >/dev/null; then
  (umask 077; mkdir -p "$socket_dir")
  "$pg_ctl" -D .local-db/data -l .local-db/postgres.log \
    -o "-c listen_addresses= -k $socket_dir -p 55432 -c unix_socket_permissions=0700" -w start
fi

set -a
source .local-db/web.env
set +a
.venv-web/bin/python -c 'from backend.db import connect; connect().close()'
npm --prefix frontend run build
exec .venv-web/bin/python -m backend.run_local "$@"
