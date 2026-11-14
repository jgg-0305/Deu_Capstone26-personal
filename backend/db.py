"""웹 API와 관리 명령에서 사용하는 PostgreSQL 연결."""

import os

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row


class DatabaseConfigurationError(ValueError):
    """연결 문자열이 없거나 예시 값인 경우를 나타낸다."""


def connect():
    dsn = os.environ.get("CCTV_DATABASE_URL", "").strip()
    if not dsn:
        raise DatabaseConfigurationError("CCTV_DATABASE_URL에 실제 PostgreSQL 연결 정보를 설정하세요.")
    try:
        options = conninfo_to_dict(dsn)
    except psycopg.ProgrammingError:
        raise DatabaseConfigurationError("CCTV_DATABASE_URL의 연결 문자열 형식을 확인하세요.") from None
    if options.get("user") == "USER" or options.get("password") == "PASSWORD":
        raise DatabaseConfigurationError("CCTV_DATABASE_URL의 USER와 PASSWORD를 실제 PostgreSQL 계정 값으로 바꾸세요.")
    return psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5)
