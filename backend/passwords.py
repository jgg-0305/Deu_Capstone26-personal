# 관제 로그인에 사용하는 Argon2 비밀번호 해시와 검증을 제공한다. 로컬 테스트 관리자 허용 여부는 환경 설정으로 판단한다.
"""사용자 비밀번호 해시와 로컬 테스트 계정 설정."""

import os

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError

hasher = PasswordHasher()
DUMMY_HASH = hasher.hash(os.urandom(32))


def verify_password(password, encoded):
    try:
        return hasher.verify(encoded or DUMMY_HASH, password) and encoded is not None
    except (VerificationError, InvalidHashError):
        return False


def test_admin_enabled():
    return os.environ.get("CCTV_ENABLE_TEST_ADMIN") == "1"
