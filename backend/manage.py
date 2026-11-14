"""데이터베이스 이전과 사용자 계정 관리를 위한 명령줄 도구."""

import argparse
import hashlib
import json
from pathlib import Path
import getpass
import re
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from backend.db import DatabaseConfigurationError, connect
from backend.passwords import hasher, test_admin_enabled

ROOT = Path(__file__).resolve().parents[1]
# SQL 문장은 그대로이며 주석만 한국어로 바꾼 최초 마이그레이션의 전후 해시다.
INITIAL_COMMENT_TRANSLATION = (
    "3280b124e6e5411bc876315c8ff344d5d2f382d55880d545d7f2d2b71b2039cf",
    "dbca08f1244a836c43f94f735bc0f96ab6b90c9e7e9dd84a4b3315873f43b1ae",
)


def migrate():
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(917201)")
        db.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        applied = {
            row["name"]: row["sha256"]
            for row in db.execute("SELECT name, sha256 FROM schema_migrations")
        }
        for path in sorted((Path(__file__).parent / "migrations").glob("*.sql")):
            script = path.read_bytes()
            digest = hashlib.sha256(script).hexdigest()
            if path.name in applied:
                if path.name == "001_initial.sql" and (applied[path.name], digest) == INITIAL_COMMENT_TRANSLATION:
                    db.execute("UPDATE schema_migrations SET sha256 = %s WHERE name = %s", (digest, path.name))
                    continue
                if applied[path.name] != digest:
                    raise RuntimeError(f"Applied migration was changed: {path.name}")
                continue
            db.execute(script.decode())
            db.execute("INSERT INTO schema_migrations VALUES (%s, %s)", (path.name, digest))


def register_models():
    manifest = json.loads((ROOT / "docs/design/model-manifest.json").read_text())
    roles = {"human_pose": "pose", "human_pose_s": "pose", "store_objects": "object", "behavior": "behavior",
             "behavior_kpt_v1": "behavior", "behavior_rgb": "behavior"}
    with connect() as db:
        for artifact in manifest["models"]:
            path = ROOT / artifact["path"]
            if artifact.get("optional") and not path.is_file():
                print(f"로컬 체크포인트 없음: {artifact['path']}")
                continue
            with path.open("rb") as model_file:
                digest = hashlib.file_digest(model_file, "sha256").hexdigest()
            if digest != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
                raise RuntimeError(f"Model differs from manifest: {artifact['path']}")
            row = db.execute("""
                INSERT INTO model_artifacts
                    (role, name, version, sha256, storage_key, size_bytes, labels, preprocessing)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (sha256) DO UPDATE SET sha256 = EXCLUDED.sha256
                RETURNING id
            """, (roles[artifact["id"]], artifact["id"], artifact.get("version", manifest["upstream"]["commit"]),
                  digest, artifact["path"], artifact["bytes"], Jsonb(artifact["classes"]),
                  Jsonb({"architecture": artifact["architecture"],
                         "input_shape": artifact.get("input_shape"),
                         "keypoint_shape": artifact.get("keypoint_shape"),
                         "status": "baseline_under_review"}))).fetchone()
            print(roles[artifact["id"]], row["id"])


def seed_test_admin():
    if not test_admin_enabled():
        raise SystemExit("CCTV_ENABLE_TEST_ADMIN=1을 설정한 로컬 환경에서 실행하세요.")
    with connect() as db:
        row = db.execute("""
            INSERT INTO users (username, display_name, password_hash, role, is_test_admin)
            VALUES ('admin', '테스트 관리자', %s, 'owner', true)
            ON CONFLICT (username) DO NOTHING RETURNING id
        """, (hasher.hash("admin"),)).fetchone()
    print("테스트 관리자 계정을 만들었습니다." if row else "admin 계정이 이미 있습니다.")


def set_login(user_id, username):
    username = username.strip().lower()
    if username == "admin" or not re.fullmatch(r"[a-z0-9_]{3,32}", username):
        raise SystemExit("예약되지 않은 영문·숫자·밑줄 아이디를 3~32자로 입력하세요.")
    password = getpass.getpass("새 비밀번호: ")
    if not 8 <= len(password) <= 128 or password != getpass.getpass("새 비밀번호 확인: "):
        raise SystemExit("비밀번호 길이 또는 확인 값이 올바르지 않습니다.")
    with connect() as db:
        user = db.execute("""
            UPDATE users SET username = %s, password_hash = %s
            WHERE id = %s AND NOT is_test_admin RETURNING id
        """, (username, hasher.hash(password), user_id)).fetchone()
        if not user:
            raise SystemExit("변경 가능한 사용자를 찾을 수 없습니다.")
        db.execute("UPDATE auth_sessions SET revoked_at = now() WHERE user_id = %s AND revoked_at IS NULL",
                   (user_id,))
    print("기존 사용자에 로그인 정보를 저장했습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    commands.add_parser("register-models")
    commands.add_parser("seed-test-admin")
    commands.add_parser("cleanup-storage")
    assign = commands.add_parser("set-login")
    assign.add_argument("--user-id", required=True, type=UUID)
    assign.add_argument("--username", required=True)
    args = parser.parse_args()
    try:
        if args.command == "migrate":
            migrate()
        elif args.command == "register-models":
            register_models()
        elif args.command == "seed-test-admin":
            seed_test_admin()
        elif args.command == "cleanup-storage":
            from backend.storage import cleanup_pending
            cleanup_pending()
            print("보류 중인 영상 파일 정리를 재시도했습니다.")
        else:
            set_login(args.user_id, args.username)
    except DatabaseConfigurationError as error:
        parser.exit(1, f"{error}\n")
    except psycopg.OperationalError:
        parser.exit(1, "PostgreSQL 서버 실행 상태와 CCTV_DATABASE_URL을 확인하세요.\n")
    except psycopg.Error:
        parser.exit(1, "PostgreSQL 작업을 처리하지 못했습니다. 마이그레이션 상태와 권한을 확인하세요.\n")


if __name__ == "__main__":
    main()
