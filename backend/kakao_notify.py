"""사건 발송 대기열을 처리하는 카카오톡 나에게 보내기 워커."""

import argparse
from datetime import timedelta, timezone
import getpass
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from urllib.parse import urlsplit

import requests

from backend.db import connect
from backend.kakao_images import signed_url
from backend.samples import ROOT

SECRET_DIR = ROOT / ".local-kakao"
WORKER_LOCK = 832741
REF = "local-kakao:"
LABELS = {"fall": "넘어짐", "vandalism": "파손", "arson": "방화", "smoking": "흡연",
          "abandon": "물건 유기", "theft_suspected": "도난 의심", "assault": "폭행"}
MEMO = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
TOKEN = "https://kauth.kakao.com/oauth/token"
TOKEN_INFO = "https://kapi.kakao.com/v1/user/access_token_info"
KST = timezone(timedelta(hours=9))


class KakaoError(Exception):
    def __init__(self, code, *, retry=False):
        super().__init__(code)
        self.retry = retry


def token_path(user_id):
    if SECRET_DIR.is_symlink():
        raise KakaoError("unsafe_secret_directory")
    path = SECRET_DIR / f"{user_id}.json"
    if path.is_symlink():
        raise KakaoError("unsafe_secret_file")
    return path


def store_tokens(user_id, tokens):
    path = token_path(user_id)
    SECRET_DIR.mkdir(mode=0o700, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=SECRET_DIR, delete=False) as fp:
        temp = Path(fp.name)
        try:
            json.dump(tokens, fp)
            fp.flush()
            os.fsync(fp.fileno())
        except Exception:
            fp.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def request(session, method, url, **kwargs):
    try:
        result = session.request(method, url, timeout=(5, 10), allow_redirects=False, **kwargs)
    except requests.RequestException:
        # 발송 성공 여부를 알 수 없는 전송 오류는 중복 알림 방지를 위해 재시도하지 않는다.
        raise KakaoError("request_outcome_unknown") from None
    if result.status_code == 429:
        raise KakaoError("rate_limited", retry=True)
    if not 200 <= result.status_code < 300:
        raise KakaoError(f"http_{result.status_code}")
    try:
        payload = result.json()
    except ValueError:
        raise KakaoError("invalid_response") from None
    if not isinstance(payload, dict):
        raise KakaoError("invalid_response")
    return payload


def bearer(tokens):
    return {"Authorization": "Bearer " + tokens["access_token"]}


def refresh(session, tokens):
    if not tokens.get("refresh_token") or not tokens.get("rest_api_key"):
        raise KakaoError("reauthorization_required")
    data = {"grant_type": "refresh_token", "client_id": tokens["rest_api_key"],
            "refresh_token": tokens["refresh_token"]}
    if tokens.get("client_secret"):
        data["client_secret"] = tokens["client_secret"]
    result = request(session, "POST", TOKEN, data=data)
    if not result.get("access_token"):
        raise KakaoError("refresh_failed")
    return {**tokens, "access_token": result["access_token"],
            "refresh_token": result.get("refresh_token") or tokens["refresh_token"]}


def identity(session, tokens):
    result = request(session, "GET", TOKEN_INFO, headers=bearer(tokens))
    if type(result.get("id")) is not int:
        raise KakaoError("identity_missing")
    return str(result["id"])


def message(event, link):
    stamp = event["detected_at"].astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
    camera = " ".join(event["camera_name"].split())[:32]
    reason = LABELS[event["event_type"]]
    target = {"web_url": link, "mobile_web_url": link}
    content = {"title": f"🚨 {reason} 상황 발생!",
               "description": f"시간: {stamp} (한국시간)\n영상: {camera}\n사건ID: {event['event_id']}",
               "link": target}
    image_url = (signed_url(event["event_id"], event.get("image_expires_at"))
                 or os.getenv("CCTV_KAKAO_IMAGE_URL", "").strip())
    if image_url.startswith("https://"):
        content["image_url"] = image_url
    return {"object_type": "feed", "content": content,
            "buttons": [{"title": "대시보드 확인", "link": target}]}


def send(session, event, link):
    path = token_path(event["recipient_user_id"])
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise KakaoError("unsafe_secret_permissions")
    tokens = json.loads(path.read_text(encoding="utf-8"))
    if tokens["subject"] != event["provider_subject"]:
        raise KakaoError("recipient_mismatch")
    payload = {"template_object": json.dumps(message(event, link), ensure_ascii=False)}
    try:
        result = request(session, "POST", MEMO, data=payload, headers=bearer(tokens))
    except KakaoError as error:
        if str(error) != "http_401":
            raise
        tokens = refresh(session, tokens)
        if identity(session, tokens) != event["provider_subject"]:
            raise KakaoError("recipient_mismatch")
        store_tokens(event["recipient_user_id"], tokens)
        result = request(session, "POST", MEMO, data=payload, headers=bearer(tokens))
    if result.get("result_code") != 0:
        raise KakaoError("message_rejected")


def claim(db):
    with db.transaction():
        # 결과를 확인하지 못한 만료 임대는 중복 알림 없이 실패로 남긴다.
        db.execute("""
            UPDATE notification_deliveries SET status = 'failed', last_error_code = 'lease_expired',
                lease_token = NULL, lease_expires_at = NULL
            WHERE status = 'delivering' AND lease_expires_at <= clock_timestamp()
        """)
        return db.execute("""
            WITH candidate AS (
                SELECT n.id, a.provider_subject, e.event_type, e.detected_at, c.name AS camera_name,
                    i.expires_at AS image_expires_at
                FROM notification_deliveries n JOIN oauth_accounts a ON a.id = n.oauth_account_id
                JOIN users u ON u.id = n.recipient_user_id
                JOIN detection_events e ON e.id = n.event_id
                JOIN inference_runs r ON r.id = e.run_id JOIN cameras c ON c.id = r.camera_id
                LEFT JOIN kakao_event_images i ON i.event_id = e.id
                WHERE n.status IN ('queued', 'retry_wait') AND n.next_attempt_at <= clock_timestamp()
                  AND n.attempt_count < n.max_attempts AND a.is_enabled AND u.is_active
                  AND e.deleted_at IS NULL AND u.disabled_at IS NULL AND c.deleted_at IS NULL
                  AND a.token_secret_ref = 'local-kakao:' || u.id::text
                ORDER BY n.next_attempt_at, n.created_at, n.id LIMIT 1 FOR UPDATE OF n SKIP LOCKED
            )
            UPDATE notification_deliveries n SET status = 'delivering', attempt_count = attempt_count + 1,
                lease_token = gen_random_uuid(), lease_expires_at = clock_timestamp() + interval '90 seconds'
            FROM candidate WHERE n.id = candidate.id
            RETURNING n.id, n.event_id, n.recipient_user_id, n.lease_token, n.attempt_count, n.max_attempts,
                candidate.provider_subject, candidate.event_type, candidate.detected_at, candidate.camera_name,
                candidate.image_expires_at
        """).fetchone()


def finish(db, event, error):
    status = "sent" if error is None else (
        "retry_wait" if error.retry and event["attempt_count"] < event["max_attempts"] else "failed")
    delay = min(900, 60 * 2 ** min(4, event["attempt_count"] - 1))
    db.execute("""
        UPDATE notification_deliveries SET status = %s, last_error_code = %s,
            sent_at = CASE WHEN %s = 'sent' THEN clock_timestamp() END,
            next_attempt_at = clock_timestamp() + make_interval(secs => %s),
            lease_token = NULL, lease_expires_at = NULL
        WHERE id = %s AND lease_token = %s AND status = 'delivering'
    """, (status, str(error) if error else None, status, delay, event["id"], event["lease_token"]))
    print(f"카카오톡 알림 {event['id']} · {status}" + (f" · {error}" if error else ""), flush=True)


def deliver_next(db, session, link):
    event = claim(db)
    if not event:
        return False
    error = None
    with db.transaction():
        lock_recipient(db, event["recipient_user_id"])
        if not db.execute("""
            SELECT 1 FROM notification_deliveries n JOIN oauth_accounts a ON a.id = n.oauth_account_id
            WHERE n.id = %s AND n.lease_token = %s AND n.status = 'delivering'
                AND n.lease_expires_at > clock_timestamp() AND a.is_enabled
        """, (event["id"], event["lease_token"])).fetchone():
            return True
        try:
            send(session, event, link)
        except KakaoError as exc:
            error = exc
        except (OSError, ValueError, KeyError, TypeError):
            error = KakaoError("invalid_local_credentials")
        finish(db, event, error)
    return True


def lock_recipient(db, user_id):
    db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (REF + str(user_id),))


def save_link(db, user_id, subject, tokens):
    with db.transaction():
        lock_recipient(db, user_id)
        account = db.execute("""
            INSERT INTO oauth_accounts (user_id, provider, provider_subject, token_secret_ref)
            VALUES (%s, 'kakao', %s, %s)
            ON CONFLICT (provider, provider_subject) DO UPDATE
            SET token_secret_ref = EXCLUDED.token_secret_ref, is_enabled = true
            WHERE oauth_accounts.user_id = EXCLUDED.user_id
            RETURNING id
        """, (user_id, subject, REF + str(user_id))).fetchone()
        if not account:
            raise KakaoError("kakao_account_already_linked")
        db.execute("""
            UPDATE oauth_accounts SET is_enabled = false
            WHERE user_id = %s AND provider = 'kakao' AND token_secret_ref = %s AND id <> %s
        """, (user_id, REF + str(user_id), account["id"]))
        db.execute("""
            UPDATE notification_deliveries n SET status = 'cancelled', lease_token = NULL, lease_expires_at = NULL
            FROM oauth_accounts a
            WHERE n.oauth_account_id = a.id AND a.user_id = %s AND a.token_secret_ref = %s
              AND NOT a.is_enabled AND n.status IN ('queued', 'retry_wait', 'delivering')
        """, (user_id, REF + str(user_id)))
        store_tokens(user_id, {**tokens, "subject": subject})


def link_account(db, session, username, legacy):
    user = db.execute("""
        SELECT id FROM users WHERE username = %s AND role = 'owner'
            AND is_active AND disabled_at IS NULL AND NOT is_test_admin
    """, (username.lower(),)).fetchone()
    if not user:
        raise KakaoError("active_owner_required")
    if legacy:
        config = json.loads((ROOT / "kakao_local_config.json").read_text(encoding="utf-8-sig"))
        token_file = ROOT / "kakao_tokens.json"
        history = json.loads(token_file.read_text(encoding="utf-8-sig")) if token_file.is_file() else {}
        current = history.get("current_user_id")
        current_tokens = history.get("users", {}).get(str(current), {})
        tokens = {**config, **current_tokens}
    else:
        current = None
        tokens = {"rest_api_key": getpass.getpass("카카오 REST API 키: ").strip(),
                  "access_token": getpass.getpass("카카오 액세스 토큰: ").strip(),
                  "refresh_token": getpass.getpass("카카오 리프레시 토큰(선택): ").strip()}
    tokens = {key: tokens[key] for key in ("rest_api_key", "access_token", "refresh_token", "client_secret")
              if tokens.get(key)}
    if os.getenv("CCTV_KAKAO_CLIENT_SECRET"):
        tokens["client_secret"] = os.environ["CCTV_KAKAO_CLIENT_SECRET"]
    try:
        subject = identity(session, tokens)
    except KakaoError as error:
        if str(error) != "http_401":
            raise
        tokens = refresh(session, tokens)
        subject = identity(session, tokens)
    if current is not None and str(current) != subject:
        raise KakaoError("legacy_account_mismatch")
    save_link(db, user["id"], subject, tokens)
    print("카카오톡 수신 계정을 연결했습니다. 메시지는 발송하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    link = commands.add_parser("link", help="관제 사용자에 카카오 계정을 연결한다.")
    link.add_argument("username")
    link.add_argument("--legacy", action="store_true", help="기존 GUI의 로컬 인증 파일을 읽는다.")
    commands.add_parser("serve", help="예약된 카카오톡 알림 발송을 시작한다.")
    args = parser.parse_args()
    url = os.getenv("CCTV_KAKAO_LINK_URL", "")
    if args.command == "serve":
        try:
            parts = urlsplit(url)
            local_http = parts.scheme == "http" and parts.hostname in {"127.0.0.1", "localhost", "::1"}
            if (not parts.hostname or "@" in parts.netloc or parts.query
                    or (parts.scheme != "https" and not local_http)):
                raise ValueError
        except ValueError:
            parser.error("카카오 앱에 등록된 관제 주소를 CCTV_KAKAO_LINK_URL에 설정하세요.")
    try:
        with connect() as db, requests.Session() as session:
            db.autocommit = True
            if not db.execute("SELECT pg_try_advisory_lock(%s) AS locked", (WORKER_LOCK,)).fetchone()["locked"]:
                raise KakaoError("notification_worker_already_running")
            if args.command == "link":
                link_account(db, session, args.username, args.legacy)
            else:
                print("카카오톡 알림 대기열을 처리합니다.", flush=True)
                next_cleanup = 0
                while True:
                    if monotonic() >= next_cleanup:
                        db.execute("DELETE FROM kakao_event_images WHERE expires_at <= clock_timestamp()")
                        next_cleanup = monotonic() + 60
                    if not deliver_next(db, session, url):
                        sleep(1)
    except KeyboardInterrupt:
        pass
    except KakaoError as error:
        raise SystemExit(f"카카오톡 설정 오류: {error}") from None
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit("카카오톡 로컬 설정을 읽을 수 없습니다.") from None


if __name__ == "__main__":
    main()
