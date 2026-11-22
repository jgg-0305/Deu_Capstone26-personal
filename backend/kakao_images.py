"""카카오톡 카드가 읽을 수 있는 단기 사건 이미지 URL을 만든다."""

import base64
import binascii
import hmac
from ipaddress import ip_address
import os
import time
from urllib.parse import urlencode, urlsplit

IMAGE_LIMIT = 192 * 1024
IMAGE_BASE64_LIMIT = 4 * ((IMAGE_LIMIT + 2) // 3)
URL_TTL = 3600


def decode_jpeg(value):
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("사진 인코딩이 올바르지 않습니다.") from None
    if not 4 <= len(data) <= IMAGE_LIMIT or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
        raise ValueError("지원되는 크기의 JPEG 사진이 아닙니다.")
    return data


def settings():
    base = os.getenv("CCTV_KAKAO_IMAGE_BASE_URL", "").strip().rstrip("/")
    secret = os.getenv("CCTV_KAKAO_IMAGE_SECRET", "").strip()
    if not base or len(secret) < 32:
        return None
    try:
        parts = urlsplit(base)
        host = parts.hostname
        if (parts.scheme != "https" or not host or parts.username is not None
                or parts.path or parts.query or parts.fragment or "." not in host
                or host.endswith((".localhost", ".local", ".internal"))):
            return None
        try:
            address = ip_address(host)
        except ValueError:
            pass
        else:
            if not address.is_global:
                return None
    except ValueError:
        return None
    return base, secret.encode()


def signature(secret, event_id, expires):
    return hmac.digest(secret, f"kakao-event-image:v1:{event_id}:{expires}".encode(), "sha256").hex()


def signed_url(event_id, stored_until):
    config = settings()
    if config is None or stored_until is None:
        return None
    base, secret = config
    now = int(time.time())
    expires = min(int(stored_until.timestamp()), now + URL_TTL)
    if expires <= now:
        return None
    query = urlencode({"expires": expires, "sig": signature(secret, event_id, expires)})
    return f"{base}/kakao/event-images/{event_id}.jpg?{query}"


def valid_signature(event_id, expires, sig):
    config = settings()
    now = int(time.time())
    return (config is not None and now < expires <= now + URL_TTL
            and hmac.compare_digest(signature(config[1], event_id, expires).encode(), sig.encode()))
