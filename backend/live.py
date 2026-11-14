"""허용된 IP 카메라를 브라우저용 JPEG 스트림으로 중계한다."""

import asyncio
from ipaddress import ip_address
import json
import os
from threading import BoundedSemaphore
from time import monotonic
from urllib.parse import urlsplit, urlunsplit

import anyio
from fastapi import HTTPException

SLOTS = BoundedSemaphore(8)
BOUNDARY = "cctv-frame"


def public_rtsp_url(value, *, configured=False):
    try:
        if any(char.isspace() or ord(char) < 32 or char in "#\\" for char in value):
            raise ValueError
        parts = urlsplit(value)
        if parts.scheme != "rtsp" or "%" in (parts.hostname or ""):
            raise ValueError
        if not configured and "@" in parts.netloc:
            raise ValueError
        if parts.netloc.count("@") > 1:
            raise ValueError
        address = ip_address(parts.hostname)
        if address.is_multicast or address.is_unspecified or address.is_link_local:
            raise ValueError
        port = parts.port if parts.port is not None else 554
        if not 1 <= port <= 65535:
            raise ValueError
        host = f"[{address}]" if address.version == 6 else str(address)
        return urlunsplit(("rtsp", f"{host}:{port}", parts.path or "/", parts.query, ""))
    except ValueError:
        raise HTTPException(422, "인증 정보를 제외한 RTSP 주소(IP, 포트, 경로)를 입력하세요.") from None


def resolve_camera(username, public_url):
    # 회원가입 계정에 서버 내부 네트워크 접근을 자동으로 허용하지 않는다.
    accounts = os.getenv("CCTV_RTSP_USERS", "").split(",")
    if username not in {account.strip() for account in accounts if account.strip()}:
        raise HTTPException(403, "이 계정에 RTSP 사용 권한이 없습니다. 서버 설정을 확인하세요.")
    try:
        urls = json.loads(os.getenv("CCTV_RTSP_URLS", "[]"))
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            raise ValueError
        allowed = {public_rtsp_url(url, configured=True): url for url in urls}
        if len(allowed) != len(urls):
            raise ValueError
    except (HTTPException, ValueError):
        raise HTTPException(503, "서버의 RTSP 설정을 확인하세요.") from None
    if public_url not in allowed:
        raise HTTPException(422, "서버에서 허용한 RTSP 주소를 입력하세요.")
    return allowed[public_url]


def open_capture(url):
    import cv2

    # RTSP 접속은 서버에서 지정한 TCP 전송으로 한정한다.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
    ])


def read_jpeg(capture, last_frame):
    import cv2

    while True:
        ready, frame = capture.read()
        if not ready:
            return None
        if monotonic() - last_frame >= 0.1:
            break
    height, width = frame.shape[:2]
    scale = min(1, 960 / width, 540 / height)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ready, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return data.tobytes() if ready else None


async def frames(url, authorized):
    # 동시 연결 수를 제한하고 화면을 닫으면 카메라 연결을 해제한다.
    if not SLOTS.acquire(blocking=False):
        return
    capture = None
    try:
        with anyio.CancelScope(shield=True):
            capture = await anyio.to_thread.run_sync(open_capture, url)
        last_frame = 0
        checked = monotonic()
        while capture.isOpened():
            await asyncio.sleep(0)
            if monotonic() - checked >= 10:
                if not await anyio.to_thread.run_sync(authorized):
                    break
                checked = monotonic()
            data = await anyio.to_thread.run_sync(read_jpeg, capture, last_frame)
            if data is None:
                break
            last_frame = monotonic()
            yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n"
                   .encode("ascii") + data + b"\r\n")
    except (ImportError, RuntimeError):
        # 카메라 주소나 인증 정보가 담길 수 있는 예외는 응답에 노출하지 않는다.
        return
    finally:
        try:
            if capture is not None:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(capture.release)
        finally:
            SLOTS.release()
