"""운영자가 웹에서 등록한 RTSP 카메라의 네트워크 범위와 인증 정보를 관리한다."""

from ipaddress import ip_address, ip_network
import json
import os
from urllib.parse import quote, urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException

from backend.live import public_rtsp_url, require_rtsp_user, resolve_camera

PRIVATE_NETWORKS = tuple(map(ip_network, ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")))


def camera_networks():
    try:
        networks = tuple(ip_network(value.strip()) for value in os.getenv("CCTV_RTSP_NETWORKS", "").split(",")
                         if value.strip())
        if any(not any(net.version == private.version and net.subnet_of(private)
                       for private in PRIVATE_NETWORKS) for net in networks):
            raise ValueError
        return networks
    except ValueError:
        raise HTTPException(503, "카메라 등록 네트워크를 사설 CIDR로 설정하세요.") from None


def direct_url(public_url):
    parts = urlsplit(public_url)
    address = ip_address(parts.hostname)
    networks = camera_networks()
    if not any(address in net for net in networks):
        raise HTTPException(403, "웹 등록이 허용된 카메라 네트워크의 IP를 입력하세요.")
    try:
        ports = {int(port.strip()) for port in os.getenv("CCTV_RTSP_PORTS", "554").split(",")}
        if any(not 1 <= port <= 65535 for port in ports):
            raise ValueError
    except ValueError:
        raise HTTPException(503, "서버의 카메라 허용 포트 설정을 확인하세요.") from None
    if parts.port not in ports:
        raise HTTPException(403, "서버에서 허용한 카메라 RTSP 포트를 사용하세요.")
    return parts


def cipher():
    try:
        return Fernet(os.getenv("CCTV_RTSP_CREDENTIAL_KEY", "").strip().encode("ascii"))
    except ValueError:
        raise HTTPException(503, "서버의 카메라 인증 정보 암호화 키를 확인하세요.") from None


def prepare_camera(username, address, port, path):
    require_rtsp_user(username)
    try:
        host = ip_address(address.strip())
    except ValueError:
        raise HTTPException(422, "카메라 IP 주소를 입력하세요.") from None
    display_host = f"[{host}]" if host.version == 6 else str(host)
    url = public_rtsp_url(f"rtsp://{display_host}:{port}{path}")
    direct_url(url)
    return url


def encrypt_camera_login(camera_id, username, password):
    # 카메라 ID도 암호화해 다른 카메라의 인증 토큰을 대신 사용할 수 없게 한다.
    data = json.dumps([str(camera_id), username, password]).encode("utf-8")
    return cipher().encrypt(data).decode("ascii")


def resolve_source(username, source):
    public_url = source["source_uri"]
    if source["rtsp_camera_id"] is None:
        return resolve_camera(username, public_url)
    require_rtsp_user(username)
    parts = direct_url(public_url)
    token = source["rtsp_auth_token"]
    if token is None:
        return public_url
    try:
        camera_id, login, password = json.loads(cipher().decrypt(token.encode("ascii")))
        if camera_id != str(source["rtsp_camera_id"]) or not isinstance(login, str) or not isinstance(password, str):
            raise ValueError
    except (InvalidToken, ValueError):
        raise HTTPException(503, "카메라 인증 정보를 읽을 수 없습니다. 서버 암호화 키를 확인하세요.") from None
    auth = f"{quote(login, safe='')}:{quote(password, safe='')}"
    return urlunsplit(parts._replace(netloc=f"{auth}@{parts.netloc}"))
