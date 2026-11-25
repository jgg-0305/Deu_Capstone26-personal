# 카카오 카드에서 쓰는 사건 사진만 중계하는 로컬 서버다. 요청 경로를 사건 이미지 형식으로 제한하고 내부 관제 서버에 전달해 이미지와 서명
# 검증 결과를 반환한다.
"""ngrok에 연결할 사건 사진 전용 로컬 게이트웨이."""

import argparse
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, HTTPServer
import re
from urllib.parse import parse_qs, urlencode, urlsplit

from backend.kakao_images import IMAGE_LIMIT

PHOTO_PATH = re.compile(
    r"/kakao/event-images/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg"
)


class Gateway(HTTPServer):
    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(5)
        return sock, address


class PhotoHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        # 접근 로그에 서명 URL을 남기지 않는다.
        pass

    def do_GET(self):
        self.photo()

    do_HEAD = do_GET

    def photo(self):
        if len(self.path) > 256:
            self.send_error(404)
            return
        try:
            target = urlsplit(self.path)
            params = parse_qs(target.query, strict_parsing=True, max_num_fields=2)
            expires, sig = params.get("expires", []), params.get("sig", [])
            if (target.scheme or target.netloc or target.fragment or not PHOTO_PATH.fullmatch(target.path)
                    or len(expires) != 1 or len(sig) != 1
                    or not re.fullmatch(r"[0-9]{1,12}", expires[0])
                    or not re.fullmatch(r"[0-9a-f]{64}", sig[0])):
                raise ValueError
        except ValueError:
            self.send_error(404)
            return
        upstream = HTTPConnection("127.0.0.1", self.server.api_port, timeout=3)
        try:
            query = urlencode({"expires": expires[0], "sig": sig[0]})
            upstream.request("GET", target.path + "?" + query)
            response = upstream.getresponse()
            if response.status != 200 or response.getheader("Content-Type") != "image/jpeg":
                self.send_error(404)
                return
            body = response.read(IMAGE_LIMIT + 1)
        except (OSError, HTTPException):
            self.send_error(502)
            return
        finally:
            upstream.close()
        if len(body) > IMAGE_LIMIT:
            self.send_error(502)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command == "GET":
            self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--api-port", type=int, default=8000)
    args = parser.parse_args()
    with Gateway(("127.0.0.1", args.port), PhotoHandler) as server:
        server.api_port = args.api_port
        print(f"사건 사진 전용 게이트웨이: 127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
