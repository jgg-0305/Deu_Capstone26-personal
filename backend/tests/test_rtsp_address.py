"""IP 주소로 등록해도 서버의 카메라 허용 목록과 인증 정보가 유지되는지 확인한다."""

import json
import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.live import registered_camera_url, resolve_camera


class RtspAddressTests(unittest.TestCase):
    def setUp(self):
        self.public = "rtsp://192.0.2.10:554/stream1"
        self.private = "rtsp://camera:camera-test@192.0.2.10:554/stream1"
        self.env = patch.dict(os.environ, {
            "CCTV_RTSP_USERS": "operator",
            "CCTV_RTSP_URLS": json.dumps([self.private]),
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_single_stream_accepts_bare_ip_and_keeps_server_credentials(self):
        self.assertEqual(registered_camera_url("operator", "192.0.2.10"), self.public)
        self.assertEqual(registered_camera_url("operator", self.public), self.public)
        self.assertEqual(resolve_camera("operator", self.public), self.private)

    def test_unapproved_ip_credentials_and_account_are_rejected(self):
        for username, address, status in (
            ("operator", "192.0.2.11", 422),
            ("operator", self.private, 422),
            ("other", "192.0.2.10", 403),
        ):
            with self.subTest(status=status, address=address), self.assertRaises(HTTPException) as error:
                registered_camera_url(username, address)
            self.assertEqual(error.exception.status_code, status)

    def test_multiple_streams_require_full_address(self):
        with patch.dict(os.environ, {"CCTV_RTSP_URLS": json.dumps(
            [self.private, "rtsp://192.0.2.10:554/stream2"]
        )}):
            with self.assertRaises(HTTPException) as error:
                registered_camera_url("operator", "192.0.2.10")
            self.assertEqual(error.exception.status_code, 422)
            self.assertEqual(registered_camera_url("operator", self.public), self.public)
