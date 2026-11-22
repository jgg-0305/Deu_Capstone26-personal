"""실제 카카오 계정에 접속하지 않고 알림과 갱신 요청을 검증한다."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import requests

from backend import kakao_notify as kakao


def response(status, payload):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


class KakaoTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        guard = patch.object(kakao, "SECRET_DIR", Path(temp.name))
        guard.start()
        self.addCleanup(guard.stop)
        self.user_id = uuid4()
        self.event = {"recipient_user_id": self.user_id, "provider_subject": "123",
                      "event_type": "theft_suspected", "camera_name": "입구",
                      "event_id": uuid4(),
                      "detected_at": datetime(2026, 1, 1, tzinfo=timezone.utc)}
        kakao.store_tokens(self.user_id, {"access_token": "old", "refresh_token": "refresh",
                                         "rest_api_key": "key", "subject": "123"})

    def test_unauthorized_send_refreshes_only_for_same_account(self):
        session = Mock()
        session.request.side_effect = [response(401, {}), response(200, {"access_token": "renewed"}),
                                       response(200, {"id": 123}), response(200, {"result_code": 0})]
        kakao.send(session, self.event, "https://example.test")
        self.assertEqual(session.request.call_count, 4)
        call = session.request.call_args
        self.assertEqual(call.args, ("POST", kakao.MEMO))
        self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer renewed")
        template = json.loads(call.kwargs["data"]["template_object"])
        self.assertEqual(template["object_type"], "feed")
        self.assertIn("도난 의심", template["content"]["title"])
        self.assertIn(str(self.event["event_id"]), template["content"]["description"])
        self.assertEqual(template["buttons"][0]["title"], "대시보드 확인")

    def test_unknown_send_outcome_is_not_retried(self):
        session = Mock()
        session.request.side_effect = requests.Timeout()
        with self.assertRaises(kakao.KakaoError) as caught:
            kakao.send(session, self.event, "https://example.test")
        self.assertFalse(caught.exception.retry)
        session.request.assert_called_once()

    def test_public_photo_url_is_included_in_feed_when_configured(self):
        event = {**self.event, "image_expires_at": datetime.now(timezone.utc) + timedelta(hours=2)}
        with patch.dict("os.environ", CCTV_KAKAO_IMAGE_BASE_URL="https://images.example.com",
                        CCTV_KAKAO_IMAGE_SECRET="test-image-secret-not-for-production"):
            card = kakao.message(event, "https://dashboard.example.com")
        self.assertTrue(card["content"]["image_url"].startswith(
            f"https://images.example.com/kakao/event-images/{event['event_id']}.jpg?expires="))
        self.assertIn("&sig=", card["content"]["image_url"])
