import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

from backend.api import EventInput
from core.bag_transfer import BagTransferTracker


class BagTransferTests(unittest.TestCase):
    def test_web_event_accepts_the_bag_transfer_evidence(self):
        event = EventInput(lease_token=uuid4(), producer_event_key="item-into-bag",
                           event_type="theft_suspected", detection_method="rule",
                           rule_code="article_trajectory", rule_version="article-trajectory-v16",
                           detected_at=datetime.now(timezone.utc),
                           evidence={"reason": "article_disappeared_inside_bag"})
        self.assertEqual(event.evidence.reason, "article_disappeared_inside_bag")

    def play(self, *, bag_seen_twice=True, move_inside=True, hide_wrist=True, article_visible=False):
        tracker = BagTransferTracker()
        detector = Mock()
        bag = (42, 32, 70, 79)
        detector.detect.side_effect = [{}, {1: [(bag, .25)]},
                                       {1: [(bag, .40)]} if bag_seen_twice else {}]
        points = np.zeros((17, 3))
        person = {"bbox": [25, 0, 80, 100], "track_id": 1, "keypoints": points}
        appearance = np.array([1., 0.], dtype=np.float32)
        for slot, x in enumerate((82, 68, 60 if move_inside else 72)):
            points[10] = (x, 55, .2 if hide_wrist and slot else .95)
            detections = [((x - 3, 51, x + 3, 59), .62 if not slot else .31, appearance)]
            self.assertFalse(tracker.update([person], detections, slot / 3, None, detector))
        remaining = appearance if article_visible else appearance[::-1].copy()
        with patch("core.bag_transfer.article_appearance", return_value=remaining):
            return tracker.update([person], [], 1., None, detector)

    def test_accepts_a_visible_transfer_when_the_wrist_becomes_unreliable(self):
        self.assertEqual(self.play(), {1})

    def test_requires_bag_to_remain_visible_after_the_article_disappears(self):
        self.assertFalse(self.play(bag_seen_twice=False))

    def test_ignores_an_article_that_disappears_outside_the_bag(self):
        self.assertFalse(self.play(move_inside=False))

    def test_detector_miss_is_not_a_transfer_if_the_item_still_looks_visible(self):
        self.assertFalse(self.play(article_visible=True))
