"""Regression coverage for fresh theft confirmation and weak item observations."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.article_tracking import ArticleTracker, SmokingMotionEvidence, article_appearance
from core.kpt_behavior_classifier import TrackBehaviorBuffer


def distribution(theft):
    return np.array([1. - theft, 0., 0., 0., 0., 0., theft, 0.], dtype=np.float32)


class TheftConfirmationTests(unittest.TestCase):
    def test_requires_two_fresh_windows_and_clears_confirmation_on_gap(self):
        model = torch.nn.Identity()
        model.behavior_input_config = {"sample_fps": 3}
        with patch.object(TrackBehaviorBuffer, "_build_and_load", return_value=model):
            buffer = TrackBehaviorBuffer("unused", seq_len=3, stride=2, device="cpu",
                                         abn_conf_thr=.9, theft_confirmation_threshold=.9)
        track = {"track_id": 1, "bbox": [0, 0, 100, 100]}
        features = {1: np.ones(34, dtype=np.float32)}

        def update(index):
            return buffer.update([track], {}, index / 3, features=features)

        with patch.object(buffer, "_infer", side_effect=[distribution(v) for v in (0., .98, .99, .99)]):
            for i in range(5):
                update(i)
            self.assertEqual(update(5)[1][0], 0)
            self.assertEqual(update(5)[1][0], 0)
            update(6)
            label, score = update(7)[1]
            self.assertEqual(label, 6)
            self.assertAlmostEqual(score, .98)
            self.assertEqual(buffer.progress(1)["state"], "classified")
            buffer.update([], {}, 8 / 3)
            self.assertEqual(update(9)[1][0], 0)


    def test_abandonment_confirmation_requires_an_opt_in_and_two_distinct_windows(self):
        for enabled, low_reach in ((False, True), (True, False), (True, True)):
            with self.subTest(enabled=enabled, low_reach=low_reach):
                model = torch.nn.Identity()
                model.behavior_input_config = {"sample_fps": 3, "confirm_abandonment": enabled}
                with patch.object(TrackBehaviorBuffer, "_build_and_load", return_value=model):
                    buffer = TrackBehaviorBuffer("unused", seq_len=3, stride=2, device="cpu",
                                                 abn_conf_thr=.9, theft_confirmation_threshold=.9)
                track = {"track_id": 1, "bbox": [0, 0, 100, 100]}
                pose = np.full((17, 2), .5, dtype=np.float32)
                if low_reach:
                    pose[9, 1] = .7
                features = {1: pose.flatten()}
                abandon = np.array([.02, 0., 0., 0., 0., .98, 0., 0.], dtype=np.float32)
                with patch.object(buffer, "_infer", side_effect=[distribution(0.), abandon, abandon]):
                    for i in range(5):
                        buffer.update([track], {}, i / 3, features=features)
                    for i in (5, 5, 6):
                        self.assertEqual(buffer.update([track], {}, i / 3, features=features)[1][0], 0)
                    self.assertEqual(buffer.update([track], {}, 7 / 3, features=features)[1][0],
                                     5 if enabled and low_reach else 0)
                    self.assertAlmostEqual(buffer.progress(1)["theft_score"], 0.)


    def test_early_assault_requires_an_observed_nearby_person(self):
        for nearby in (False, True):
            with self.subTest(nearby=nearby):
                model = torch.nn.Identity()
                model.behavior_input_config = {"sample_fps": 3, "confirm_assault": True}
                with patch.object(TrackBehaviorBuffer, "_build_and_load", return_value=model):
                    buffer = TrackBehaviorBuffer("unused", seq_len=3, stride=2, device="cpu",
                                                 abn_conf_thr=.9, theft_confirmation_threshold=.9)
                person = {"track_id": 1, "bbox": [0, 0, 100, 100]}
                other = {"track_id": 2, "bbox": [45, 0, 145, 100], "observed": nearby}
                features = {1: np.ones(34, dtype=np.float32), 2: np.zeros(34, dtype=np.float32)}
                assault = np.array([.04, 0., 0., 0., 0., 0., 0., .96], dtype=np.float32)
                with patch.object(buffer, "_infer", side_effect=[distribution(0.), assault, assault]):
                    for i in range(8):
                        result = buffer.update([person, other], {}, i / 3, features=features)
                self.assertEqual(result[1][0], 7 if nearby else 0)

    def test_arson_specialist_requires_main_model_agreement(self):
        model = torch.nn.Identity()
        model.arson_specialist = torch.nn.Identity()
        model.behavior_input_config = {"sample_fps": 3}
        with patch.object(TrackBehaviorBuffer, "_build_and_load", return_value=model):
            buffer = TrackBehaviorBuffer("unused", seq_len=1, stride=1, device="cpu",
                                         abn_conf_thr=.9, theft_confirmation_threshold=.9)
        track = {"track_id": 1, "bbox": [0, 0, 100, 100]}
        features = {1: np.ones(34, dtype=np.float32)}
        arson = np.array([.35, 0., 0., .65, 0., 0., 0., 0.], dtype=np.float32)
        specialist = np.array([.07, 0., 0., .93, 0., 0., 0., 0.], dtype=np.float32)
        with patch.object(buffer, "_infer_model", return_value=specialist), \
             patch.object(buffer, "_infer", side_effect=[distribution(0.), arson]):
            self.assertEqual(buffer.update([track], {}, 0., features=features)[1][0], 0)
            buffer.reset()
            label, score = buffer.update([track], {}, 1 / 3, features=features)[1]
            self.assertEqual(label, 3)
            self.assertAlmostEqual(score, .93)


class ArticleContinuationTests(unittest.TestCase):
    def test_weak_detection_bridges_existing_carry_without_adding_a_hit(self):
        points = np.zeros((17, 3))
        points[10] = [50, 50, 1]
        points[12] = [50, 70, 1]
        person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
        hist = np.ones(32, dtype=np.float32) / 32
        detection = (45, 45, 55, 55)
        tracker = ArticleTracker()

        self.assertFalse(tracker.update([person], [(detection, .32, hist)], 0.))
        self.assertFalse(tracker.tracks)
        tracker.update([person], [(detection, .75, hist)], 1 / 3)
        tracker.update([person], [], 2 / 3)
        self.assertFalse(tracker.update([person], [(detection, .32, hist)], 1.))
        state = tracker.tracks[(1, 10)]
        self.assertEqual((state.hits, state.last, tracker.weak_bridged_frames), (1, 1 / 3, 1))
        tracker.update([person], [(detection, .75, hist)], 4 / 3)
        self.assertEqual(tracker.tracks[(1, 10)].hits, 2)
        self.assertEqual(tracker.reacquired_tracks, 1)


    def test_pocket_recovery_requires_same_item_after_a_short_gap(self):
        hist = np.eye(2, dtype=np.float32)[0]
        other_hist = np.eye(2, dtype=np.float32)[1]
        for recovered_hist, expected in ((hist, True), (other_hist, False)):
            with self.subTest(same_item=expected):
                points = np.zeros((17, 3))
                points[10] = [80, 40, 1]
                points[12] = [50, 65, 1]
                person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
                tracker = ArticleTracker(pocket_recovery=True)

                for slot in range(3):
                    tracker.update([person], [((81, 40, 93, 48), .65, hist)], slot / 3)
                for slot in (3, 4):
                    self.assertFalse(tracker.update([person], [], slot / 3))

                points[10] = [51, 55, 1]
                partial = ((52, 59, 58, 63), .55, recovered_hist)
                self.assertFalse(tracker.update([person], [partial], 5 / 3))
                self.assertEqual(1 in tracker.update([person], [], 2.), expected)


    def test_extended_bridge_requires_visible_item_appearance(self):
        red = np.full((100, 100, 3), [0, 0, 255], dtype=np.uint8)
        blue = np.full_like(red, [255, 0, 0])
        for final_frame, continued in ((red, True), (blue, False)):
            with self.subTest(continued=continued):
                points = np.zeros((17, 3))
                points[12] = [50, 65, 1]
                person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
                tracker = ArticleTracker()
                for slot, (x, y) in enumerate(((80, 45), (70, 50), (60, 55), (54, 60))):
                    points[10] = [x, y, 1]
                    box = (x-4, y-4, x+4, y+4)
                    tracker.update([person], [(box, .65, article_appearance(red, box))], slot / 3, frame=red)

                with patch("core.article_tracking.follow_box", return_value=box):
                    for slot in (4, 5):
                        self.assertFalse(tracker.update([person], [], slot / 3, frame=red))
                    events = tracker.update([person], [], 2., frame=final_frame)
                self.assertEqual(1 in events, not continued)
                if continued:
                    carry = tracker.tracks[(1, 10)]
                    self.assertEqual((carry.hits, carry.last, carry.bridge_frames), (4, 1., 3))


    def test_live_compact_carry_can_conceal_without_an_intermediate_redetection(self):
        points = np.zeros((17, 3))
        points[12] = [50, 65, 1]
        person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
        hist = np.ones(32, dtype=np.float32) / 32
        tracker = ArticleTracker(pocket_recovery=True, continuous_carry=True)
        for slot, x in enumerate((72, 70, 67, 64, 56)):
            points[10] = [x, 64, 1]
            box = (x-4, 60, x+4, 68)
            self.assertFalse(tracker.update([person], [(box, .65, hist)], slot / 3))
        self.assertFalse(tracker.update([person], [], 5 / 3))
        self.assertIn(1, tracker.update([person], [], 2.))


    def test_strict_carry_checks_the_item_near_the_hip_instead_of_only_the_wrist(self):
        hist = np.ones(32, dtype=np.float32) / 32
        for strict in (False, True):
            with self.subTest(strict=strict):
                points = np.zeros((17, 3))
                points[12] = [50, 65, 1]
                person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
                tracker = ArticleTracker(strict_concealment=strict)
                for slot, (x, y) in enumerate(((80, 45), (70, 50), (60, 55), (54, 60))):
                    points[10] = [x, y, 1]
                    box = (x+7, y-10, x+15, y-2)
                    tracker.update([person], [(box, .65, hist)], slot / 3)
                tracker.update([person], [], 4 / 3)
                self.assertEqual(1 in tracker.update([person], [], 5 / 3), not strict)

    def test_strict_compact_carry_requires_a_reliable_wrist_at_the_hip(self):
        for wrist_x, confidence, expected in ((56, .95, True), (56, .39, False), (68, .95, False)):
            with self.subTest(wrist_x=wrist_x, confidence=confidence):
                points = np.zeros((17, 3))
                points[12] = [50, 65, 1]
                person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
                hist = np.ones(32, dtype=np.float32) / 32
                tracker = ArticleTracker(pocket_recovery=True, continuous_carry=True, strict_concealment=True)
                for slot, x in enumerate((72, 70, 67, 64, 56)):
                    points[10] = [x, 64, 1]
                    tracker.update([person], [((x-4, 60, x+4, 68), .65, hist)], slot / 3)
                tracker.update([person], [], 5 / 3)
                points[10] = [wrist_x, 64, confidence]
                self.assertEqual(1 in tracker.update([person], [], 2.), expected)

    def test_strict_compact_carry_recovers_a_visible_item_after_a_long_detection_gap(self):
        first_item = np.array([1., 0.], dtype=np.float32)
        for same_item in (True, False):
            with self.subTest(same_item=same_item):
                points = np.zeros((17, 3))
                points[12] = [50, 65, 1]
                person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
                tracker = ArticleTracker(pocket_recovery=True, continuous_carry=True, strict_concealment=True)
                for slot, x in enumerate((82, 83, 81, 82, 81)):
                    points[10] = [x, 65, .95]
                    tracker.update([person], [((x-4, 61, x+4, 69), .8, first_item)], slot / 3)
                for slot in range(5, 16):
                    self.assertFalse(tracker.update([person], [], slot / 3))
                visible_item = first_item if same_item else first_item[::-1].copy()
                for slot, x in enumerate((62, 60, 56), 16):
                    points[10] = [x, 65, .95]
                    box = (x-4, 61, x+4, 69)
                    self.assertFalse(tracker.update([person], [(box, .8, visible_item)], slot / 3))
                self.assertFalse(tracker.update([person], [], 19 / 3))
                self.assertEqual(1 in tracker.update([person], [], 20 / 3), same_item)

    def test_rtsp_carry_recovers_when_a_confirmed_sample_was_dropped(self):
        points = np.zeros((17, 3))
        points[12] = [50, 65, 1]
        person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
        appearance = np.array([1., 0.], dtype=np.float32)
        tracker = ArticleTracker(pocket_recovery=True, continuous_carry=True, strict_concealment=True,
                                 recover_compressed=True)
        for slot, x in enumerate((82, 81, 80)):
            points[10] = [x, 65, .98]
            tracker.update([person], [((x-4, 61, x+4, 69), .8, appearance)], slot / 3)
        for slot in range(3, 18):
            tracker.update([person], [], slot / 3)
        for slot, x in enumerate((64, 60, 56), 18):
            points[10] = [x, 65, .98]
            tracker.update([person], [((x-4, 61, x+4, 69), .44 if slot == 18 else .70, appearance)], slot / 3)
        self.assertFalse(tracker.update([person], [], 21 / 3))
        self.assertIn(1, tracker.update([person], [], 22 / 3))


class SmokingMotionTests(unittest.TestCase):
    def test_same_frame_cannot_confirm_and_evidence_expires(self):
        points = np.zeros((17, 3))
        points[0] = [50, 10, 1]
        points[10] = [51, 12, 1]
        person = {"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points}
        other = {**person, "track_id": 2}
        evidence = SmokingMotionEvidence()
        self.assertFalse(evidence.update([person], 0.))
        self.assertFalse(evidence.update([person], .1))
        self.assertFalse(evidence.update([other], 1 / 3))
        self.assertEqual(set(evidence.update([other], 2 / 3)), {2})
        points[10] = [50, 50, 1]
        self.assertEqual(set(evidence.update([other], 1.)), {2})
        self.assertFalse(evidence.update([other], 4.))
