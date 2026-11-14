"""짧은 관절 검출 누락과 추적 연결이 행동 입력을 훼손하지 않는지 검증한다."""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from torch import nn

from core.tracker import HumanTracker
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from core.tracking_geometry import match_keypoints


class TrackingContinuityTests(unittest.TestCase):
    def make_buffer(self):
        with patch.object(TrackBehaviorBuffer, '_build_and_load', return_value=nn.Identity()):
            model = TrackBehaviorBuffer('unused', max_missing=8, max_gap=3)
        probabilities = np.zeros(8, dtype=np.float32)
        probabilities[0], probabilities[6] = .2, .8
        model._infer = Mock(return_value=probabilities)
        return model

    def test_one_missing_frame_does_not_restart_thirty_frame_collection(self):
        model = self.make_buffer()
        tracks = [{'track_id': 1, 'bbox': [0, 0, 40, 60], 'observed': True}]
        points = {1: np.ones((17, 3))}
        for _ in range(29):
            model.update(tracks, points)
        self.assertEqual(model.update([], {}), {})
        self.assertEqual(model.progress(1)['collected'], 30)
        self.assertEqual(model.progress(1)['state'], 'tracking_gap')
        for _ in range(5):
            result = model.update(tracks, points)
        self.assertEqual(model.inference_count, 1)
        self.assertEqual(result[1][0], 6)
        self.assertEqual(model.reset_count, 0)

    def test_lost_person_produces_no_prediction_and_long_gap_requires_new_window(self):
        model = self.make_buffer()
        tracks = [{'track_id': 1, 'bbox': [0, 0, 40, 60], 'observed': True}]
        points = {1: np.ones((17, 3))}
        for _ in range(30):
            model.update(tracks, points)
        for _ in range(4):
            self.assertEqual(model.update([], {}), {})
        self.assertEqual(model.update(tracks, points), {})
        self.assertEqual(model.progress(1)['collected'], 1)
        self.assertEqual(model.reset_count, 1)
        for _ in range(9):
            model.update([], {})
        self.assertNotIn(1, model._bufs)

    def test_missing_detections_preserve_confirmed_track_without_borrowed_keypoints(self):
        track = Mock()
        track.is_confirmed.return_value = True
        track.time_since_update = 1
        track.track_id = '7'
        track.to_ltrb.return_value = [1, 2, 50, 90]
        with patch('core.tracker.DeepSort') as deep:
            deep.return_value.update_tracks.return_value = [track]
            tracker = HumanTracker()
            result = tracker.update([], np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['track_id'], 7)
        self.assertFalse(result[0]['observed'])
        self.assertIsNone(result[0]['keypoints'])
        self.assertEqual(match_keypoints(result, [{'bbox': [1, 2, 50, 90], 'keypoints': np.ones((17, 3))}]), {7: None})

    def test_uncertain_anomaly_preserves_the_actual_candidate(self):
        model = self.make_buffer()
        probabilities = np.zeros(8, dtype=np.float32)
        probabilities[0], probabilities[4], probabilities[6] = .3, .25, .45
        model._infer.return_value = probabilities
        for _ in range(30):
            model.update([{'track_id': 1, 'bbox': [0, 0, 40, 60]}], {1: np.ones((17, 3))})
        progress = model.progress(1)
        self.assertEqual(progress['state'], 'uncertain')
        self.assertEqual(progress['candidate']['label'], 6)
        self.assertAlmostEqual(progress['candidate']['score'], .45, places=6)

    def test_insufficient_observation_does_not_reuse_old_confident_prediction(self):
        model = self.make_buffer()
        track = [{'track_id': 1, 'bbox': [0, 0, 40, 60]}]
        points = {1: np.ones((17, 3))}
        for _ in range(30):
            model.update(track, points)
        for _ in range(9):
            model.update([], {})
            model.update([], {})
            result = model.update(track, points)
        self.assertEqual(model.progress(1)['state'], 'insufficient_keypoints')
        self.assertEqual(result, {})

    def test_keypoints_follow_the_trackers_association_in_overlapping_people(self):
        first, second = np.ones((17, 3)), np.ones((17, 3)) * 2
        tracks = [{'track_id': 1, 'bbox': [0, 0, 40, 60], 'keypoints': first},
                  {'track_id': 2, 'bbox': [0, 0, 40, 60], 'keypoints': second}]
        result = match_keypoints(tracks, [{'bbox': [0, 0, 40, 60], 'keypoints': second}])
        self.assertIs(result[1], first)
        self.assertIs(result[2], second)


if __name__ == '__main__':
    unittest.main()
