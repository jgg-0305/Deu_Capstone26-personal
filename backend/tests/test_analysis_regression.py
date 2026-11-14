"""원본 GUI의 판정 기준과 웹 분석 상태 전달을 검증한다."""

from contextlib import contextmanager
import unittest
from unittest.mock import Mock, patch

import numpy as np
from torch import nn

from backend.api import AnalysisInput
from backend.worker import analyze
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from core.tracking_geometry import match_keypoints


class AnalysisRegressionTests(unittest.TestCase):
    def test_overlapping_tracks_keep_original_keypoint_mapping(self):
        points = np.ones((17, 3))
        tracks = [{"track_id": i, "bbox": [0, 0, 40, 60]} for i in (1, 2)]
        result = match_keypoints(tracks, [{"bbox": [0, 0, 40, 60], "keypoints": points}])
        self.assertIs(result[1], points)
        self.assertIs(result[2], points)

    def run_video(self, probability):
        calls = []
        frames = 40
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        tracks = [{"track_id": 1, "bbox": [0, 0, 40, 60]}]
        points = np.ones((17, 3), dtype=np.float32)

        @contextmanager
        def source(run, camera_id):
            run['total_frames'] = frames
            yield ((frame, i * 40) for i in range(frames))

        def request(method, path, body=None):
            calls.append((path, body))
            if path == '/runs':
                return {'id': 'test-run', 'lease_token': 'test-lease', 'source_type': 'sample'}

        api = Mock(request=request)
        probabilities = np.zeros(8, dtype=np.float32)
        probabilities[0], probabilities[6] = 1 - probability, probability
        pose = Mock(is_pose=True)
        pose.detect.return_value = [{'bbox': tracks[0]['bbox'], 'keypoints': points}]
        detector = Mock(names={0: 'stand', 1: 'pos'})
        detector.detect.return_value = []
        tracker = Mock()
        tracker.update.return_value = tracks
        rule = Mock()
        rule.update.return_value = set()
        with patch('backend.worker.verified_models', return_value=({'pose': '', 'object': '', 'behavior': ''}, {})), \
             patch('backend.worker.source_frames', source), \
             patch('backend.worker.HumanDetector', return_value=pose), \
             patch('backend.worker.ObjectDetector', return_value=detector), \
             patch('backend.worker.HumanTracker', return_value=tracker), \
             patch('backend.worker.TheftMonitor', return_value=rule), \
             patch.object(TrackBehaviorBuffer, '_build_and_load', return_value=nn.Identity()), \
             patch.object(TrackBehaviorBuffer, '_infer', return_value=probabilities):
            analyze(api, 'test-camera', AnalysisInput().threshold)
        self.assertEqual(detector.detect.call_count, frames)
        return calls

    def test_eighty_percent_anomaly_is_visible_and_saved(self):
        calls = self.run_video(.8)
        snapshots = [body['snapshot'] for path, body in calls if path.endswith('/snapshot')]
        final = snapshots[-1]
        self.assertEqual(final['phase'], 'completed')
        self.assertEqual(final['people'][0]['label'], 6)
        self.assertAlmostEqual(final['people'][0]['score'], .8, places=6)
        self.assertEqual(final['inference_count'], 3)
        self.assertEqual(final['pose_count'], 40)
        self.assertEqual(final['people'][0]['buffer']['collected'], 30)
        self.assertEqual(final['people'][0]['buffer']['state'], 'classified')
        events = [body for path, body in calls if path.endswith('/events')]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event_type'], 'theft_suspected')
        self.assertEqual(events[0]['source_frame_index'], 29)
        self.assertAlmostEqual(events[0]['model_score'], .8, places=6)

    def test_display_and_recording_thresholds_are_separate(self):
        calls = self.run_video(.55)
        final = [body['snapshot'] for path, body in calls if path.endswith('/snapshot')][-1]
        self.assertEqual(final['people'][0]['label'], 6)
        self.assertFalse(any(path.endswith('/events') for path, _ in calls))
        self.assertEqual(AnalysisInput().detection_interval, 1)
        self.assertEqual(AnalysisInput().detection_confidence, .4)


if __name__ == '__main__':
    unittest.main()
