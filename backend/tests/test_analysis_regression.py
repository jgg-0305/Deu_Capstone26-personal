"""원본 GUI의 판정 기준과 웹 분석 상태 전달을 검증한다."""

from contextlib import contextmanager
import unittest
from unittest.mock import Mock, patch

import numpy as np
from torch import nn

from backend.api import AnalysisInput
from backend.worker import EventEpisodes, RtspMediaClock, analyze
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from core.tracking_geometry import match_keypoints


class AnalysisRegressionTests(unittest.TestCase):
    def test_live_vandalism_needs_two_new_predictions_at_the_same_threshold(self):
        model = nn.Identity()
        model.behavior_input_config = {"sample_fps": 3}
        with patch.object(TrackBehaviorBuffer, "_build_and_load", return_value=model):
            buffer = TrackBehaviorBuffer("unused", seq_len=1, stride=1, device="cpu",
                                         abn_conf_thr=.9, theft_confirmation_threshold=.9,
                                         verify_live_vandalism=True)
        track = {"track_id": 1, "bbox": [0, 0, 100, 100]}
        features = {1: np.ones(34, np.float32)}
        damage = np.array([.03, 0., .97, 0., 0., 0., 0., 0.], np.float32)
        normal = np.array([.99, 0., .01, 0., 0., 0., 0., 0.], np.float32)
        with patch.object(buffer, "_infer", side_effect=[damage, normal, damage, damage]):
            self.assertEqual(buffer.update([track], {}, 0., features=features)[1][0], 0)
            self.assertEqual(buffer.progress(1)["state"], "uncertain")
            self.assertEqual(buffer.update([track], {}, 0., features=features)[1][0], 0)
            self.assertEqual(buffer.update([track], {}, 1 / 3, features=features)[1][0], 0)
            self.assertEqual(buffer.update([track], {}, 2 / 3, features=features)[1][0], 0)
            self.assertEqual(buffer.update([track], {}, 1., features=features)[1][0], 2)

    def test_rtsp_capture_bursts_keep_the_encoder_frame_spacing(self):
        clock = RtspMediaClock(3)
        actual = [clock.timestamp(tick, 0, arrival) for tick, arrival in
                  ((8, 1200), (9, 1210), (10, 1212), (10, 2200), (12, 2530))]
        for i, timestamp in enumerate(actual):
            self.assertAlmostEqual(timestamp, 1200 + i * 1000 / 3)

    def test_bag_rule_resumes_display_without_creating_another_event(self):
        episodes = EventEpisodes()
        rule = (1, "theft_suspected", "rule")
        self.assertEqual(episodes.started([rule], 1000, {1: 3.0}), [rule])
        self.assertEqual(episodes.started([], 1333, {1: 3.0}), [])
        self.assertEqual(episodes.started([rule], 3000, {1: 3.0}), [])
        self.assertEqual(episodes.started([], 3333), [])
        self.assertEqual(episodes.started([rule], 5000, {1: 7.0}), [rule])

    def test_theft_handoff_keeps_one_incident_per_track(self):
        episodes = EventEpisodes()
        rule = (1, "theft_suspected", "rule")
        model = (1, "theft_suspected", "model")
        other = (2, "theft_suspected", "model")
        self.assertEqual(episodes.started([rule, model], 1000), [rule])
        self.assertEqual(episodes.started([rule], 1333), [])
        self.assertEqual(episodes.started([], 1667), [])
        self.assertEqual(episodes.started([model, other], 2000), [other])
        self.assertEqual(episodes.started([model], 2333), [])
        self.assertEqual(episodes.started([rule], 4000), [rule])
        episodes.reset()
        self.assertEqual(episodes.started([model], 100), [model])

    def test_overlapping_tracks_keep_original_keypoint_mapping(self):
        points = np.ones((17, 3))
        tracks = [{"track_id": i, "bbox": [0, 0, 40, 60]} for i in (1, 2)]
        result = match_keypoints(tracks, [{"bbox": [0, 0, 40, 60], "keypoints": points}])
        self.assertIs(result[1], points)
        self.assertIs(result[2], points)

    def run_video(self, probability, sample_fps=None, threshold=.50, event_threshold=.60):
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
        if np.ndim(probability):
            probabilities = np.asarray(probability, dtype=np.float32)
        else:
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
        behavior_model = nn.Identity()
        if sample_fps is not None:
            behavior_model.behavior_input_config = {'sample_fps': sample_fps}
        with patch('backend.worker.verified_models', return_value=({'pose': '', 'object': '', 'behavior': ''}, {})), \
             patch('backend.worker.source_frames', source), \
             patch('backend.worker.HumanDetector', return_value=pose), \
             patch('backend.worker.ObjectDetector', return_value=detector), \
             patch('backend.worker.HumanTracker', return_value=tracker), \
             patch('backend.worker.TheftMonitor', return_value=rule), \
             patch.object(TrackBehaviorBuffer, '_build_and_load', return_value=behavior_model), \
             patch.object(TrackBehaviorBuffer, '_infer', return_value=probabilities):
            analyze(api, 'test-camera', threshold, {'id': 'test-job', 'claim_token': 'test-claim',
                                                   'settings': {'event_threshold': event_threshold}})
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

    def test_worker_respects_checkpoint_sample_fps(self):
        calls = self.run_video(.8, sample_fps=3)
        final = [body['snapshot'] for path, body in calls if path.endswith('/snapshot')][-1]
        self.assertEqual(final['inference_count'], 0)
        self.assertEqual(final['pose_count'], 40)
        self.assertEqual(final['behavior_sample_fps'], 3)
        self.assertEqual(final['people'][0]['buffer']['collected'], 5)
        self.assertFalse(any(path.endswith('/events') for path, _ in calls))

    def test_worker_does_not_publish_weak_normal_as_confirmed_normal(self):
        calls = self.run_video([.4, 0, 0, 0, .3, 0, .3, 0])
        final = [body['snapshot'] for path, body in calls if path.endswith('/snapshot')][-1]
        person = final['people'][0]
        self.assertIsNone(person['label'])
        self.assertIsNone(person['score'])
        self.assertEqual(person['buffer']['state'], 'uncertain')
        self.assertEqual(person['buffer']['candidate']['label'], 0)
        self.assertFalse(any(path.endswith('/events') for path, _ in calls))

    def test_defaults_require_ninety_percent_for_model_events(self):
        defaults = AnalysisInput()
        self.assertEqual((defaults.threshold, defaults.event_threshold), (.9, .9))
        calls = self.run_video(.8, threshold=defaults.threshold, event_threshold=defaults.event_threshold)
        final = [body['snapshot'] for path, body in calls if path.endswith('/snapshot')][-1]
        self.assertIsNone(final['people'][0]['label'])
        self.assertAlmostEqual(final['people'][0]['buffer']['theft_score'], .8, places=6)
        self.assertFalse(any(path.endswith('/events') for path, _ in calls))


if __name__ == '__main__':
    unittest.main()
