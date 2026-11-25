"""행동 버퍼가 원본 시각에 맞춰 관절을 모으는지 확인한다. FPS 차이, 긴 시각 간격과 되감기, 기존 프레임 기반 모델, 라벨 순서를
다룬다.
"""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from torch import nn
from core.kpt_behavior_classifier import TrackBehaviorBuffer, KeypointLSTMv2


class BehaviorSamplingTests(unittest.TestCase):
    def buffer(self, fps=3):
        model = nn.Identity()
        if fps is not None:
            model.behavior_input_config = {'sample_fps': fps}
        with patch.object(TrackBehaviorBuffer, '_build_and_load', return_value=model):
            result = TrackBehaviorBuffer('unused')
        result._infer = Mock(return_value=np.array([.1,0,0,0,0,0,.9,0], dtype=np.float32))
        return result

    def feed(self, model, seconds):
        return model.update([{'track_id': 1, 'bbox': [0,0,10,10]}],
                            {1: np.ones((17,3))}, source_seconds=seconds)

    def test_thirty_fps_and_three_fps_have_same_ten_second_window(self):
        for source_fps in [3, 30]:
            model = self.buffer()
            for i in range(source_fps * 10):
                result = self.feed(model, i/source_fps)
                if i/source_fps < 29/3:
                    self.assertFalse(result)
            self.assertEqual(model.inference_count, 1)
            self.assertEqual(result[1][0], 6)
            self.assertEqual(model._cnts[1], 30)

    def test_timestamp_gap_resets_the_old_prediction(self):
        model = self.buffer()
        for i in range(30):
            self.feed(model, i/3)
        self.assertFalse(self.feed(model, 12))
        self.assertEqual(model.progress(1)['collected'], 1)
        self.assertEqual(model.reset_count, 1)

    def test_seek_backwards_and_invisible_person_do_not_reuse_predictions(self):
        model = self.buffer()
        for i in range(30):
            self.feed(model, i/3)
        self.assertFalse(model.update([], {}, source_seconds=9.7))
        self.assertFalse(self.feed(model, 0))
        self.assertEqual(model.progress(1)['collected'], 1)

    def test_plain_legacy_model_retains_frame_based_behavior(self):
        model = self.buffer(None)
        for i in range(30):
            result = self.feed(model, i/30)
        self.assertEqual(result[1][0], 6)

    def test_metadata_checkpoint_rejects_swapped_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pth'
            config = {'labels': ['normal','fall','vandalism','arson','smoking','abandon','theft','assault'],
                      'sample_fps':3.0,'sequence_length':30}
            state = KeypointLSTMv2().state_dict()
            torch.save({'state_dict':state,'input_config':config},path)
            loaded = TrackBehaviorBuffer(str(path))
            self.assertEqual(loaded.sample_fps,3.0)
            config['labels'][4],config['labels'][6] = config['labels'][6],config['labels'][4]
            torch.save({'state_dict':state,'input_config':config},path)
            with self.assertRaises(ValueError):
                TrackBehaviorBuffer(str(path))


if __name__ == '__main__':
    unittest.main()
