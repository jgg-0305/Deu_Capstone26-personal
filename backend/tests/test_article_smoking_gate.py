"""Regression for stale behavior windows becoming new events after article evidence."""
import unittest

import numpy as np

from core.article_tracking import ArticleSmokingGate


class ArticleSmokingGateTests(unittest.TestCase):
    def test_expiry_and_ema_do_not_resurface_smoking_without_new_evidence(self):
        gate = ArticleSmokingGate()
        points = np.zeros((17, 3))
        points[0] = [50, 10, 1]
        points[9] = [50, 60, 1]
        tracks = [{"track_id": 1, "bbox": [0, 0, 100, 100], "keypoints": points},
                  {"track_id": 2, "bbox": [0, 0, 100, 100], "keypoints": points}]
        predictions = {1: (4, .75), 2: (4, .75)}

        self.assertEqual(gate.update(predictions, {1}, tracks, {1: (10, 4, .9), 2: (11, 4, .9)}, .5), set())
        self.assertEqual(gate.update(predictions, set(), tracks, {1: (10, 4, .9), 2: (11, 4, .9)}, .5), {1})
        self.assertEqual(gate.update(predictions, set(), tracks, {1: (12, 0, .8), 2: (11, 4, .9)}, .5), {1})
        self.assertEqual(gate.update(predictions, set(), tracks, {1: (13, 4, .9), 2: (11, 4, .9)}, .5), {1})

        points[9] = [50, 12, 1]
        self.assertEqual(gate.update(predictions, set(), tracks, {1: (13, 4, .9), 2: (11, 4, .9)}, .5), set())
