"""전체 사건을 평가할 때 미확정 또는 관측 불가 구간을 정답으로 세지 않는지 검사한다. 준비한 예측 시퀀스의 집계 결과를 검증한다.
"""
import unittest
import numpy as np
from diagnose import summarize


class EvaluationCoverageTests(unittest.TestCase):
    def test_missing_and_uncertain_frames_are_not_counted_as_correct_normal(self):
        box = [0, 0, 10, 10]
        tracks = {7: {1: (box, np.ones(34))}}
        events = [((1, box), (4, box))]
        rows = [
            {'frame': 1, 'predictions': {}, 'states': {'7': 'collecting'}},
            {'frame': 2, 'predictions': {'7': [0, .4]}, 'states': {'7': 'uncertain'}},
            {'frame': 3, 'predictions': {'7': [0, .8]}, 'states': {'7': 'classified'}},
            {'frame': 4, 'predictions': {'7': [6, .9]}, 'states': {'7': 'classified'}},
        ]
        report = summarize(rows, tracks, events, 6)
        self.assertEqual(report['event_annotated_frames'], 4)
        self.assertEqual(report['event_visible_predictions'], {'normal': 1, 'theft': 1})
        self.assertEqual(report['event_uncertain_frames'], 1)
        self.assertEqual(report['event_missing_or_collecting_frames'], 1)
        self.assertEqual(sum(report['event_visible_predictions'].values())
                         + report['event_uncertain_frames'] + report['event_missing_or_collecting_frames'], 4)


if __name__ == '__main__':
    unittest.main()
