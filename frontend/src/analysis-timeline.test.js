import test from 'node:test';
import assert from 'node:assert/strict';
import { frameAt } from './analysis-timeline.js';

test('탐색한 시점의 결과만 선택하고 미래·오래된 결과를 숨긴다', () => {
  const frames = [{ time_ms: 100 }, { time_ms: 200 }, { time_ms: 300 }];
  assert.equal(frameAt(frames, 90), null);
  assert.equal(frameAt(frames, 250).time_ms, 200);
  assert.equal(frameAt(frames, 150).time_ms, 100);
  assert.equal(frameAt(frames, 600), null);
});

test('3 FPS 원본 영상에서 다음 영상 프레임까지 관절 표시를 유지한다', () => {
  const frames = [{ time_ms: 0, frame_duration_ms: 333.333 }, { time_ms: 333, frame_duration_ms: 333.333 }];
  assert.equal(frameAt(frames, 300).time_ms, 0);
  assert.equal(frameAt(frames, 650).time_ms, 333);
  assert.equal(frameAt(frames, 900), null);
  assert.equal(frameAt([{ time_ms: 0 }, { time_ms: 333 }], 300).time_ms, 0);
});
