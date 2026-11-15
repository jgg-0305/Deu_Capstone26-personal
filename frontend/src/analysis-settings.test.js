import assert from "node:assert/strict";
import test from "node:test";
import { initialSettings, settingsForSource, settingsPreset } from "./analysis-settings.js";

test("새로고침 후 등록 영상의 업로드 모델과 실제 실행 기준을 복원한다", () => {
  const restored = settingsForSource({
    behavior_model: "upload_new-checkpoint", pose_model: "human_pose_s",
    config_snapshot: { behavior_threshold: 0.7, event_threshold: 0.8,
      detection_confidence: 0.4, resolution: 640, detection_interval: 2,
      rule_version: "stand-pos-v1", alert_threshold: 0.9, max_age: 45 },
  });
  assert.equal(restored.behavior_model, "upload_new-checkpoint");
  assert.equal(restored.pose_model, "human_pose_s");
  assert.equal(restored.threshold, 0.7);
  assert.equal(restored.event_threshold, 0.8);
  assert.equal(restored.resolution, 640);
  assert.equal(restored.detection_interval, 2);
  assert.equal(restored.max_age, 45);
  assert.deepEqual(Object.keys(restored).sort(), Object.keys(initialSettings).sort());
});

test("원본 및 데모 설정 버튼이 선택한 새 행동 모델을 구형으로 교체하지 않는다", () => {
  const selected = { ...initialSettings, behavior_model: "upload_new-checkpoint", threshold: 0.9 };
  for (const demo of [false, true]) {
    const preset = settingsPreset(selected, true, demo);
    assert.equal(preset.behavior_model, selected.behavior_model);
    assert.equal(preset.threshold, 0.5);
    assert.equal(preset.detection_confidence, demo ? 0.9 : 0.4);
  }
});
