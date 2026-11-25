/*
 * 영상별 모델과 실행 설정 복원이 올바른지 확인한다. 초기값, 은퇴 모델 제외와 프리셋 적용 후 행동 모델 및 사건 기준을 검사한다.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { configuredSettings, initialSettings, mediumPoseSettings, settingsForSource, settingsPreset } from "./analysis-settings.js";

test("새 영상의 기본값과 이전 영상의 재분석에 은퇴한 모델을 사용하지 않는다", () => {
  const models = [
    { role: "pose", name: "human_pose_m", is_default: true },
    { role: "behavior", name: "upload_current", is_default: true },
  ];
  const defaults = configuredSettings(models);
  assert.equal(defaults.pose_model, "human_pose_m");
  assert.equal(defaults.behavior_model, "upload_current");
  assert.equal(defaults.threshold, 0.9);
  assert.equal(defaults.event_threshold, 0.9);
  const history = { pose_model: "human_pose", behavior_model: "upload_old" };
  assert.equal(settingsForSource(history, models).pose_model, "human_pose_m");
  assert.equal(settingsForSource(history, models).behavior_model, "upload_current");
});

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
    assert.equal(preset.threshold, 0.9);
    assert.equal(preset.event_threshold, 0.9);
    assert.equal(preset.detection_confidence, demo ? 0.9 : 0.4);
  }
});

test("정밀 관절 프리셋은 행동 모델과 사건 기준을 보존한다", () => {
  const settings = mediumPoseSettings({ ...initialSettings, behavior_model: "upload_current" });
  assert.equal(settings.pose_model, "human_pose_m");
  assert.equal(settings.detection_confidence, 0.6);
  assert.equal(settings.behavior_model, "upload_current");
  assert.equal(settings.event_threshold, 0.9);
});
