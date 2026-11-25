/*
 * 새 소스의 기본 설정과 기존 실행에서 복원할 설정을 결정한다. 은퇴한 모델은 선택을 복원할 때 걸러내고, 프리셋을 적용할 때 선택한 행동
 * 가중치를 보존한다.
 */
export const initialSettings = {
  behavior_model: "behavior", pose_model: "human_pose", resolution: 480, detection_interval: 1,
  detection_confidence: 0.4, threshold: 0.9, event_threshold: 0.9, max_age: 30,
};

export function configuredSettings(models) {
  const pose = models.find((model) => model.role === "pose" && model.is_default);
  const behavior = models.find((model) => model.role === "behavior" && model.is_default);
  if (!pose && !behavior) return null;
  return {
    ...initialSettings,
    pose_model: pose?.name ?? initialSettings.pose_model,
    behavior_model: behavior?.name ?? initialSettings.behavior_model,
    detection_confidence: pose?.name === "human_pose_m" ? 0.6 : initialSettings.detection_confidence,
  };
}

export function settingsForSource(source, models = []) {
  const config = source.config_snapshot || {};
  const settings = { ...initialSettings };
  for (const key of Object.keys(settings)) {
    if (config[key] != null) settings[key] = config[key];
  }
  settings.threshold = config.behavior_threshold ?? settings.threshold;
  settings.behavior_model = source.behavior_model || settings.behavior_model;
  settings.pose_model = source.pose_model || settings.pose_model;
  const configured = configuredSettings(models);
  if (configured && !models.some((model) => model.name === settings.behavior_model)) {
    settings.behavior_model = configured.behavior_model;
  }
  if (configured && !models.some((model) => model.name === settings.pose_model)) {
    settings.pose_model = configured.pose_model;
    settings.detection_confidence = configured.detection_confidence;
  }
  return settings;
}

export function settingsPreset(current, hasPoseS, demo = false) {
  return {
    ...initialSettings,
    behavior_model: current.behavior_model,
    pose_model: current.pose_model === "human_pose_m" ? current.pose_model : hasPoseS ? "human_pose_s" : "human_pose",
    detection_confidence: demo ? 0.9 : current.pose_model === "human_pose_m" ? 0.6 : initialSettings.detection_confidence,
  };
}

export function mediumPoseSettings(current) {
  return { ...current, pose_model: "human_pose_m", detection_confidence: 0.6 };
}
