export const initialSettings = {
  behavior_model: "behavior", pose_model: "human_pose", resolution: 480, detection_interval: 1,
  detection_confidence: 0.4, threshold: 0.5, event_threshold: 0.6, max_age: 30,
};

export function settingsForSource(source) {
  const config = source.config_snapshot || {};
  const settings = { ...initialSettings };
  for (const key of Object.keys(settings)) {
    if (config[key] != null) settings[key] = config[key];
  }
  settings.threshold = config.behavior_threshold ?? settings.threshold;
  settings.behavior_model = source.behavior_model || settings.behavior_model;
  settings.pose_model = source.pose_model || settings.pose_model;
  return settings;
}

export function settingsPreset(current, hasPoseS, demo = false) {
  return {
    ...initialSettings,
    behavior_model: current.behavior_model,
    pose_model: hasPoseS ? "human_pose_s" : "human_pose",
    detection_confidence: demo ? 0.9 : initialSettings.detection_confidence,
  };
}
