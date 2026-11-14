-- 카메라별 분석 설정과 브라우저 웹캠의 세션을 저장한다.
ALTER TABLE analysis_jobs ADD COLUMN settings JSONB NOT NULL DEFAULT '{}';
ALTER TABLE cameras ADD COLUMN webcam_session_id UUID REFERENCES auth_sessions(id) ON DELETE SET NULL;
ALTER TABLE cameras DROP CONSTRAINT cameras_source_type_check;
ALTER TABLE cameras DROP CONSTRAINT cameras_check;
ALTER TABLE cameras ADD CHECK (source_type IN ('rtsp', 'file', 'sample', 'webcam'));
ALTER TABLE cameras ADD CHECK (
    (source_type = 'rtsp' AND source_uri ~ '^rtsps?://' AND source_uri IS NOT NULL
        AND source_storage_key IS NULL)
    OR (source_type IN ('file', 'sample') AND source_uri IS NULL AND source_storage_key IS NOT NULL
        AND length(source_storage_key) > 0 AND credentials_ref IS NULL)
    OR (source_type = 'webcam' AND source_uri IS NULL AND source_storage_key IS NULL
        AND credentials_ref IS NULL)
);
ALTER TABLE cameras ADD CHECK (source_type = 'webcam' OR webcam_session_id IS NULL);

-- 이전에 등록하고 한 번도 분석하지 않은 파일을 대기열에 넣는다.
INSERT INTO analysis_jobs (camera_id, threshold)
SELECT id, 0.90 FROM cameras c
WHERE source_type IN ('file', 'sample') AND is_enabled AND deleted_at IS NULL
  AND NOT EXISTS (SELECT 1 FROM analysis_jobs j WHERE j.camera_id = c.id)
  AND NOT EXISTS (SELECT 1 FROM inference_runs r WHERE r.camera_id = c.id);
