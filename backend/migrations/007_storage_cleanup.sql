-- 삭제한 사건의 중복 수신을 막기 위해 식별자는 남기고 조회에서 제외한다.
ALTER TABLE detection_events ADD COLUMN deleted_at TIMESTAMPTZ;
ALTER TABLE cameras ADD COLUMN purge_requested_at TIMESTAMPTZ;
CREATE INDEX detection_events_visible_idx ON detection_events(run_id, detected_at DESC)
    WHERE deleted_at IS NULL;

-- DB에 삭제 의도를 먼저 커밋한 후 파일을 지운다. 파일 사용 중 오류는 재시도한다.
CREATE TABLE storage_deletions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID NOT NULL REFERENCES cameras(id),
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'deleted')),
    freed_bytes BIGINT NOT NULL DEFAULT 0 CHECK (freed_bytes >= 0),
    attempts INTEGER NOT NULL DEFAULT 0,
    error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    UNIQUE(camera_id, filename)
);
