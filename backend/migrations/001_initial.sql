-- 최초 스키마를 생성한다. 트랜잭션은 마이그레이션 도구가 관리한다.

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    display_name TEXT NOT NULL CHECK (length(btrim(display_name)) > 0),
    role TEXT NOT NULL DEFAULT 'owner' CHECK (role IN ('owner', 'demo_viewer')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    disabled_at TIMESTAMPTZ
);

CREATE TABLE auth_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE CHECK (token_hash ~ '^[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ,
    CHECK (expires_at > created_at),
    CHECK (revoked_at IS NULL OR revoked_at >= created_at)
);
CREATE INDEX auth_sessions_user_idx ON auth_sessions(user_id, expires_at);
CREATE INDEX auth_sessions_expiry_idx ON auth_sessions(expires_at)
    WHERE revoked_at IS NULL;

CREATE TABLE oauth_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    provider TEXT NOT NULL CHECK (provider = 'kakao'),
    provider_subject TEXT NOT NULL CHECK (length(provider_subject) > 0),
    token_secret_ref TEXT NOT NULL CHECK (length(token_secret_ref) > 0),
    access_expires_at TIMESTAMPTZ,
    refresh_expires_at TIMESTAMPTZ,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, provider_subject),
    -- 알림 전송에서 수신자와 연결된 제공자를 함께 참조한다.
    UNIQUE (id, user_id, provider)
);
CREATE INDEX oauth_accounts_user_idx ON oauth_accounts(user_id);

CREATE TABLE cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    name TEXT NOT NULL CHECK (length(btrim(name)) > 0),
    source_type TEXT NOT NULL CHECK (source_type IN ('rtsp', 'file', 'sample')),
    source_uri TEXT,
    source_storage_key TEXT,
    credentials_ref TEXT,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(config) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    CHECK (
        (source_type = 'rtsp' AND source_uri IS NOT NULL
            AND source_uri ~ '^rtsps?://' AND source_storage_key IS NULL)
        OR
        (source_type IN ('file', 'sample') AND source_uri IS NULL
            AND source_storage_key IS NOT NULL AND length(source_storage_key) > 0
            AND credentials_ref IS NULL)
    )
);
-- RTSP 주소는 서비스에서 허용 범위를 확인하고 비밀정보를 제거한다. 업로드 영상은 저장소 키로 참조한다.
CREATE UNIQUE INDEX cameras_owner_name_idx ON cameras(owner_id, name)
    WHERE deleted_at IS NULL;

CREATE TABLE model_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role TEXT NOT NULL CHECK (role IN ('pose', 'object', 'behavior', 'reid')),
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    storage_key TEXT NOT NULL UNIQUE,
    size_bytes BIGINT NOT NULL CHECK (size_bytes > 0),
    labels JSONB NOT NULL CHECK (jsonb_typeof(labels) IN ('object', 'array')),
    preprocessing JSONB NOT NULL CHECK (jsonb_typeof(preprocessing) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (id, role)
);

CREATE TABLE inference_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE RESTRICT,
    worker_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('starting', 'running', 'stopping', 'stopped', 'completed', 'failed')
    ),
    lease_token UUID NOT NULL DEFAULT gen_random_uuid(),
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_expires_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    config_snapshot JSONB NOT NULL CHECK (jsonb_typeof(config_snapshot) = 'object'),
    error_code TEXT,
    CHECK (lease_expires_at > heartbeat_at),
    CHECK (ended_at IS NULL OR ended_at >= started_at),
    CHECK (
        (status IN ('starting', 'running', 'stopping') AND ended_at IS NULL)
        OR (status IN ('stopped', 'completed', 'failed') AND ended_at IS NOT NULL)
    )
);
CREATE UNIQUE INDEX inference_runs_one_active_camera_idx ON inference_runs(camera_id)
    WHERE status IN ('starting', 'running', 'stopping');
CREATE INDEX inference_runs_camera_time_idx ON inference_runs(camera_id, started_at DESC);
CREATE INDEX inference_runs_expired_lease_idx ON inference_runs(lease_expires_at)
    WHERE status IN ('starting', 'running', 'stopping');
-- 워커는 갱신과 결과 저장에 실행 임대 토큰을 사용한다. 조정기는 만료된 실행을 정리한다.

CREATE TABLE run_models (
    run_id UUID NOT NULL REFERENCES inference_runs(id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK (role IN ('pose', 'object', 'behavior', 'reid')),
    model_artifact_id UUID NOT NULL,
    PRIMARY KEY (run_id, role),
    FOREIGN KEY (model_artifact_id, role)
        REFERENCES model_artifacts(id, role) ON DELETE RESTRICT
);
-- 분석 시작 전에 관절·객체·행동 모델과 필요 시 DeepSORT 외형 모델이 등록됐는지 확인한다.

CREATE TABLE detection_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES inference_runs(id) ON DELETE RESTRICT,
    producer_event_key TEXT NOT NULL CHECK (length(producer_event_key) > 0),
    payload_sha256 TEXT NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
    event_type TEXT NOT NULL CHECK (length(event_type) > 0),
    track_id TEXT,
    detection_method TEXT NOT NULL CHECK (detection_method IN ('model', 'rule', 'hybrid')),
    model_score DOUBLE PRECISION CHECK (model_score >= 0 AND model_score <= 1),
    score_model_role TEXT,
    rule_code TEXT,
    rule_version TEXT,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(evidence) = 'object'),
    -- 감지 시각은 워커에서, 수신 시각은 저장 API에서 기록한다.
    detected_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 실제 촬영 시각을 확인할 수 있는 경우에만 기록한다.
    occurred_at TIMESTAMPTZ,
    source_time_ms BIGINT CHECK (source_time_ms >= 0),
    source_frame_index BIGINT CHECK (source_frame_index >= 0),
    source_end_time_ms BIGINT,
    UNIQUE (run_id, producer_event_key),
    UNIQUE (id, run_id),
    FOREIGN KEY (run_id, score_model_role) REFERENCES run_models(run_id, role),
    CHECK ((model_score IS NULL) = (score_model_role IS NULL)),
    CHECK (detection_method <> 'rule' OR model_score IS NULL),
    CHECK (detection_method NOT IN ('rule', 'hybrid')
        OR (rule_code IS NOT NULL AND rule_version IS NOT NULL)),
    CHECK (source_end_time_ms IS NULL
        OR (source_time_ms IS NOT NULL AND source_end_time_ms >= source_time_ms))
);
-- 규칙 판정의 근거는 별도로 저장한다. 장비 시계가 다를 수 있어 감지와 수신 시각의 순서를 강제하지 않는다.
CREATE INDEX detection_events_run_time_idx ON detection_events(run_id, detected_at DESC);
CREATE INDEX detection_events_type_time_idx ON detection_events(event_type, detected_at DESC);

CREATE TABLE media_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES inference_runs(id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('clip', 'snapshot', 'thumbnail')),
    storage_key TEXT NOT NULL UNIQUE CHECK (length(storage_key) > 0),
    mime_type TEXT NOT NULL,
    size_bytes BIGINT CHECK (size_bytes >= 0),
    sha256 TEXT CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    width INTEGER CHECK (width > 0),
    height INTEGER CHECK (height > 0),
    duration_ms BIGINT CHECK (duration_ms >= 0),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'ready', 'failed', 'missing', 'deleting', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ready_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    UNIQUE (id, run_id),
    CHECK (status <> 'ready' OR (ready_at IS NOT NULL AND sha256 IS NOT NULL
        AND size_bytes IS NOT NULL AND size_bytes > 0)),
    CHECK (status <> 'deleted' OR deleted_at IS NOT NULL),
    CHECK (expires_at IS NULL OR expires_at >= created_at)
);
CREATE INDEX media_assets_run_idx ON media_assets(run_id, created_at DESC);
CREATE INDEX media_assets_retention_idx ON media_assets(expires_at)
    WHERE status = 'ready';
-- 검증이 끝난 비공개 저장소 키를 기록한다. 파일 쓰기와 데이터베이스 기록 사이에 생긴 불일치는 별도로 복구한다.

CREATE TABLE event_media (
    event_id UUID NOT NULL,
    media_asset_id UUID NOT NULL,
    run_id UUID NOT NULL,
    PRIMARY KEY (event_id, media_asset_id),
    FOREIGN KEY (event_id, run_id) REFERENCES detection_events(id, run_id) ON DELETE RESTRICT,
    FOREIGN KEY (media_asset_id, run_id) REFERENCES media_assets(id, run_id) ON DELETE RESTRICT
);
-- 영상 하나를 같은 분석 실행의 여러 사건에 연결할 수 있다.
CREATE INDEX event_media_asset_idx ON event_media(media_asset_id);

CREATE TABLE event_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES detection_events(id) ON DELETE RESTRICT,
    reviewer_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    decision TEXT NOT NULL CHECK (decision IN ('confirmed', 'false_positive', 'needs_review')),
    note TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX event_reviews_latest_idx ON event_reviews(event_id, created_at DESC, id DESC);
-- 현재 판정은 최신 검토를 따른다. 수정은 새 기록으로 남기며 소유권과 검토 시각은 서비스에서 확인한다.
CREATE FUNCTION reject_event_review_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'event_reviews is append-only; insert a correcting review';
END;
$$;
CREATE TRIGGER event_reviews_append_only
    BEFORE UPDATE OR DELETE ON event_reviews
    FOR EACH ROW EXECUTE FUNCTION reject_event_review_mutation();

CREATE TABLE notification_deliveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES detection_events(id) ON DELETE RESTRICT,
    recipient_user_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    oauth_account_id UUID NOT NULL,
    channel TEXT NOT NULL CHECK (channel = 'kakao'),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'delivering', 'retry_wait', 'sent', 'failed', 'cancelled')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 5 CHECK (max_attempts > 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_token UUID,
    lease_expires_at TIMESTAMPTZ,
    last_error_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sent_at TIMESTAMPTZ,
    FOREIGN KEY (oauth_account_id, recipient_user_id, channel)
        REFERENCES oauth_accounts(id, user_id, provider) ON DELETE RESTRICT,
    UNIQUE (event_id, recipient_user_id, channel),
    CHECK (attempt_count <= max_attempts),
    CHECK (status <> 'sent' OR sent_at IS NOT NULL),
    CHECK (
        (status = 'delivering' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (status <> 'delivering' AND lease_token IS NULL AND lease_expires_at IS NULL)
    )
);
CREATE INDEX notification_deliveries_due_idx
    ON notification_deliveries(next_attempt_at, created_at)
    WHERE status IN ('queued', 'retry_wait');
CREATE INDEX notification_deliveries_expired_lease_idx
    ON notification_deliveries(lease_expires_at) WHERE status = 'delivering';
CREATE INDEX notification_deliveries_recipient_idx
    ON notification_deliveries(recipient_user_id, created_at DESC);
-- 사건과 알림 예약을 한 트랜잭션에서 기록한다. 발송기는 임대를 얻은 뒤 외부 요청을 처리하고 실패 시 재시도한다. 오류 기록에 인증 정보를 포함하지 않는다.
