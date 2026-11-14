CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID NOT NULL REFERENCES cameras(id),
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'claimed', 'completed', 'failed', 'stopped')),
    threshold DOUBLE PRECISION NOT NULL CHECK (threshold BETWEEN 0.5 AND 1.0),
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claim_token UUID,
    claim_expires_at TIMESTAMPTZ,
    run_id UUID UNIQUE REFERENCES inference_runs(id),
    CHECK ((status = 'claimed') = (claim_token IS NOT NULL AND claim_expires_at IS NOT NULL))
);
CREATE UNIQUE INDEX analysis_jobs_one_pending_camera ON analysis_jobs (camera_id)
    WHERE status IN ('queued', 'claimed');
CREATE INDEX analysis_jobs_pending ON analysis_jobs (requested_at, id) WHERE status = 'queued';
