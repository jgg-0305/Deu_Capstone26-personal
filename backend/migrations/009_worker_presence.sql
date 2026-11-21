CREATE TABLE analysis_worker_presence (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    last_seen_at timestamptz NOT NULL
);
