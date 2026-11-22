-- 카카오 로그인 요청을 기존 관제 세션에 연결하고 콜백에 한 번만 사용한다.
CREATE TABLE kakao_oauth_states (
    state_hash text PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES auth_sessions(id) ON DELETE CASCADE,
    expires_at timestamptz NOT NULL
);
CREATE INDEX kakao_oauth_states_expiry_idx ON kakao_oauth_states(expires_at);
CREATE UNIQUE INDEX kakao_oauth_states_session_idx ON kakao_oauth_states(session_id);
