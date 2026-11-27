-- 웹에서 등록한 카메라만 별도로 표시하고 서버가 암호화한 인증 정보를 저장한다.
CREATE TABLE rtsp_camera_registrations (
    camera_id UUID PRIMARY KEY REFERENCES cameras(id) ON DELETE CASCADE,
    auth_token TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE rtsp_camera_registrations IS
    '웹 화면에서 등록된 RTSP 카메라와 암호화한 접속 정보를 보관한다. camera_id는 cameras의 외래 키이자 기본 키여서 카메라당 한 행만 허용한다.';
COMMENT ON COLUMN rtsp_camera_registrations.auth_token IS
    '서버 키로 암호화한 카메라 계정이다. 인증이 필요 없는 카메라는 비워 둔다.';
