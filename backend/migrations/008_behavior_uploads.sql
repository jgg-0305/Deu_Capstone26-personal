-- 같은 가중치를 첨부한 계정별로 파일 이름과 사용 권한을 기록한다.
CREATE TABLE behavior_uploads (
    owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    artifact_id UUID NOT NULL REFERENCES model_artifacts(id),
    filename TEXT NOT NULL,
    PRIMARY KEY (owner_id, artifact_id)
);
