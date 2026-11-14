-- 기존 사용자의 식별자와 소유권을 유지하면서 로그인 정보를 추가한다.
ALTER TABLE users
    ADD COLUMN username TEXT UNIQUE,
    ADD COLUMN password_hash TEXT,
    ADD COLUMN is_test_admin BOOLEAN NOT NULL DEFAULT false,
    ADD CONSTRAINT users_credentials_pair CHECK ((username IS NULL) = (password_hash IS NULL)),
    ADD CONSTRAINT users_username_format CHECK (username ~ '^[a-z0-9_]{3,32}$'),
    ADD CONSTRAINT users_test_admin_identity CHECK (NOT is_test_admin OR (username IS NOT NULL AND username = 'admin' AND role = 'owner'));
