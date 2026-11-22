-- 알림에 첨부할 감지 프레임 하나만 제한된 기간 동안 보관한다.
CREATE TABLE kakao_event_images (
    event_id uuid PRIMARY KEY REFERENCES detection_events(id) ON DELETE CASCADE,
    jpeg bytea NOT NULL CHECK (octet_length(jpeg) BETWEEN 4 AND 196608),
    expires_at timestamptz NOT NULL DEFAULT clock_timestamp() + interval '1 day'
);

CREATE INDEX kakao_event_images_expiry_idx ON kakao_event_images (expires_at);

CREATE FUNCTION delete_kakao_event_image() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM kakao_event_images WHERE event_id = NEW.id;
    RETURN NEW;
END;
$$;

CREATE TRIGGER delete_kakao_event_image_on_event_removal
    AFTER UPDATE OF deleted_at ON detection_events
    FOR EACH ROW WHEN (OLD.deleted_at IS NULL AND NEW.deleted_at IS NOT NULL)
    EXECUTE FUNCTION delete_kakao_event_image();
