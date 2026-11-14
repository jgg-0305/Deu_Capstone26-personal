-- 사용자가 등록한 영상의 원래 파일 이름을 표시용으로 보관한다.
ALTER TABLE cameras ADD COLUMN original_filename TEXT;
