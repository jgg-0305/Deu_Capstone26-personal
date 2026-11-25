-- 운영 스키마 설명을 PostgreSQL 메타데이터에 저장한다.
-- SQL과 pgAdmin 객체 속성에서 테이블, 관계, 인덱스의 용도를 확인할 수 있다.
-- 이전 마이그레이션의 해시 이력은 유지하며 테이블 구조와 데이터는 변경하지 않는다.
-- PK와 UNIQUE 제약은 PostgreSQL이 인덱스를 자동으로 만든다. 외래 키는 참조하는 열 쪽에 인덱스를 자동으로 만들지 않는다.

-- 계정과 인증
COMMENT ON TABLE users IS
    '관제 계정과 권한을 관리한다. 세션, 카카오 계정, 소스, 모델 첨부, 사건 검토와 알림 수신의 기준 테이블이다. id가 기본 키이며 username은 중복을 허용하지 않는다.';
COMMENT ON COLUMN users.password_hash IS
    '로그인 검증용 Argon2 해시다. 기존 계정에 로그인 정보를 부여하기 전에는 비어 있을 수 있다.';
COMMENT ON TABLE auth_sessions IS
    '관제 로그인 세션이다. user_id는 users.id를 참조하며 계정이 물리 삭제되면 함께 삭제된다. token_hash의 고유 제약으로 세션을 조회한다.';
COMMENT ON INDEX auth_sessions_user_idx IS
    '사용자별 세션을 만료 시각과 함께 조회할 때 사용하는 복합 인덱스다.';
COMMENT ON INDEX auth_sessions_expiry_idx IS
    '철회 시각이 없는 세션만 대상으로 만료 시각 조회를 돕는 부분 인덱스다.';
COMMENT ON TABLE oauth_accounts IS
    'users에 연결된 카카오 계정이다. 제공자와 외부 계정 식별자 조합은 고유하다. 알림에서 수신자와 제공자까지 일치하도록 id, user_id, provider 조합을 참조할 수 있다.';
COMMENT ON COLUMN oauth_accounts.token_secret_ref IS
    '카카오 토큰을 보관하는 서버 측 비밀 저장 위치의 식별자다.';
COMMENT ON INDEX oauth_accounts_user_idx IS
    '관제 사용자별로 연결된 카카오 계정을 찾는다.';
COMMENT ON TABLE kakao_oauth_states IS
    '카카오 로그인 콜백에서 한 번 확인할 state의 해시와 만료 시각을 저장한다. session_id는 auth_sessions를 참조하며 세션을 물리 삭제하면 함께 삭제된다.';
COMMENT ON INDEX kakao_oauth_states_expiry_idx IS
    '만료 시각을 기준으로 사용 기한이 지난 state를 찾는다.';
COMMENT ON INDEX kakao_oauth_states_session_idx IS
    '한 관제 세션에 카카오 로그인 state가 하나만 존재하도록 한다. 새 요청은 기존 state를 교체한다.';

-- 영상 소스, 모델 및 분석
COMMENT ON TABLE cameras IS
    'RTSP, 파일, 샘플, 웹캠 소스를 관리한다. owner_id는 users를 참조한다. 웹캠은 등록한 auth_sessions 행에 연결하며 해당 세션을 물리 삭제하면 연결 열을 비운다.';
COMMENT ON COLUMN cameras.source_storage_key IS
    '파일과 샘플 영상의 저장소 참조 키다. 실제 영상 바이트는 로컬 파일에 있다.';
COMMENT ON INDEX cameras_owner_name_idx IS
    '삭제 표시가 없는 소스에 한해 소유자와 소스 이름 조합을 고유하게 유지한다.';
COMMENT ON TABLE model_artifacts IS
    '모델 역할, 파일 해시와 크기, 저장 키, 라벨과 전처리 설정을 등록한다. SHA-256과 저장 키는 각각 고유하다. id와 role의 고유 조합은 실행별 모델 역할을 검증할 때 참조한다.';
COMMENT ON TABLE behavior_uploads IS
    '사용자와 첨부 행동 모델을 연결하고 원래 파일명을 보존한다. owner_id는 users, artifact_id는 model_artifacts를 참조한다. 두 열의 복합 기본 키가 같은 사용자의 중복 연결을 막는다.';
COMMENT ON TABLE analysis_jobs IS
    '카메라별 분석 요청과 설정, 작업 선점 토큰 및 만료 시각을 보관한다. camera_id는 cameras를 참조하고 시작된 요청의 run_id는 inference_runs를 고유하게 참조한다.';
COMMENT ON COLUMN analysis_jobs.settings IS
    '등록 또는 재분석 시 선택한 모델, 해상도, 검출 간격 등 분석 옵션을 저장한다.';
COMMENT ON INDEX analysis_jobs_one_pending_camera IS
    '대기 중이거나 워커가 선점한 요청을 카메라당 하나로 제한하는 부분 고유 인덱스다.';
COMMENT ON INDEX analysis_jobs_pending IS
    '대기 요청만 요청 시각과 ID 순으로 읽어 먼저 등록된 작업을 가져오는 데 사용한다.';
COMMENT ON TABLE analysis_worker_presence IS
    '분석 워커가 마지막으로 연결 신호를 보낸 시각을 저장한다. 참으로 제한한 singleton 기본 키로 전체 상태를 한 행에 유지한다. 개별 분석 실행의 임대 상태는 inference_runs에 있다.';
COMMENT ON TABLE inference_runs IS
    '분석 실행의 소스, 상태, 임대, 시작·종료 시각과 실행 설정을 보관한다. camera_id는 cameras를 참조한다. 사건과 미디어는 실행 ID로 연결된다.';
COMMENT ON INDEX inference_runs_one_active_camera_idx IS
    '시작, 실행, 종료 중 상태에 한해 카메라당 활성 실행 하나를 강제한다.';
COMMENT ON INDEX inference_runs_camera_time_idx IS
    '카메라별 최근 분석 실행을 시작 시각 내림차순으로 조회한다.';
COMMENT ON INDEX inference_runs_expired_lease_idx IS
    '활성 실행의 임대 만료 시각을 탐색한다.';
COMMENT ON TABLE run_models IS
    '분석 실행과 실제 사용 모델을 역할별로 연결한다. run_id와 role이 복합 기본 키이며 model_artifact_id와 role이 함께 model_artifacts를 참조한다.';

-- 사건, 재생 및 검토
COMMENT ON TABLE detection_events IS
    '모델 점수 또는 규칙 근거로 생성된 사건이다. run_id는 inference_runs를 참조하고 점수의 모델 역할은 같은 실행의 run_models를 참조한다. 실행 ID와 워커 사건 키의 고유 제약으로 재전송 중복을 막는다.';
COMMENT ON COLUMN detection_events.source_time_ms IS
    '원본 영상 또는 분석 입력 시간축에서 사건이 발생한 재생 위치를 밀리초로 기록한다.';
COMMENT ON COLUMN detection_events.deleted_at IS
    '사건이 화면에서 숨겨진 시각이다. 중복 재수신을 차단하기 위해 원래 행과 사건 키를 유지한다.';
COMMENT ON INDEX detection_events_run_time_idx IS
    '실행별 사건의 최근 감지 시각 조회를 지원한다.';
COMMENT ON INDEX detection_events_type_time_idx IS
    '사건 유형으로 필터링하고 감지 시각 순으로 조회할 때 사용한다.';
COMMENT ON INDEX detection_events_visible_idx IS
    '삭제 표시가 없는 사건만 대상으로 실행별 시간순 조회를 지원한다.';
COMMENT ON TABLE media_assets IS
    '원본 영상, 클립, 스냅샷, 썸네일의 저장 키와 메타데이터다. run_id는 inference_runs를 참조하고 저장 키는 고유하다. id와 run_id 조합은 사건 미디어의 같은 실행 제약에 사용한다.';
COMMENT ON INDEX media_assets_run_idx IS
    '실행별 미디어의 최근 생성 순 조회를 지원한다.';
COMMENT ON INDEX media_assets_retention_idx IS
    '준비된 미디어 중 보관 기한이 지난 대상을 찾는다.';
COMMENT ON INDEX media_assets_one_source_per_run IS
    '원본 유형 미디어를 실행당 하나로 제한하는 부분 고유 인덱스다.';
COMMENT ON TABLE event_media IS
    '사건과 미디어를 연결하는 다대다 테이블이다. event_id와 media_asset_id가 복합 기본 키이며, 양쪽 복합 외래 키에 run_id를 포함해 같은 분석 실행의 항목만 연결한다.';
COMMENT ON INDEX event_media_asset_idx IS
    '미디어에서 연결된 사건을 역방향으로 조회한다. 복합 기본 키는 사건에서 미디어를 조회하는 데 사용된다.';
COMMENT ON TABLE event_reviews IS
    'detection_events의 사건에 users의 검토자가 내린 판정을 추가로 기록한다. 트리거는 기존 행 수정과 삭제를 막아 판정 변경 이력을 보존한다.';
COMMENT ON INDEX event_reviews_latest_idx IS
    '사건별 최신 검토를 생성 시각과 ID의 내림차순으로 조회한다.';
COMMENT ON TABLE storage_deletions IS
    'cameras에 연결된 파일 삭제 요청과 재시도 횟수, 회수한 바이트를 보관한다. camera_id와 filename의 고유 제약으로 같은 파일 삭제 요청이 중복되지 않는다.';

-- 알림 및 사진
COMMENT ON TABLE notification_deliveries IS
    '사건별 카카오 알림을 발송하기 위한 대기열이다. 사건과 수신자를 참조하며 연결된 OAuth 계정의 소유자 및 채널이 일치해야 한다. 사건, 수신자, 채널 조합으로 중복 발송 예약을 막는다.';
COMMENT ON INDEX notification_deliveries_due_idx IS
    '발송 대기 및 재시도 행을 대상으로 예정 시각과 생성 시각 순의 조회를 지원한다.';
COMMENT ON INDEX notification_deliveries_expired_lease_idx IS
    '발송 중 상태에서 처리 권한이 만료된 알림을 찾는다.';
COMMENT ON INDEX notification_deliveries_recipient_idx IS
    '수신자별 발송 이력의 최근 생성 순 조회를 지원한다.';
COMMENT ON TABLE kakao_event_images IS
    '카카오 알림에 첨부하는 사건 프레임 JPEG를 보관한다. event_id가 기본 키이자 detection_events 외래 키여서 사건당 한 장을 저장한다. 사건을 삭제 표시하면 트리거가 사진을 제거한다.';
COMMENT ON INDEX kakao_event_images_expiry_idx IS
    '보관 기한이 지난 사건 사진을 만료 시각 순으로 찾는다.';
