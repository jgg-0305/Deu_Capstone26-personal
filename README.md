# CCTV 이상 현상 감지 시스템

YOLO11 + DeepSORT + KeypointLSTM 기반 실시간 **이상행동 감지** 시스템입니다.
인물을 감지·추적하며 17관절 키포인트 시퀀스로 **인물(트랙 ID)별** 이상행동을 분류하고, 매장 **가판대·결제기**를 감지해 **도난 의심 규칙**을 적용합니다.

현재 관제 화면은 **React 웹**으로 실행합니다. FastAPI와 분석 워커가 MP4·웹캠·RTSP를 처리하고 사건과 분석 이력을 PostgreSQL에 기록합니다. 초기 **PyQt5 데스크톱 관제**의 소스와 이상행동 자동 녹화 기능은 저장소에 남아 있습니다.

- [파이프라인](#파이프라인)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치와 실행](#설치와-실행)
- [영상 등록과 사건 확인](#영상-등록과-사건-확인)
- [모델 선택과 비교](#모델-선택과-비교)
- [카카오톡 알림](#카카오톡-알림)
- [데이터베이스와 pgAdmin](#데이터베이스와-pgadmin)
- [문제 해결](#문제-해결)
- [학습과 검증](#학습과-검증)

## 파이프라인

```text
MP4 · 웹캠 · RTSP → 분석 워커 → 영상 프레임
                    ↓
YOLO11 Pose → 인물 상자 + 17관절 → DeepSORT → 인물별 track_id와 손목 좌표
                    ↓
├─ 행동 분류 → KeypointLSTM → 인물별 행동·신뢰도
│  └─ HandFusion 가중치 선택 시 손목 주변 RGB 영상과 움직임 특징을 함께 사용
├─ 가판대·결제기 규칙 → objects.pt로 영역 검출 → 가판대 상호작용·결제기 경유 확인
└─ 상품·가방 보조 추적 → 손목 주변에서 상품 검출 → 같은 손에 연결해 이동 추적
   ├─ 골반 근처에서 상품의 가림 판정
   └─ 가방 검출이 활성화된 경우 가방으로 이동한 상품의 가림 판정
                    ↓
모델 결과와 규칙 근거 결합 → 상자·관절·행동 표시 및 도난 의심 판정
                    ↓
FastAPI
├─ React 관제 화면 → 원본 재생·AI 오버레이·사건 조회
└─ PostgreSQL → 사건 기록·알림 대기열 → 카카오톡 워커(설정 시)
```

손목 좌표는 YOLO11 Pose의 17관절에서 가져옵니다. **손 영상 융합**은 행동 분류에 손 주변 영상과 움직임을 더하는 모델이고, **손 주변 상품·가방 추적**은 상품의 이동과 가림을 확인해 도난 의심 근거를 만드는 별도 경로입니다. 상품 추적에는 인물별 행동 모델과 상품 가중치가 필요합니다. 가방 추적은 여기에 YOLO11m Pose와 가방 가중치가 모두 있을 때 실행됩니다. 활성화 조건과 파일 경로는 [모델 선택과 비교](#모델-선택과-비교)에 있습니다.

웹 워커의 기본 분석에는 포즈·가판대/결제기 검출·행동 가중치가 필요합니다. 기존 PyQt5는 포즈 가중치만으로 인물 감지·추적과 관절 표시를 지원하며, 이상행동 감지 시 전후 영상과 `recordings.json`을 남기는 경로가 코드에 보존되어 있습니다.

## 주요 기능

| 기능 | 설명 |
| --- | --- |
| **인간 감지 + 포즈** | YOLO11 Pose로 인물 상자와 COCO 17관절을 검출합니다. |
| **다중 객체 추적** | DeepSORT로 인물별 트랙 ID를 유지하고 화면에 구분해 표시합니다. |
| **인물별 이상행동 분류** | KeypointLSTM으로 트랙별 8종 행동을 분류합니다. 기존 PyQt5는 이상행동 인물을 빨간 상자와 `ID:1 \| ASSAULT 92%` 같은 라벨로 표시합니다. |
| **가판대·결제기 감지** | 커스텀 YOLO(`objects.pt`)로 가판대와 결제기 영역을 검출합니다. 데스크톱 화면에는 가판대 주황색·결제기 파란색 표시 옵션이 있습니다. |
| **손 영상 융합** | HandFusion 행동 가중치를 선택하면 손 주변 이미지와 손의 움직임을 관절 시퀀스에 결합합니다. |
| **상품·가방 추적** | 손 주변 상품을 검출해 같은 손에 연결하고, 골반이나 가방 근처의 이동과 가림을 추적합니다. |
| **도난 의심 규칙** | 가판대 상호작용 이후 결제기 경유 여부와 활성화된 상품·가방 보조 추적 결과를 각각 판정 근거로 사용합니다. |
| **이벤트 녹화** | 기존 PyQt5는 이상행동 발생 전후 영상과 발생 시각·행동·신뢰도를 `recordings.json`과 함께 저장합니다. |
| **녹화 기록 메뉴** | 기존 PyQt5의 **녹화 기록** 메뉴에서 녹화 목록을 보고 재생하거나 삭제합니다. |
| **다중 소스 + MDI** | 파일·웹캠·RTSP를 연결합니다. 기존 PyQt5는 독립 영상창의 이동·크기 조절과 영상별 재생바를 지원하고, 웹은 소스를 타일로 배치합니다. |
| **IP 카메라** | RTSP 스트림을 처리하며 연결이 끊기면 다시 연결합니다. |
| **경량화** | 감지 모델(n/s/m), 해상도(384/480/640), 검출 간격을 조정합니다. CUDA에서는 지원 모델에 FP16을 사용하며, 기존 데스크톱 코드에는 Raspberry Pi용 카메라 입력 설정도 있습니다. |
| **웹 관제와 사건 관리** | 회원가입·로그인, 원본·AI 화면 전환, 모델 파일 첨부, 사건 검토·재생·삭제와 저장 공간 정리를 지원합니다. |
| **카카오톡 알림** | 연결한 계정의 **나와의 채팅**으로 사건을 전송합니다. 공개 HTTPS 사진 경로를 설정하면 피드 카드에 사건 사진을 표시합니다. |

이상행동 분류 8종: `정상 / 전도(쓰러짐) / 파손 / 방화 / 흡연 / 유기 / 절도(도난 의심) / 폭행`

## 기술 스택

| 구성 | 기술 | 역할 |
| --- | --- | --- |
| 웹 프런트엔드 | React · JavaScript/JSX · CSS · Vite | 영상 관제, 모델 설정, 사건 조회·재생·삭제 화면 |
| 웹 백엔드 | Python · FastAPI · Uvicorn | 관제 로그인과 세션, 영상·사건 API, 분석 워커 연결 |
| 데이터베이스 | **PostgreSQL** · psycopg 3 | 계정·카메라·대기열·분석 실행·사건·발송 기록 및 SQL 마이그레이션 |
| DB 관리 도구 | pgAdmin 4 | PostgreSQL 테이블, 관계와 인덱스 조회 |
| 영상 처리와 추론 | OpenCV · Ultralytics YOLO11 · DeepSORT · PyTorch | 포즈 검출, 인물 추적, 행동 분류, 손 주변 상품·가방 추적 |
| 알림 | 카카오 로그인 · 카카오톡 메시지 API | 연결 계정의 나와의 채팅 및 사건 사진 피드 |
| 기존 데스크톱 코드 | PyQt5 | 초기 관제 화면과 로컬 이벤트 녹화 |

## 프로젝트 구조

```text
Deu_Capstone26/
├── frontend/
│   ├── index.html                 # 웹 진입점
│   ├── package.json               # React 의존성과 빌드·테스트 명령
│   ├── vite.config.js             # 웹 개발 서버와 API 프록시
│   └── src/
│       ├── main.jsx               # 로그인, 관제 타일, 사건 기록
│       ├── analysis.jsx           # 원본·AI 모드, 모델 설정, 분석 오버레이
│       ├── analysis-settings.js   # 모델과 실행 설정 복원
│       ├── analysis-timeline.js   # 재생 위치에 맞는 관절 분석 선택
│       ├── storage.jsx            # 저장 용량과 소스 파일 정리
│       └── style.css              # 웹 화면 스타일
├── backend/
│   ├── api.py                      # 관제 화면, 인증, 소스·모델·사건 API
│   ├── worker.py                   # 대기열에서 분석 작업을 받아 실행
│   ├── db.py                       # PostgreSQL 연결
│   ├── manage.py                   # 마이그레이션, 기본 모델과 계정 관리
│   ├── migrations/                 # PostgreSQL 스키마 변경 및 객체 주석
│   ├── behavior_upload.py          # 행동 가중치 검증
│   ├── pose_upload.py              # 등록된 YOLO 포즈 가중치 첨부
│   ├── live.py                     # RTSP 연결과 웹 미리보기 중계
│   ├── analysis_media.py           # 웹캠 프레임과 분석 결과 저장 경로
│   ├── article_trace.py            # 상품 검출·손 연결·은닉 진단 조회
│   ├── storage.py                  # 저장 공간 집계 및 파일 정리
│   ├── run_local.py                # API와 분석 워커 통합 실행
│   ├── run_local_mac.sh            # 기존 Mac 소켓 DB와 통합 실행기 시작
│   ├── kakao_notify.py             # 카카오톡 알림 대기열 처리
│   ├── kakao_image_gateway.py      # 카카오톡 사건 사진 전용 게이트웨이
│   └── tests/                      # 분석·권한·삭제·알림 테스트
├── core/
│   ├── detector.py                 # YOLO 인물·관절·가판대·결제기 검출
│   ├── tracker.py                  # DeepSORT 추적
│   ├── tracking_geometry.py        # 인물 추적 ID와 관절 연결
│   ├── kpt_behavior_classifier.py # 관절 시퀀스 기반 행동 분류
│   ├── hand_behavior.py            # 손 주변 RGB 영상·움직임 특징 융합
│   ├── rgb_behavior_classifier.py # 장면 단위 RGB 행동 분류
│   ├── theft_monitor.py            # 가판대·결제기 규칙
│   ├── article_tracking.py         # 손 주변 상품 추적
│   ├── bag_transfer.py             # 가방으로의 상품 이동 판정
│   ├── video_thread.py             # 기존 PyQt5 영상 처리 스레드
│   └── recorder.py                 # 기존 PyQt5 이벤트 녹화
├── gui/
│   ├── main_window.py              # 기존 PyQt5 관제 창
│   └── recordings_dialog.py        # 기존 녹화 기록 창
├── main.py                        # 기존 PyQt5 실행 진입점
├── run.bat                        # 기존 Windows 데스크톱 실행
├── model/
│   └── objects.pt                 # 가판대·결제기 가중치
├── yolo11n-pose.pt                 # 기본 포즈 가중치
├── train/
│   ├── 1_behavior/                 # 초기 행동 학습과 output/kpt_behavior.pth
│   ├── 2_keypoint/                 # UniPose 관절 학습
│   ├── 3_stand/                    # 가판대·결제기 학습
│   ├── behavior_repair/            # AI-Hub 기반 재학습과 검증
│   └── behavior_tf/                # TensorFlow 학습과 PyTorch 가중치 변환
└── docs/design/                    # 모델 매니페스트, DB 설계, 검증 자료
```

실행 환경에는 `.local-db/`(기존 Mac 로컬 DB), `.local-models/`(첨부·보조 가중치), `.local-videos/`(업로드 영상·분석 파일), `.local-kakao/`(카카오 계정 토큰)이 별도로 생길 수 있습니다.

## 설치와 실행

Python 3.12, PostgreSQL 16 이상, Node.js 22.12 이상을 준비합니다. 모든 셸 명령은 이 저장소의 루트에서 실행합니다. Apple Silicon Mac에서는 ARM64 Python을 사용합니다.

통합 실행기는 PostgreSQL 마이그레이션과 기본 모델 등록을 처리하고, FastAPI가 준비되면 분석 워커를 시작합니다. 워커는 내부 토큰으로 API에 연결하므로 관제 화면에는 회원가입과 로그인으로 접속하면 됩니다.

### Windows PowerShell

1. PostgreSQL 서비스를 시작합니다. 첫 실행이라면 PostgreSQL 관리자 계정으로 `psql -U postgres -d postgres`에 접속하고 아래 명령을 실행합니다. `\password`가 요청하는 비밀번호를 직접 정합니다. 이미 프로젝트 DB가 있다면 기존 계정과 DB를 사용합니다.

   ```sql
   CREATE ROLE cctv_user LOGIN;
   \password cctv_user
   CREATE DATABASE cctv OWNER cctv_user;
   \q
   ```

2. PowerShell에서 의존성을 설치하고 웹 관제를 시작합니다. 아래 PyTorch 설치는 CPU 실행 기준입니다. CUDA로 실행하는 컴퓨터는 해당 환경에 맞는 CUDA 빌드의 PyTorch와 torchvision을 먼저 설치합니다.

   ```powershell
   py -3.12 -m venv .venv-web
   $Py = ".\.venv-web\Scripts\python.exe"
   & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   & $Py -m pip install -r backend/requirements-worker.txt
   npm.cmd --prefix frontend ci
   npm.cmd --prefix frontend run build

   $env:CCTV_DATABASE_URL = 'postgresql://cctv_user:DB_PASSWORD@127.0.0.1:5432/cctv'
   & $Py -m backend.run_local --port 8000
   ```


   `DB_PASSWORD`를 앞에서 정한 실제 DB 비밀번호로 바꿉니다. 비밀번호에 `@`, `:` 등 URL 예약 문자가 있으면 URL 인코딩합니다. `psql`이 PATH에 없다면 PostgreSQL 설치 폴더의 `bin`에 있는 `psql.exe`를 사용합니다.



### macOS Terminal

먼저 가상환경과 프런트엔드 의존성을 설치합니다.

```bash
python3.12 -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r backend/requirements-worker.txt
npm --prefix frontend ci
```

**기존 Mac 프로젝트 DB를 이어서 사용하는 경우**에는 아래 명령으로 시작합니다. 저장소에 기존 `.local-db/web.env`와 `.local-db/data`가 있어야 합니다. 스크립트는 해당 PostgreSQL 버전과 Unix 소켓의 포트 번호 `55432`를 사용하고, 웹 빌드와 워커 시작까지 처리합니다.

```bash
bash backend/run_local_mac.sh --port 8000
```

**새 Mac에서 PostgreSQL을 구성하는 경우**에는 설치한 PostgreSQL 서비스를 시작합니다. Homebrew PostgreSQL 18을 쓰는 예시는 아래와 같습니다.

```bash
brew services start postgresql@18
"$(brew --prefix postgresql@18)/bin/psql" -d postgres
```

`psql`에서 Windows 안내의 `CREATE ROLE`, `\password`, `CREATE DATABASE`, `\q`를 차례로 실행합니다. 그런 다음 프로젝트 루트의 터미널에서 다음과 같이 실행합니다.

```bash
source .venv-web/bin/activate
export CCTV_DATABASE_URL='postgresql://cctv_user:DB_PASSWORD@127.0.0.1:5432/cctv'
npm --prefix frontend run build
python -m backend.run_local --port 8000
```

### 첫 로그인

1. [로컬 관제 화면](http://127.0.0.1:8000/)을 열고 회원가입한 계정으로 로그인합니다.
2. 파일이나 샘플을 등록하기 전에 [모델 선택과 비교](#모델-선택과-비교)에서 기본 행동 모델의 등록 상태를 확인합니다.
3. **기본 샘플**에서 영상을 등록하고 타일의 분석 상태와 **사건 기록**을 확인합니다.

작업을 마친 뒤 서버를 종료하려면 실행 터미널에서 `Ctrl+C`를 누릅니다. PostgreSQL은 독립적으로 실행됩니다.

로컬에서 테스트 관리자 `admin / admin`이 필요하면 먼저 실제 DB 접속 설정을 불러옵니다. Windows는 위의 `CCTV_DATABASE_URL`을 사용합니다. 기존 Mac 소켓 환경에서는 `source .local-db/web.env`를 실행합니다.

```powershell
# Windows PowerShell
$env:CCTV_ENABLE_TEST_ADMIN = '1'
& $Py -m backend.manage migrate
& $Py -m backend.manage seed-test-admin
& $Py -m backend.run_local --port 8000
```


```bash
# macOS Terminal
export CCTV_ENABLE_TEST_ADMIN=1
python -m backend.manage migrate
python -m backend.manage seed-test-admin
bash backend/run_local_mac.sh --port 8000
```

### 세팅 완료후 평소에 작동시킬때 


```powershell
# Windows PowerShell
& $Py -m backend.run_local --port 8000
```


```bash
# macOS Terminal
bash backend/run_local_mac.sh --port 8000
```


새 Mac의 TCP DB를 사용하는 경우 마지막 줄은 `python -m backend.run_local --port 8000`으로 실행합니다. 기존 Mac 소켓 DB가 꺼져 있다면 먼저 통합 스크립트로 한 번 시작합니다. 테스트 관리자는 로컬 검증용이며 카카오 연결은 일반 회원 계정에서 진행합니다.

### API와 분석 워커를 별도로 실행하는 경우

대부분의 로컬 실행은 위의 통합 실행기를 사용합니다. 분리해서 실행할 때는 PostgreSQL과 프런트엔드 빌드를 먼저 준비하고, 두 터미널의 `CCTV_WORKER_TOKEN`에 동일한 32자 이상의 비밀값을 설정합니다. API를 먼저 시작하고 준비 상태를 확인한 뒤 워커를 시작합니다. 기존 Mac 소켓 DB에서 분리 실행할 때는 API 터미널에서 `.local-db/web.env`를 불러옵니다.

```powershell
# Windows PowerShell 첫 번째 터미널
$Py = ".\.venv-web\Scripts\python.exe"
$Token = Read-Host '워커 토큰' -AsSecureString
$env:CCTV_WORKER_TOKEN = [System.Net.NetworkCredential]::new('', $Token).Password
& $Py -m backend.manage migrate
& $Py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

```powershell
# Windows PowerShell 두 번째 터미널
$Token = Read-Host '같은 워커 토큰' -AsSecureString
$env:CCTV_WORKER_TOKEN = [System.Net.NetworkCredential]::new('', $Token).Password
& .\.venv-web\Scripts\python.exe -m backend.worker --serve --api-url http://127.0.0.1:8000
```

```bash
# macOS Terminal 첫 번째 터미널
printf '워커 토큰: '
read -rs CCTV_WORKER_TOKEN
printf '\n'
export CCTV_WORKER_TOKEN
.venv-web/bin/python -m backend.manage migrate
.venv-web/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

```bash
# macOS Terminal 두 번째 터미널
printf '같은 워커 토큰: '
read -rs CCTV_WORKER_TOKEN
printf '\n'
export CCTV_WORKER_TOKEN
.venv-web/bin/python -m backend.worker --serve --api-url http://127.0.0.1:8000
```

분리 실행에서도 모델은 최초 통합 실행에서 등록했거나 이미 첨부한 상태여야 합니다. 카카오 알림이 필요할 경우 같은 DB 설정에서 `python -m backend.kakao_notify serve`도 실행합니다.

## 영상 등록과 사건 확인

| 소스 | 화면에서 하는 일 |
| --- | --- |
| 기본 샘플 | **기본 샘플**을 선택해 프로젝트에 포함된 영상으로 시작합니다. |
| MP4 | **영상으로 CCTV 테스트**에서 개별 `.mp4` 파일을 선택합니다. 파일당 한도는 256 MiB입니다. |
| 웹캠 | **웹캠 추가**에서 브라우저의 카메라 권한을 허용합니다. 외부 주소에서는 HTTPS가 필요합니다. |
| IP 카메라 | **IP 카메라 (RTSP)**에 서버가 허용한 주소와 표시 이름을 입력합니다. |

소스를 등록하면 자동으로 분석이 대기열에 들어갑니다. **원본**은 영상 재생을, **AI 분석**은 사람 상자·트랙 ID·관절·행동 결과를 보여줍니다. 파일 영상의 오버레이는 재생 위치에 맞춰 표시됩니다. 처리 해상도, 검출 간격, 포즈·행동 신뢰도, DeepSORT 최대 소실 프레임은 **모델 설정**에서 조절합니다. 기존 소스에 새 설정을 쓰려면 **다시 분석**을 누릅니다.

**사건 기록**에서 감지 시점으로 이동하고 판정 방식을 확인하거나 검토 결과를 추가할 수 있습니다. 타일의 **×**는 소스를 관제 화면에서 빼고 분석을 중단합니다. 사건만 지우려면 **이 사건 삭제**, 업로드 원본과 분석 결과까지 정리하려면 **저장 공간 → 영상·기록 삭제**를 사용합니다.

[AI-Hub 매장 이상행동 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71550)로 검증할 때는 `Validation`의 원천 영상을 사용합니다. Windows에서는 압축을 푼 개별 MP4를 선택합니다. macOS에서는 준비된 `combined_video_data`의 개별 MP4를 사용합니다. 원본이 분할 ZIP이면 조각을 숫자 순서대로 합쳐 압축을 푼 뒤 업로드합니다.

### RTSP 허용 주소 설정

서버를 시작할 터미널에서 웹 로그인 계정과 카메라 주소를 허용합니다. 카메라 인증이 필요하면 계정과 비밀번호가 포함된 주소는 서버 환경에만 설정합니다. 관제 화면에는 인증 정보를 뺀 주소를 입력합니다.

```powershell
# Windows PowerShell
$env:CCTV_RTSP_USERS = 'myuser'
$env:CCTV_RTSP_URLS = '["rtsp://192.168.1.50:554/live"]'
```

```bash
# macOS Terminal
export CCTV_RTSP_USERS='myuser'
export CCTV_RTSP_URLS='["rtsp://192.168.1.50:554/live"]'
```

설정 후 통합 실행기를 재시작합니다. 서버가 카메라 네트워크에 연결되어 있어야 합니다. RTSP 연결은 서버에서 처리하므로 관제 탭을 닫아도 분석이 계속됩니다. 브라우저 웹캠은 해당 탭에서 프레임을 전송합니다.

## 모델 선택과 비교

| 역할 | 모델과 파일 |
| --- | --- |
| 사람과 관절 검출 | 기본 `yolo11n-pose.pt`. **YOLO 포즈 모델 파일 첨부**에서는 프로젝트 매니페스트에 등록된 YOLO11n/s/m Pose 가중치를 받습니다. |
| 인물별 행동 분류 | 기본 `train/1_behavior/output/kpt_behavior.pth` (KeypointLSTM v2 + HandFusion). 다른 호환 `.pth`는 **행동 모델 파일 첨부**에서 선택합니다. |
| 장면 전체 행동 분류 | 매니페스트에 등록된 `model_final.pth`. ResNet50과 LSTM으로 장면을 분석하는 선택 모델입니다. |
| 가판대와 결제기 | `model/objects.pt`. 인물의 관절과 함께 가판대·결제기 규칙에 사용합니다. |
| 상품과 가방 | Windows와 macOS 모두 `.local-models/article.pt` 또는 `.local-models/article-item.pt`를 자동으로 찾습니다. `CCTV_ARTICLE_MODEL`을 지정하면 해당 경로가 우선합니다. 가방 가중치는 `.local-models/yolo11s-bag.pt`입니다. |

상품·가방 가중치가 있고 YOLO11m Pose가 등록돼 있으면 새 영상의 기본 포즈 모델로 YOLO11m을 선택합니다. 가방 추적은 상품 추적과 YOLO11m Pose를 함께 쓸 때 켜집니다. 직접 선택한 포즈 모델이나 `CCTV_DEFAULT_POSE_MODEL` 설정은 우선합니다. 장면 단위 행동 모델을 사용하면 상품·가방 추적은 적용되지 않습니다.

통합 실행기는 `train/1_behavior/output/kpt_behavior.pth`의 SHA-256을 검증해 기본 행동 모델로 등록합니다. 새 영상에서는 기본 모델이 자동 선택됩니다. 이전 소스에 이 가중치를 적용하려면 모델 설정에서 **KeypointLSTM v2 + HandFusion**을 선택하고 **다시 분석**을 누릅니다. 영상 타일에서 실제 가중치 파일명과 SHA-256 앞 12자리(`55c3c9e13bd2`)를 확인할 수 있습니다.

행동 분류와 모델 사건 저장의 기본 기준은 각각 90%입니다. 가판대와 상품·가방 규칙으로 기록된 사건에는 모델 점수 대신 규칙 근거가 표시됩니다.

Windows와 macOS의 결과를 비교할 때는 포즈·행동·상품·가방 가중치의 SHA-256과 실행 설정을 함께 맞춥니다. 가중치 해시는 Windows의 `Get-FileHash -Algorithm SHA256 파일경로`, macOS의 `shasum -a 256 파일경로`로 확인할 수 있습니다. 상품이 검출·손 연결·은닉 판정 중 어디에서 탈락했는지 보려면 `inference_runs.id`에서 실제 실행 UUID를 확인하고 아래 명령의 자리표시자를 바꿉니다.

```powershell
& $Py -m backend.article_trace 실행_UUID --from-sec 0 --to-sec 60
```

```bash
.venv-web/bin/python -m backend.article_trace 실행_UUID --from-sec 0 --to-sec 60
```

## 카카오톡 알림

1. 카카오 앱에서 카카오 로그인과 메시지 전송 동의 항목 `talk_message`를 활성화합니다. 로컬 실행은 콜백 `http://127.0.0.1:8000/kakao/callback`을 등록합니다.
2. 실행 환경에 `CCTV_KAKAO_REST_API_KEY`, `CCTV_KAKAO_REDIRECT_URI`, `CCTV_KAKAO_LINK_URL`을 설정합니다. 링크 주소는 앱의 제품 링크 관리에 등록한 관제 주소를 사용합니다. 카카오 앱에 클라이언트 시크릿을 설정했다면 `CCTV_KAKAO_CLIENT_SECRET`도 서버에 설정합니다. Windows는 `$env:변수명 = '값'`, macOS는 `export 변수명='값'` 형식으로 설정합니다.
3. 통합 실행기를 재시작하고 일반 관제 계정으로 로그인합니다. 상단 **카카오 로그인**에서 계정을 연결하고 메시지 전송에 동의합니다.
4. 이후 RTSP·웹캠에서 발생한 대상 사건이 **나와의 채팅**으로 발송됩니다. 파일과 샘플도 확인하려면 서버에 `CCTV_KAKAO_INCLUDE_FILES=1`을 설정하고 다시 시작합니다. 기존 사건은 소급 발송하지 않습니다.

메시지 발송은 기존 API와 알림 워커를 사용합니다. **사건 사진**을 피드 카드에 표시할 때는 카카오 서버가 읽을 수 있는 HTTPS 주소를 추가합니다. 예를 들어 별도 터미널에서 사진 게이트웨이와 터널을 실행합니다.

```powershell
# Windows PowerShell
& .\.venv-web\Scripts\python.exe -m backend.kakao_image_gateway --port 18082 --api-port 8000
```

```bash
# macOS Terminal
.venv-web/bin/python -m backend.kakao_image_gateway --port 18082 --api-port 8000
```

ngrok을 쓰는 경우 다른 터미널에서 `ngrok http http://127.0.0.1:18082 --inspect=false`를 실행합니다. 표시된 HTTPS 주소를 `CCTV_KAKAO_IMAGE_BASE_URL`에 넣고, Python의 `secrets.token_urlsafe(32)`로 생성한 서버 비밀키를 `CCTV_KAKAO_IMAGE_SECRET`에 설정합니다. 환경을 저장한 뒤 관제 서버를 다시 시작합니다. HTTPS로 공개하는 대상은 사건 사진 경로이며, 서명 링크의 유효 기간은 최대 1시간입니다.

## 데이터베이스와 pgAdmin

PostgreSQL은 계정, 영상 소스, 작업 대기열, 분석 실행과 사용 모델, 사건·검토 및 알림 이력을 관리합니다. MP4와 모델 가중치는 로컬 파일에 보관하며, DB에서는 저장 키와 메타데이터를 관리합니다. 사건 알림용 JPEG는 보관 기한을 두고 DB에 저장합니다.

| 연결 항목 | 일반 TCP 연결 | 기존 Mac 소켓 연결 |
| --- | --- | --- |
| 호스트 | `127.0.0.1` | `/private/tmp/deu-cctv-local-db` |
| 포트 | `5432` | `55432` |
| 데이터베이스 | `cctv` | `cctv` |
| 사용자 | PostgreSQL에 만든 계정 | 이 Mac에서 발급한 pgAdmin용 계정 |

pgAdmin의 **객체 탐색기 → Servers → 서버 → 데이터베이스 → cctv → 스키마 → public**에서 테이블을 확인합니다. 속성에 한국어 설명이 보이지 않으면 객체를 새로고침합니다. 테이블 관계와 인덱스 설명은 [스키마 주석 마이그레이션](backend/migrations/012_schema_comments.sql)에 있으며 통합 실행 시 적용됩니다.

외부에서 접속할 때는 HTTPS를 적용하고 내부 워커 API 경로를 공개 프록시에서 제외합니다. DB 접속 정보, 카카오 키와 토큰, RTSP 인증 정보는 서버 환경에 보관합니다.

## 실행시 생길수 있는 이슈

| 증상 | 확인할 항목 |
| --- | --- |
| 로그인·회원가입 503 | PostgreSQL 서비스와 `CCTV_DATABASE_URL`을 확인합니다. 기존 Mac 소켓 DB는 전용 실행 스크립트를 사용합니다. |
| 분석 워커 연결 끊김 | 단독 Uvicorn을 종료하고 `backend.run_local` 또는 Mac 실행 스크립트로 웹과 워커를 함께 시작합니다. `관제실 영상 분석 요청을 기다립니다.` 로그를 확인합니다. |
| 포트 8000 사용 중 | 기존 실행 터미널을 찾아 `Ctrl+C`로 종료합니다. 다른 포트를 쓸 경우 카카오 콜백과 관제 주소 설정도 맞춥니다. |
| 화면이 비어 있음 | 프런트엔드 빌드가 성공했는지 확인한 뒤 실행기를 다시 시작합니다. |
| 등록한 행동 모델이 적용되지 않음 | **모델 설정**에서 새 모델을 선택하고 **다시 분석**을 눌러 타일의 실제 모델명과 해시를 확인합니다. |
| 카톡은 오지만 사건 사진이 보이지 않음 | 사진 게이트웨이, 공개 HTTPS 주소, 서명 키와 사건의 사진 보관 기한을 확인합니다. |

API와 DB 준비 상태는 다음 명령으로 확인합니다.

```powershell
(Invoke-RestMethod http://127.0.0.1:8000/health/ready).status
```

```bash
curl -fsS http://127.0.0.1:8000/health/ready
```

## 학습과 검증

재학습 절차는 [AI-Hub 기반 재학습](train/behavior_repair/README.md)과 [TensorFlow 학습 및 변환](train/behavior_tf/README.md)에 있습니다. 학습 코드에서 출력한 호환 행동 가중치는 관제 화면에 첨부한 뒤, 학습에 사용하지 않은 원본 영상에서 평가합니다.

```powershell
# Windows PowerShell
& $Py -m pip install -r backend/requirements-test.txt
& $Py -m unittest discover -s backend/tests
npm.cmd --prefix frontend test
```

```bash
# macOS Terminal
.venv-web/bin/python -m pip install -r backend/requirements-test.txt
.venv-web/bin/python -m unittest discover -s backend/tests
npm --prefix frontend test
```

PostgreSQL 통합 테스트를 포함하려면 임시 스키마를 만들 수 있는 개발 DB의 접속 정보를 `CCTV_TEST_DATABASE_URL`에 설정합니다.

## 프로젝트 안내

학술 목적으로 제작된 프로젝트입니다.
