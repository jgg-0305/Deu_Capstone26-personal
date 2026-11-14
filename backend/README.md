# CCTV 웹 백엔드 실행 가이드

Python 3.12 이상, PostgreSQL 16 이상, Node.js 22.12 이상을 준비한다. 명령은 저장소 루트에서 실행한다. PostgreSQL에 개발용 사용자와 빈 데이터베이스 `cctv`를 만들고 예제의 `USER:PASSWORD`를 실제 연결 값으로 바꾼다. 특수 문자가 있는 비밀번호는 URL 인코딩한다. `admin / admin`은 웹 로그인용 테스트 계정이며 PostgreSQL 연결에는 별도의 DB 계정을 사용한다. 서버를 열기 전에 같은 터미널에서 마이그레이션과 관리자 생성 명령이 성공했는지 확인한다.

## Windows PowerShell

첫 번째 터미널에서 의존성을 설치하고 데이터베이스와 모델을 등록한다.

```powershell
py -3.12 -m venv .venv-web
$Py = ".\.venv-web\Scripts\python.exe"
& $Py -m pip install -r backend/requirements.txt
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $Py -m pip install -r backend/requirements-worker.txt
$env:CCTV_DATABASE_URL = 'postgresql://USER:PASSWORD@127.0.0.1:5432/cctv'
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
& $Py -m backend.manage migrate
& $Py -m backend.manage register-models
$env:CCTV_ENABLE_TEST_ADMIN = '1'
& $Py -m backend.manage seed-test-admin
$env:CCTV_WORKER_TOKEN = & $Py -c "import secrets; print(secrets.token_urlsafe(32))"
$env:CCTV_WORKER_TOKEN
& $Py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

출력된 워커 토큰은 본인만 보관한다. 두 번째 PowerShell에서 분석 워커를 실행한다.

```powershell
$Py = ".\.venv-web\Scripts\python.exe"
$env:CCTV_WORKER_TOKEN = Read-Host '첫 번째 터미널의 워커 토큰'
& $Py -m backend.worker --serve
```

## macOS Terminal

첫 번째 터미널에서 실행한다. Apple Silicon에서는 네이티브 ARM64 Python을 사용한다.

```bash
python3 -m venv .venv-web
source .venv-web/bin/activate
python -m pip install -r backend/requirements.txt -r backend/requirements-worker.txt
export CCTV_DATABASE_URL='postgresql://USER:PASSWORD@127.0.0.1:5432/cctv'
npm --prefix frontend ci
npm --prefix frontend run build
python -m backend.manage migrate
python -m backend.manage register-models
export CCTV_ENABLE_TEST_ADMIN=1
python -m backend.manage seed-test-admin
export CCTV_WORKER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf '%s\n' "$CCTV_WORKER_TOKEN"
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

출력된 워커 토큰은 본인만 보관한다. 두 번째 터미널에서 실행한다.

```bash
source .venv-web/bin/activate
printf '첫 번째 터미널의 워커 토큰: '
read -rs CCTV_WORKER_TOKEN
export CCTV_WORKER_TOKEN
printf '\n'
python -m backend.worker --serve
```

## 화면에서 확인

1. 브라우저에서 [영상 관제](http://127.0.0.1:8000/)를 열고 회원가입 후 로그인한다. 로컬 테스트 관리자로 접속할 때는 아이디와 비밀번호에 각각 `admin`을 입력한다.
2. 오른쪽 **영상 소스 추가 → 파일 추가…**에서 MP4를 선택하고 표시 이름을 입력한다. **영상 등록**을 누르면 바로 분석 대기열에 들어간다. 내장 영상은 **기본 샘플**에서 등록한다.
3. 별도 워커가 대기 중인 작업을 가져간다. 분석 중에는 타일에 AI가 처리한 프레임이 표시되고 인원 수, 행동 분류, FPS가 갱신된다. 파일은 일반 영상 플레이어에서 탐색할 수 있고 **원본 / AI 분석**으로 같은 영상 위의 오버레이를 전환한다. AI 분석 화면은 분석 결과가 아직 없으면 자동 대기한 뒤 이어서 재생한다. 이전 분석에는 오버레이가 없을 수 있으므로 **다시 분석**을 실행한다. 타일에서 영상을 재생하거나 바둑판·계단식 정렬과 집중 보기를 전환할 수 있다.
4. **사건 기록**에서 소스와 판정 방식을 필터링한다. 사건을 선택하면 영상의 감지 시점으로 이동한다. 운영자는 검토 결과를 저장하거나 **이 사건 삭제**를 누를 수 있다.
5. **저장 공간**에서 원본 영상과 분석 파일의 실제 용량을 확인한다. **영상·기록 삭제**는 해당 소스의 분석을 중단하고 업로드 원본·분석 파일·관련 사건을 정리한다. 이미 관제 화면에서 제거한 소스도 여기서 정리할 수 있다.

명령으로 한 번만 분석하려면 `--serve` 대신 `--camera-id UUID`를 사용한다. 화면에서 이미 요청한 작업은 `--serve` 워커가 처리한다.

React 화면을 수정하며 확인할 때는 API를 8000번 포트에서 실행한 상태로 별도 터미널에서 Windows의 경우 `npm.cmd --prefix frontend run dev`, macOS의 경우 `npm --prefix frontend run dev`를 실행한다. Vite가 표시하는 주소에서 변경 사항을 확인한다. FastAPI가 제공하는 주소에 변경 사항을 반영하려면 다시 빌드한다.

기존 소스와 사건의 소유권을 이어받으려면 새 계정을 만들기 전에 기존 사용자 UUID를 확인하고 계정에 로그인 정보를 설정한다. 명령에서 새 비밀번호를 대화형으로 입력한다.

Windows PowerShell:

```powershell
& $Py -m backend.manage set-login --user-id UUID --username myuser
```

macOS Terminal:

```bash
python -m backend.manage set-login --user-id UUID --username myuser
```

일반 회원은 자신의 소스와 사건을 관리한다. 테스트 관리자는 로컬 검증용 운영자 계정이며 `CCTV_ENABLE_TEST_ADMIN=1`을 설정한 서버에서만 접속할 수 있다. 외부 배포에서는 이 설정을 제거하고 HTTPS를 적용한다.

영상 재생은 원본 MP4와 사건의 영상 내 시점을 사용한다. 미디어는 소유권을 확인한 뒤 브라우저의 구간 요청으로 전송한다. 별도 카카오 발송은 후속 연동이다. 외부 서버에서는 HTTPS를 적용하고 프록시에서 `/internal/`을 공개 경로에서 제외한다.

## 이상행동 모델 선택

**모델 설정**에서 분류 모델을 고른 다음 소스를 등록하거나 기존 소스에서 **다시 분석**을 누른다. 소스 타일에는 해당 실행에 사용한 모델이 표시된다.

| 모델 | 체크포인트 | 결과 범위 |
| --- | --- | --- |
| KeypointLSTM v2 (기본) | `train/1_behavior/output/kpt_behavior.pth` | 사람별 17관절의 위치와 변화량 |
| KeypointLSTM v1 | `kpt_behavior.pth` | 사람별 17관절 위치 |
| ResNet50 + LSTM | `model_final.pth` | BGR 3프레임 기반의 장면 전체 분류 |

v1과 ResNet50 + LSTM은 로컬 가중치가 있을 때 등록된다. 파일을 놓은 후 OS별 설치 절차의 `python -m backend.manage register-models`를 실행한다. `train/2_keypoint`는 UniPose 관절 추정 학습 코드이고, `train/3_stand`의 `model/objects.pt`는 가판대와 POS를 검출한다. 가판대 관련 도난 의심은 사람 관절과 두 영역을 함께 사용하는 규칙으로 기록된다.

## 웹캠과 IP 카메라 연결

- **웹캠 추가**에서 이 기기의 카메라를 선택하고 **웹캠 연결**을 누른다. 처음 연결할 때 브라우저 및 운영체제의 카메라 권한을 허용한다. 외부 주소에서는 HTTPS가 필요하다. 타일의 **×**를 누르면 웹캠과 분석이 종료된다. 브라우저를 닫거나 프레임 송신이 25초 동안 멈추면 워커가 웹캠 연결을 종료한다.
- **IP 카메라 (RTSP)**에서 서버가 허용한 주소와 표시 이름을 입력하고 **IP 카메라 등록**을 누른다. 등록과 동시에 분석이 대기열에 들어간다. 타일의 **연결**로 원본 미리보기를 시작할 수 있다. PostgreSQL에는 표시 이름과 인증 정보가 없는 주소가 저장된다.
- 파일·RTSP·웹캠 모두 등록할 때 자동으로 분석을 요청한다. 파일 작업과 실시간 작업은 별도 실행 슬롯을 사용한다. 한 워커의 기본 실시간 슬롯은 3개이며 `--live-slots`로 조절할 수 있다.
- **모델 설정**에서 처리 해상도, 검출 간격, 포즈·행동 신뢰도, DeepSORT 최대 소실 프레임을 설정한다. 새 소스에 즉시 적용되며 **다시 분석**을 누르면 선택한 소스를 새 설정으로 재시작한다. **표시 옵션**은 현재 화면에 즉시 반영된다.
- **×**를 누르면 소스가 관제 화면에서 제거되고 분석이 종료된다. 이미 저장된 사건과 MP4 재생은 사건 기록에서 계속 조회할 수 있다. 웹캠·RTSP는 사건의 시각과 분류 결과를 저장한다.

RTSP를 사용하려면 Uvicorn을 시작하는 터미널에서 사용을 허용할 **웹 로그인 계정**과 **카메라 주소**를 설정한다. 인증이 필요한 카메라는 서버 설정에만 `rtsp://카메라계정:카메라비밀번호@IP:포트/경로`를 넣는다. 브라우저에서는 계정과 비밀번호를 뺀 주소를 입력한다. URL에 들어가는 인증 정보에 특수 문자가 있다면 URL 인코딩한다. 이 설정을 저장소에 커밋하지 않는다.

Windows PowerShell:

```powershell
$env:CCTV_RTSP_USERS = 'myuser'
$env:CCTV_RTSP_URLS = '["rtsp://192.168.1.50:554/live"]'
```

macOS Terminal:

```bash
export CCTV_RTSP_USERS='myuser'
export CCTV_RTSP_URLS='["rtsp://192.168.1.50:554/live"]'
```

계정은 쉼표로 구분하며 URL은 JSON 배열로 설정한다. RTSP는 기본적으로 비활성화되어 있다. FastAPI 서버에 앞의 워커 의존성을 설치하지 않았다면 같은 Python 환경에 `opencv-python`을 설치한다. 환경을 변경한 뒤 Uvicorn을 재시작한다. 서버가 카메라 네트워크에 연결되어 있어야 하며, 외부 배포에서는 방화벽에서 중계 프로세스의 외부 연결을 신뢰하는 카메라 주소와 포트로 제한한다.

## 로그인과 회원가입에서 503이 나올 때

Windows PowerShell에서 `(Invoke-RestMethod http://127.0.0.1:8000/health/ready).status`, macOS Terminal에서 `curl -fsS http://127.0.0.1:8000/health/ready`를 실행한다. `ready` 응답은 데이터베이스 연결과 로그인 스키마 적용을 확인한다.

연결 오류가 나면 PostgreSQL을 실행하고 위 OS별 가이드의 연결 예시를 실제 DB 사용자와 비밀번호로 바꾼다. 마이그레이션 오류가 나면 같은 연결 정보로 마이그레이션 명령을 실행한다. 환경 변수를 바꾼 경우 Uvicorn을 재시작해야 새 값이 적용된다.

## AI-Hub 데이터로 테스트

[AI-Hub 실내(편의점, 매장) 사람 이상행동 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71550)의 `Validation` 원천 영상을 사용한다.

- **Windows**에서는 압축을 푼 폴더에서 개별 `.mp4`를 선택한다.
- **macOS**에서는 이미 처리해 둔 `combined_video_data` 폴더를 열고 개별 `.mp4`를 선택한다. 같은 폴더에 보이는 `combined_video_data.zip`이나 `.zip.part숫자`는 압축 자료이므로, 원본만 가지고 있다면 숫자 오프셋 순서로 조각을 합쳐 ZIP으로 만든 뒤 압축을 푼다.
- 파일 한 개의 등록 한도는 256 MiB다. 등록한 영상은 로컬 `.local-videos`에 저장되며 계정으로 보호된다. API와 워커는 같은 저장소와 영상 디렉터리를 사용해야 한다.
- 영상을 등록한 뒤 **완료**를 기다린다. 사건 기록에는 실제 모델이나 규칙이 감지한 항목만 표시된다. 데이터셋의 라벨과 결과를 비교하는 정확도 평가는 별도다.


## 원본 영상 설정으로 업데이트하고 재검증하기

기존 API와 워커 터미널에서 각각 `Ctrl+C`로 종료한 뒤 실행한다. 8000번 포트를 유지한다. 기존 DB 연결과 워커 토큰 설정을 그대로 사용하며, 새 토큰을 만들 때는 API와 워커 양쪽에 같은 값을 설정해야 한다.

Windows PowerShell — 기존 DB 설정을 한 첫 번째 터미널:

```powershell
$Py = ".\.venv-web\Scripts\python.exe"
curl.exe -fL "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt" -o yolo11s-pose.pt
& $Py -m backend.manage register-models
npm.cmd --prefix frontend run build
& $Py -m unittest discover -s backend/tests -v
npm.cmd --prefix frontend test
& $Py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Windows 두 번째 터미널 — 기존 API와 같은 워커 토큰을 설정한 상태:

```powershell
& .\.venv-web\Scripts\python.exe -m backend.worker --serve
```

macOS Terminal — 기존 DB 설정을 한 첫 번째 터미널:

```bash
source .venv-web/bin/activate
# 현재 프로젝트에 보관한 로컬 DB 설정이 있으면 불러온다.
if [ -f .local-db/web.env ]; then source .local-db/web.env; fi
curl -fL 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt' -o yolo11s-pose.pt
python -m backend.manage register-models
npm --prefix frontend run build
python -m unittest discover -s backend/tests -v
npm --prefix frontend test
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

macOS 두 번째 터미널 — 기존 API와 같은 워커 토큰을 설정한 상태:

```bash
source .venv-web/bin/activate
if [ -f .local-db/web.env ]; then source .local-db/web.env; fi
python -m backend.worker --serve
```

YOLO11s를 이미 다운로드한 경우 `curl` 단계는 생략한다. 모델 등록 명령이 파일 크기와 SHA-256을 검증한다. s 가중치는 선택 설치 파일이며, 없어도 기존 n 모델은 사용할 수 있다. 현재 작업한 Mac에는 s 모델 설치·등록과 빌드가 완료돼 있다. Windows 명령은 함께 제공하지만 이번 실기 검증은 macOS에서 했다.

1. 브라우저를 새로고침한다. **모델 설정 → 원본 영상 기본값**을 누른다.
2. `YOLO11s`, `KeypointLSTM v2`, 해상도 `480`, 검출 간격 `1`, 포즈 `40%`, 행동 표시 `50%`, 사건 저장 `60%`인지 확인한다. 버튼은 설정을 채우며 기존 실행을 자동 변경하지 않는다.
3. 기존 소스를 선택하고 **다시 분석**을 누르거나 새 영상을 등록한다. 이전 실행은 기록으로 보존되고 새 설정의 결과가 생성된다.
4. 영상 타일에서 처리 프레임, 사람 검출 횟수, 행동 추론 횟수를 확인한다. **관절 수집 n/30**은 모델 고장이 아니라 동일 인물의 입력 수집 단계다. 새 트랙은 다시 수집한다.
5. **AI 분석**에서 재생한다. 결과가 늦으면 대기 후 자동으로 이어지며, 완료 후에는 원본 속도로 재생·탐색한다. **원본**에서는 분석을 기다리지 않고 재생한다.
6. **사건 기록**에서 모델 점수와 규칙 판정을 구분해 보고 해당 시점의 영상을 재생한다.

`demo-2.mp4`는 기존 GUI의 화면 녹화라 원본 파일에 과거 관절·라벨이 포함돼 있다. 원본/AI 전환은 이번 웹이 그리는 표시만 전환한다. 순수한 정확도 비교에는 화면 녹화 대신 실제 CCTV MP4와 정답 구간이 필요하다. 자세한 원인과 실제 비교 수치는 [데모 회귀 검토](../docs/design/demo-regression-review.md)에 기록했다.


## 원본 데이터셋 재검증

`demo-2.mp4`는 참고 화면 녹화다. 아래 검사는 AI-Hub 원본 MP4 6개를 파일 목록대로 실행한다. `combined_video_data` 폴더의 전체 경로를 입력한다. 결과 폴더의 `results.json`과 영상별 `*-trace.json`에는 추론 횟수·클래스 분포·추적 상태·영상 시각이 저장된다. 원본 영상과 가중치는 수정하지 않는다. DB나 실행 중인 웹 서버 없이도 검사할 수 있다.

Windows PowerShell:

```powershell
$Py = ".\.venv-web\Scripts\python.exe"
$DatasetDir = Read-Host 'combined_video_data 폴더 전체 경로'
& $Py -m backend.evaluate_videos --manifest docs/design/raw-video-test-set.json --data-root "$DatasetDir" --output .local-evaluation/raw --rgb
& $Py -m unittest discover -s backend/tests -v
npm.cmd --prefix frontend test
```

macOS Terminal:

```bash
source .venv-web/bin/activate
printf 'combined_video_data 폴더 전체 경로: '
read -r DATASET_DIR
python -m backend.evaluate_videos --manifest docs/design/raw-video-test-set.json --data-root "$DATASET_DIR" --output .local-evaluation/raw --rgb
python -m unittest discover -s backend/tests -v
npm --prefix frontend test
```

3 FPS 원본의 관절 입력 30프레임은 10초다. **원본 영상 기본값**은 사람 검출 40%이고, 90%의 **데모 비교용 설정**은 화면 녹화와의 비교를 위해 따로 남겨 두었다. 영상에 사람이 실제로 보이지 않거나 관절이 가려진 구간은 `관절 미관측 · 추적 유지`로 표시된다. 실제로 측정하지 못한 관절 좌표를 만들어 채우지 않는다.

현재 체크포인트는 원본 영상에서 오분류·미탐이 남아 있다. 사건이 저장됐다는 사실만으로 제대로 감지됐다고 평가하지 않는다. 이번 실측 결과는 [원본 영상 검토](../docs/design/raw-video-validation.md)에 있다. 정확도 계산과 재학습에는 XML 라벨과 정상·다른 행동 데이터가 필요하다. Windows 명령은 함께 제공하며 이번 실제 실행 검증은 macOS에서 했다.

## 사건 삭제와 저장 공간 정리

- **사건 기록 → 사건 선택 → 이 사건 삭제**: 선택한 사건을 목록과 집계에서 제외한다. 원본 MP4는 여러 사건이 공유하므로 유지한다. 분석 중인 소스는 새 사건을 계속 기록할 수 있다.
- **저장 공간 → 영상·기록 삭제 → 삭제 확인**: 해당 소스의 분석 대기열과 실행을 중단하고, 모든 실행에 연결된 사건과 업로드 원본·분석 시간축·미리보기 파일을 삭제한다. 삭제 대상과 용량을 확인한 후 실행한다. 관제 화면의 `×`로 제거했던 소스도 이 목록에 포함된다.
- 다른 소스가 공유하는 원본, 저장소의 기본 샘플, 처음 파일 선택 창에서 골랐던 PC·외장 디스크의 원본은 유지한다. 업로드 과정에서 서버에 만든 복사본만 삭제 대상이다.
- 파일을 지울 수 없으면 **정리 재시도**가 표시된다. 영상 재생이나 파일을 사용하던 프로그램을 종료한 후 다시 누른다. 특히 Windows에서는 열린 파일이 삭제를 막을 수 있다. 이미 삭제한 파일은 반복 요청해도 오류 없이 처리한다.
- 파일 삭제 요청을 DB에 먼저 확정하므로 파일 삭제 중 실패하거나 서버가 중단돼도 재시도할 수 있다. 삭제된 사건이 워커 재전송으로 다시 나타나지 않도록 작은 식별·검토 기록은 DB에 보존한다. 영상과 분석 파일을 제거하는 기능이며 PostgreSQL 전체 크기를 0으로 만드는 기능은 아니다.

기존 서버를 업데이트할 때 API 터미널에서 `Ctrl+C`로 종료한 뒤, 기존 DB 연결·워커 토큰을 유지하고 실행한다. 워커 실행 명령은 변경되지 않았다.

Windows PowerShell:

```powershell
$Py = ".\.venv-web\Scripts\python.exe"
& $Py -m backend.manage migrate
npm.cmd --prefix frontend run build
& $Py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

macOS Terminal:

```bash
source .venv-web/bin/activate
if [ -f .local-db/web.env ]; then source .local-db/web.env; fi
python -m backend.manage migrate
npm --prefix frontend run build
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

현재 작업한 Mac에는 `007_storage_cleanup.sql` 적용과 API 재시작이 완료됐다. 브라우저를 새로고침하면 **저장 공간** 메뉴가 나타난다.

서버 재시작 후 보류된 파일 삭제를 명령으로 다시 처리할 수도 있다. 이미 사용자가 삭제를 확인한 대상만 처리한다.

Windows PowerShell:

```powershell
& .\.venv-web\Scripts\python.exe -m backend.manage cleanup-storage
```

macOS Terminal:

```bash
python -m backend.manage cleanup-storage
```

삭제 통합 검사는 개발용 PostgreSQL에 임시 스키마를 만들고 임시 파일만 사용한다. 종료 시 임시 스키마를 정리한다. 계정에 스키마 생성 권한이 있어야 한다. `CCTV_TEST_DATABASE_URL`이 없으면 DB 통합 검사만 건너뛴다.

Windows PowerShell:

```powershell
$Py = ".\.venv-web\Scripts\python.exe"
& $Py -m pip install -r backend/requirements-test.txt
$env:CCTV_TEST_DATABASE_URL = $env:CCTV_DATABASE_URL
& $Py -m unittest discover -s backend/tests -v
npm.cmd --prefix frontend test
Remove-Item Env:CCTV_TEST_DATABASE_URL
```

macOS Terminal:

```bash
python -m pip install -r backend/requirements-test.txt
export CCTV_TEST_DATABASE_URL="$CCTV_DATABASE_URL"
python -m unittest discover -s backend/tests -v
npm --prefix frontend test
unset CCTV_TEST_DATABASE_URL
```

macOS에서 Python 검사 17개, 프런트엔드 검사 3개, React 빌드가 통과했다. 독립 Chrome 검증에서는 사건 한 건 삭제 후 원본 보존, 삭제 취소 시 데이터 보존, 전체 삭제 후 실제 업로드 파일 제거와 **42.7 MB → 0 B**를 확인했다. 실제 사용자 자료를 지우는 방식으로 검증하지 않았다. Windows 명령은 함께 제공하며 Windows 실기 검증은 수행하지 않았다.

현재 절도 영상의 흡연 오분류와 재학습 판단은 [행동 모델 진단](../docs/design/theft-smoking-diagnosis.md)에 있다.

## 행동 모델 파일 첨부

**모델 설정 → 행동 모델 파일 첨부 (.pth)**에서 파일을 선택한다. 등록이 끝나면 선택한 파일과 해시가 표시된다. 기존 영상에는 **다시 분석**으로 적용한다. 영상 타일에는 실제 실행에 사용한 파일 이름과 SHA-256 앞 12자리가 표시된다.

원본 GUI의 **찾기**는 로컬 파일 경로를 영상 처리 스레드에 전달한다. 웹에서는 원본 GUI와 같은 30프레임·17관절·8클래스 KeypointLSTM v1/v2 가중치를 서버에 첨부한다. 클래스 순서는 정상, 전도, 파손, 방화, 흡연, 유기, 절도, 폭행이다. 상태 사전에는 클래스 이름이 없을 수 있으므로 첨부 전에 학습 코드의 클래스 순서를 확인한다. 모델 등록은 구조와 출력값을 검사하며 정확도 평가는 별도로 진행한다.

새로 첨부하는 KeypointLSTM 가중치는 최대 8 MiB의 PyTorch ZIP 기반 상태 사전을 지원한다. 프로젝트의 `model/model_final.pth` 또는 동일한 루트 파일을 첨부하면 SHA-256을 확인해 기존 **ResNet50 + LSTM · 장면 전체** 모델을 선택한다. 새 장면 모델 파일은 별도 등록이 필요하다. 기존 기본 모델과 동일한 파일은 SHA-256으로 식별해 등록된 기본 모델을 선택한다. 새 가중치는 로컬 `.local-models`에 저장되므로 API와 워커가 같은 프로젝트 디렉터리 또는 같은 모델 볼륨을 사용해야 한다.

기존 서버는 API와 워커를 종료하고 같은 DB 연결 및 워커 토큰으로 다시 실행한다.

Windows PowerShell:

```powershell
$Py = ".\.venv-web\Scripts\python.exe"
& $Py -m backend.manage migrate
npm.cmd --prefix frontend run build
& $Py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

별도 PowerShell에서 동일한 워커 토큰을 설정하고 실행한다.

```powershell
& .\.venv-web\Scripts\python.exe -m backend.worker --serve
```

macOS Terminal:

```bash
source .venv-web/bin/activate
if [ -f .local-db/web.env ]; then source .local-db/web.env; fi
python -m backend.manage migrate
npm --prefix frontend run build
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

별도 터미널에서 같은 가상 환경과 워커 토큰을 불러와 실행한다.

```bash
source .venv-web/bin/activate
if [ -f .local-db/web.env ]; then source .local-db/web.env; fi
python -m backend.worker --serve
```
