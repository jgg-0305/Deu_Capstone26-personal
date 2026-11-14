# CCTV 이상 현상 감지 시스템

YOLO11 + DeepSORT + KeypointLSTM 기반 실시간 **이상행동 감지** 프로그램(PyQt5).
인물을 감지·추적하며 17관절 키포인트 시퀀스로 **인물(트랙 ID)별** 이상행동을 분류하고,
매장 **가판대·결제기**를 감지해 **도난 규칙**까지 판정한다. 이상행동 발생 시 자동 녹화한다.

## 파이프라인

```
프레임 → YOLO pose (yolo11n/s/m-pose)  → bbox + 17관절
       → DeepSORT                       → track_id
       → KeypointLSTM (kpt_behavior.pth) → 인물별 (행동, 신뢰도)
       → 객체 감지 (objects.pt)          → 가판대·결제기 영역
       → 도난 규칙 (theft_monitor)       → 손이 가판대→결제기 미경유 시 도난
       → 박스·키포인트·라벨·영역 그리기 + 이상행동 녹화
```

`kpt_behavior.pth` / `objects.pt` 가 없어도 감지·추적·키포인트는 정상 동작한다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| **인간 감지 + 포즈** | YOLO11 pose 로 인물 bbox + COCO 17관절 스켈레톤 |
| **다중 객체 추적** | DeepSORT 로 인원별 고유 트랙 ID + 색상 |
| **인물별 이상행동 분류** | KeypointLSTM — 트랙 ID별 8종 행동, 이상행동 시 빨간 박스 + `"ID:1 \| ASSAULT 92%"` |
| **가판대·결제기 감지** | 커스텀 YOLO(`objects.pt`) — 가판대(주황)·결제기(파랑) 반투명 영역, 각각 on/off |
| **도난 규칙** | 손이 가판대에서 물품 집고 결제기 미경유 시 도난(절도)으로 판정 |
| **이벤트 녹화** | 이상행동 발생 구간 자동 녹화 + `recordings.json`(발생 시각·행동·신뢰도) |
| **녹화 기록 메뉴** | 상단 메뉴바 "녹화 기록" → 목록·재생·삭제 |
| **다중 소스 + MDI** | 파일·웹캠·RTSP 동시 연결, 영상창 자유 이동·크기조절, 영상별 일시정지·클릭 탐색 재생바 |
| **IP 카메라** | RTSP 스트림 + 끊김 시 자동 재연결 |
| **경량화** | 감지 모델(n/s/m)·해상도(384/480/640)·검출 간격·CUDA FP16 (Raspberry Pi 5 CPU 대응) |

이상행동 8종: `정상 / 전도(쓰러짐) / 파손 / 방화 / 흡연 / 유기 / 절도 / 폭행`

## 기술 스택

| 라이브러리 | 용도 |
|---|---|
| ultralytics (YOLO11) | 포즈 추정 · 커스텀 객체 감지 |
| deep-sort-realtime | 다중 객체 추적 |
| opencv-python | 영상 입출력·프레임 처리 |
| PyQt5 | GUI |
| torch + torchvision | 추론 백엔드 (CUDA 권장) |

## 프로젝트 구조

```
proto-cctv/
├── main.py                          # 진입점 (Qt 플러그인 경로 설정)
├── run.bat                          # 더블클릭 실행
├── core/
│   ├── detector.py                  # YOLO pose 감지기 + ObjectDetector(가판대·결제기)
│   ├── tracker.py                   # DeepSORT 트래커
│   ├── kpt_behavior_classifier.py   # KeypointLSTM 행동 분류 + 트랙 버퍼 (모델 정의 정본)
│   ├── theft_monitor.py             # 규칙 기반 도난 감지
│   ├── recorder.py                  # 이벤트 녹화 + recordings.json
│   ├── labels.py                    # 이상행동 8종 라벨
│   └── video_thread.py              # QThread 통합 처리 파이프라인
├── gui/
│   ├── main_window.py               # MDI 메인 윈도우
│   └── recordings_dialog.py         # 녹화 기록 보기
├── model/
│   ├── objects.pt                   # 가판대·결제기 감지 (앱 사용)
│   └── model_final.pth              # 레거시 ConvLSTM (앱 미사용)
├── recordings/                      # 녹화 영상 + recordings.json (런타임 생성)
└── train/
    ├── 1_behavior/                  # KeypointLSTM 행동 분류 학습
    └── 3_stand/                     # 가판대·결제기 객체 감지 학습
```

## 설치 및 실행

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
$Py = ".\.venv\Scripts\python.exe"
& $Py -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
& $Py main.py
```

### macOS Terminal

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements-worker.txt PyQt5
python main.py
```

## 웹 백엔드

웹 관제실에서는 MP4·웹캠·RTSP 카메라를 추가하고 바둑판, 계단식, 집중 보기로 배치할 수 있습니다. PostgreSQL 초기화와 Windows·macOS에서의 실행 및 카메라 연결 방법은 [백엔드 실행 가이드](backend/README.md)를 참고하세요.

## 사용 방법

1. **영상 소스 추가** — 파일 / 웹캠 / RTSP. 각 영상은 독립 창(MDI)으로 뜨며 자유롭게 이동·크기조절.
2. 화면에 인물 박스·트랙 ID·키포인트·`"ID:n \| 행동 %"` 라벨, 가판대·결제기 영역이 표시되고,
   우측 패널에 감지 인원·FPS·대표 이상행동이 갱신된다. 이상행동 인물은 박스가 빨간색으로 바뀐다.
3. **표시 옵션** 체크박스로 박스 / 트랙 ID / 키포인트 / 행동 라벨 / 가판대 / 결제기를 실시간 on/off.
4. 동영상 파일은 **재생바**로 원하는 위치 클릭 탐색, **⏸/▶** 버튼으로 영상별 일시정지.
5. 이상행동 발생 시 자동 녹화되며, 상단 **녹화 기록** 메뉴에서 목록·재생·삭제.

> 경량 PC: `yolo11n-pose` + 해상도 384 권장. CUDA에서는 FP16·GPU 임베더 자동 적용.

## 학습 (재훈련)

학습 데이터와 프로젝트 가중치는 별도로 백업한다. 학습 스크립트는 결과 파일을 갱신하므로, 웹 백엔드에서 새 가중치를 사용하려면 모델 manifest의 버전·해시도 갱신해야 한다.

### Windows PowerShell

```powershell
$Py = (Resolve-Path ".\.venv\Scripts\python.exe").Path
Push-Location train/1_behavior
Remove-Item split/kpt_train*.npz, split/kpt_val.npz -ErrorAction SilentlyContinue
& $Py extract_kpts_video.py
Copy-Item split/kpt_train.npz split/kpt_train_raw.npz
& $Py refine_dataset.py --in-npz split/kpt_train_raw.npz --out-npz split/kpt_train.npz
& $Py demo_onset.py --repeat 20
& $Py train_kpt.py
Pop-Location
Push-Location train/3_stand
& $Py auto_label.py
& $Py train_stand.py
Pop-Location
```

### macOS Terminal

```bash
source .venv/bin/activate
(
  cd train/1_behavior || exit
  find split -maxdepth 1 -type f \( -name 'kpt_train*.npz' -o -name 'kpt_val.npz' \) -delete
  python extract_kpts_video.py
  cp split/kpt_train.npz split/kpt_train_raw.npz
  python refine_dataset.py --in-npz split/kpt_train_raw.npz --out-npz split/kpt_train.npz
  python demo_onset.py --repeat 20
  python train_kpt.py
)
(
  cd train/3_stand || exit
  python auto_label.py
  python train_stand.py
)
```

## 라이선스

학술 목적으로 제작되었습니다.
