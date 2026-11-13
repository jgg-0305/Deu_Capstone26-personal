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

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt        # CUDA면 PyTorch 별도 선설치 권장

.\.venv\Scripts\python.exe main.py     # 또는 run.bat 더블클릭
```

## 카카오 알림 설정

카카오 인증 정보는 Git에 포함되지 않습니다. 환경 변수 `KAKAO_REST_API_KEY`와
`KAKAO_ACCESS_TOKEN`을 설정하거나 프로젝트 루트의 `kakao_local_config.json`에
다음 형식으로 저장하세요. 액세스 토큰은 카카오 로그인으로 발급받을 수도 있습니다.

```json
{
  "rest_api_key": "YOUR_KAKAO_REST_API_KEY",
  "access_token": "YOUR_KAKAO_ACCESS_TOKEN"
}
```

로그인으로 발급받은 토큰은 `kakao_tokens.json`에 저장되며, 두 JSON 파일은
Git에서 제외됩니다.

## 사용 방법

1. **영상 소스 추가** — 파일 / 웹캠 / RTSP. 각 영상은 독립 창(MDI)으로 뜨며 자유롭게 이동·크기조절.
2. 화면에 인물 박스·트랙 ID·키포인트·`"ID:n \| 행동 %"` 라벨, 가판대·결제기 영역이 표시되고,
   우측 패널에 감지 인원·FPS·대표 이상행동이 갱신된다. 이상행동 인물은 박스가 빨간색으로 바뀐다.
3. **표시 옵션** 체크박스로 박스 / 트랙 ID / 키포인트 / 행동 라벨 / 가판대 / 결제기를 실시간 on/off.
4. 동영상 파일은 **재생바**로 원하는 위치 클릭 탐색, **⏸/▶** 버튼으로 영상별 일시정지.
5. 이상행동 발생 시 자동 녹화되며, 상단 **녹화 기록** 메뉴에서 목록·재생·삭제.

> 경량 PC: `yolo11n-pose` + 해상도 384 권장. CUDA에서는 FP16·GPU 임베더 자동 적용.

## 학습 (재훈련)

### 행동 분류 모델 (`kpt_behavior.pth`)
`train/1_behavior/split/data/` 하위 클래스 폴더(`01/04/05` 정상, `07~13` 이상행동)의 mp4 사용.

```powershell
cd train\1_behavior
Remove-Item split\kpt_train*.npz, split\kpt_val.npz -EA SilentlyContinue

python extract_kpts_video.py    # ① 영상 → 키포인트 npz
Copy-Item split\kpt_train.npz split\kpt_train_raw.npz
python refine_dataset.py --in-npz split\kpt_train_raw.npz --out-npz split\kpt_train.npz  # ② 정제
python demo_onset.py --repeat 20   # ③ 데모 영상 온셋(시작 프레임) 시간 라벨 추가
python train_kpt.py             # ④ 학습 → output/kpt_behavior.pth 자동 배포
```

### 가판대·결제기 모델 (`objects.pt`)
```powershell
cd train\3_stand
python auto_label.py    # YOLO-World 로 매장 영상 의사 라벨링
python train_stand.py   # 학습 → model/objects.pt 자동 배포
```

## 라이선스

학술 목적으로 제작되었습니다.

## RTSP 테스트 계정
`rtsp://capstonetc72deu:officedeuackr26@172.30.1.93:554/stream1`
