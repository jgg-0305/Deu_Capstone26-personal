# 행동 분류기 TensorFlow 재학습

사용자가 제공한 폴더는 AI-Hub의 `Validation/01.원천데이터`다. 절도 영상 이름 80개가 두 폴더에 있으며, 병합된 ZIP 안에도 MP4 80개만 있다. XML은 발견되지 않았고 저장소의 기존 `split/*.csv`에서 해당 영상 이름과 일치하는 라벨도 없었다. 원본은 학습에 섞지 않고, 대응 라벨을 받으면 별도 평가에 사용한다. [AI-Hub 데이터와 라벨 형식](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71550)

## 데이터 준비

1. AI-Hub에서 **Training 원천 영상과 대응 라벨링 데이터**, 그리고 평가에 사용할 **Validation 라벨링 데이터**를 받는다. 정상 움직임과 구매·상품 취급, 전도, 파손, 방화, 흡연, 유기, 절도, 폭행이 모두 필요하다.
2. 영상과 라벨을 대조해 다음 헤더의 UTF-8 CSV를 작성한다. AI-Hub XML의 행동 속성 및 프레임 번호와 실제 영상의 대응 관계를 확인한 다음 변환하거나 사람이 검토한 구간을 기록한다. 현재 받은 폴더에 XML이 없어 자동 변환기는 제공하지 않는다.

```csv
video,split,group,label,start,stop,anchor,x1,y1,x2,y2,context_start
```

- `video`는 데이터 루트 기준 상대 경로다. `group`은 같은 촬영·연기의 여러 카메라가 공유하는 ID다. 같은 그룹은 한 분할에만 포함한다. AI-Hub Training을 그룹별로 `train`과 `val`로 나누고, 별도로 보존한 영상은 `test`로 둔다.
- `label`은 `normal,fall,vandalism,arson,smoking,abandon,theft,assault` 중 하나다. 정상 구간은 영상 내용으로 확인한다.
- `start`는 라벨이 적용되는 첫 프레임이고 `stop`은 마지막 다음 프레임이다. 둘 다 디코딩된 영상의 0부터 시작하는 프레임 번호다. `anchor`와 상자 네 좌표는 이 구간에서 대상 인물을 지정한다. 상자 좌표는 원본 영상의 픽셀 단위다. 한 영상에서 라벨 구간을 겹치게 입력하면 오류를 낸다.
- 선택 항목 `context_start`는 30프레임 입력의 시작이 허용되는 가장 앞 프레임이다. 짧은 사건에 직전 행동을 문맥으로 넣을 때 사용한다. 결과 라벨은 마지막 프레임이 해당 구간 안에 있는 시퀀스에만 붙는다. 비우면 구간 내부만 사용한다.
- 모든 분할에 8개 클래스가 들어가야 한다. 입력 영상의 FPS를 통일한다. 3 FPS 원본에서는 30프레임이 약 10초다. 학습 후 웹에서 비교할 때 같은 FPS, 포즈 가중치, 480 해상도, 사람 검출 40%, 검출 간격 1을 사용한다. 다른 FPS의 실시간 영상에 적용하려면 별도 검증이 필요하다.

## Windows RTX 3070

TensorFlow CUDA 학습은 WSL2의 Linux 터미널에서 실행한다. NVIDIA 드라이버를 준비한 다음 WSL에서 `nvidia-smi`와 GPU 조회를 확인한다. 프로젝트와 데이터 경로가 Windows 드라이브에 있으면 예를 들어 `/mnt/d/data`처럼 접근한다. [TensorFlow 공식 Windows WSL2 설치](https://www.tensorflow.org/install/pip#windows-wsl2)

```bash
python3.12 -m venv .venv-web
python3.12 -m venv .venv-tf
.venv-web/bin/python -m pip install -r backend/requirements-worker.txt
.venv-tf/bin/python -m pip install 'tensorflow[and-cuda]==2.21.0'
.venv-tf/bin/python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'

.venv-web/bin/python train/behavior_tf/prepare.py --pose yolo11n-pose.pt --data-root /mnt/d/aihub --labels /mnt/d/labels.csv --output .local-training/dataset
.venv-tf/bin/python train/behavior_tf/fit.py --dataset .local-training/dataset --output .local-training/run-01 --require-gpu
.venv-web/bin/python train/behavior_tf/export_pth.py .local-training/run-01/weights.npz .local-training/run-01/kpt_behavior_tf.pth
```

## macOS CPU

Mac에서 설치한 별도 TensorFlow 환경으로 학습 검증을 실행했다. 기존 워커 가상 환경을 사용해 포즈를 추출하고 가중치를 변환한다. 새 환경이 필요하면 프로젝트의 [macOS 백엔드 가이드](../../backend/README.md#macos-terminal)를 참고한다. 소규모 데이터는 CPU로 학습할 수 있으며, 원본이 많으면 RTX 3070에서 포즈 추출과 학습을 실행할 수 있다.

```bash
python3.12 -m venv .venv-tf
.venv-tf/bin/python -m pip install 'tensorflow==2.21.0'

.venv-web/bin/python train/behavior_tf/prepare.py --pose yolo11n-pose.pt --data-root "/Volumes/데이터 드라이브/AI-Hub" --labels "/Volumes/데이터 드라이브/labels.csv" --output .local-training/dataset
.venv-tf/bin/python train/behavior_tf/fit.py --dataset .local-training/dataset --output .local-training/run-01
.venv-web/bin/python train/behavior_tf/export_pth.py .local-training/run-01/weights.npz .local-training/run-01/kpt_behavior_tf.pth
```

## 결과 확인

- `metrics.json`에서 보존한 시험셋의 클래스별 정밀도·재현율과 혼동행렬을 본다. 절도와 흡연의 혼동, 정상 상품 취급에서 발생하는 오탐을 확인한다. 이 수치는 시퀀스 단위이며 사건 단위의 오탐 횟수나 감지 지연은 웹에서 영상으로 추가 확인한다.
- 변환기는 TensorFlow와 PyTorch의 출력값을 대조한 후 웹 첨부용 `.pth`를 저장한다. **모델 설정 → 행동 모델 파일 첨부 → 다시 분석**에서 이 파일을 사용한다.
- 라벨이 없는 기존 영상은 `prepare.py --pose yolo11n-pose.pt --video "영상 전체 경로"`로 포즈 입력을 확인할 수 있다.
