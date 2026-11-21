# 전체 행동 분류 재학습 파이프라인

## 손 주변 영상과 관절을 함께 학습

`prepare_hand.py`는 AI-Hub 71550 공식 Training·Validation의 영상과 XML을 연결한다.
추적 인물별 3 FPS 관절에 양쪽 손목 주변 RGB 영역, 얼굴·어깨·팔꿈치·골반과 손목의 거리,
손목 궤적을 기록한다. 30시점 입력에서 손 주변 영상과 이동 특징의 시퀀스를 함께 학습한다.
이 데이터의 행동 XML을 사용해 집기·옮기기 동작의 단서를 학습하며 상품 상자나 실제 결제 정보는 제공되지 않는다.

`train_hand.py`는 기존 관절 모델을 고정하고 새 영상·움직임 분기를 학습한다.
공식 Training에서 촬영 묶음별로 학습과 모델 선택 자료를 나누고 공식 Validation으로 시험한다.
이미 알려진 주머니 오분류 촬영은 별도의 회귀 검사로 분리한다.
모델 비교에는 동일한 YOLO 검출과 DeepSORT 추적을 사용한다. XML의 시작·종료 시점에서 YOLO 인물 상자 겹침,
추적 인물 연결 실패, 관절 수집 지연, 확정 오분류를 구분해 기록한다.

```powershell
python train/behavior_repair/prepare_hand.py --data 'D:/AIHub/01-1.정식개방데이터' --out 'D:/runs/hand-fusion' --train-per-class 16 --val-per-class 4 --test-per-class 8
python train/behavior_repair/train_hand.py --out 'D:/runs/hand-fusion'
```

같은 데이터·설정에서 추출을 다시 실행하면 완료된 영상 캐시를 재사용한다. 데이터나 추출 규격을 바꿀 때는 새 출력 폴더를 쓴다.
`evaluation.json`에는 모델별 8개 클래스 점수, 사건 프레임 집계, 주머니 회귀 검사, 가중치 해시가 기록된다.
`hand_fusion_3fps.pth`는 수정된 웹의 행동 모델 업로드에서 선택할 수 있다.
평가의 정상 구간은 XML 사건 전 대체 구간이며, 현장 정상 구매에 대한 독립 검증은 별도다.

## 상품 궤적 실험

`train_article.py`는 AI-Hub XML의 `object_article` 상자로 손목 주변 상품 검출기를 학습한다.
흡연과 다른 행동에서 추출한 손 주변 영상도 학습에 넣어 손 자체를 상품으로 검출하는 오탐을 줄인다.
`ArticleTracker`는 검출 상자와 손의 거리, 상대 이동, 색상 정보를 이용해 동일한 손에 상품 후보를 연결한다.
검출이 끊기면 먼저 두 시점에서 상품 상자의 영상 특징을 앞뒤 방향으로 따라간다. 그 이후에도 기존
상품의 외관이 일치하면 마지막 실제 검출에서 1.34초 이내에는 가림 판정을 보류한다. 추정 상자는 검출
횟수에 포함하지 않으며, 다시 검출될 때 예상 위치와 색을 확인한다. 양손이 가까우면 위치와 외관이 일치하는
기존 상품 연결을 우선한다. 손목이 이미 골반 부근에 있는 경우에는 충분히 반복 검출된 상품 상자 중심의
골반 접근도 확인한다. 마지막 검출 뒤 손이 골반 부근에 머물고 상품이 가려지면 도난 의심의 보조 증거를 생성한다.
거리가 짧은 동작에서는 반복 검출된 상품이 골반에 가까워지고, 마지막 두 상품 검출에서도 골반 근처에
있었던 경우에 한해 짧은 이동을 보조 증거로 사용한다.
웹 워커는 기본 손목 영역과 별도로 손목 중심의 더 좁은 영역에서도 상품을 검출한다. 별도 추적에서
같은 손에 반복 연결된 상품이 짧게 사라졌다가 골반 가까이에서 더 작은 상자로 다시 잡히면, 손과 상자의
위치 및 외관을 대조해 이전 궤적을 잇는다. 이어진 상품의 골반 접근과 재가림을 모두 관측한 경우에만
도난 의심 규칙에 추가 증거로 전달한다.
RTSP에서는 두 크기의 좁은 손목 영역을 한 번에 추론하고, 동일 손의 신뢰할 수 있는 연속 궤적을 우선한다.
흡연 사건은 같은 사람의 서로 다른 시점에서 손이 얼굴 가까이 관측된 경우에 기록한다.
기존 행동 모델 및 사건 임계값은 같은 값을 사용한다.
다만 이 보조 신호가 남아 있는 동안에도 손이 얼굴 가까이에서 관측되고 행동 모델이 흡연을 분류하면
기존 흡연 결과를 유지한다.
가림 신호가 끝나도 이전에 가려진 흡연 추론 결과는 새 사건으로 등록하지 않는다.
해당 인물이 새 추론 창에서 다시 흡연으로 분류되고 손이 얼굴 근처에서 관측되면 흡연 판정을 다시 표시한다.
정답의 상품 상자와 소멸 시점은 시험 결과 측정에 사용한다.

```powershell
python train/behavior_repair/train_article.py --data 'D:/AIHub/01-1.정식개방데이터' --out 'D:/runs/article'
python train/behavior_repair/evaluate_article.py --data 'D:/AIHub/01-1.정식개방데이터' --out 'D:/runs/article' --hand-run 'D:/runs/hand-fusion' --pose-cache-only
python train/behavior_repair/evaluate_article.py --data 'D:/AIHub/01-1.정식개방데이터' --out 'D:/runs/article' --hand-run 'D:/runs/hand-fusion'
```

고해상도 상품 후보는 새 출력 폴더에서 `--size 256 --initialize 'D:/runs/article/article.pt'`로 학습한다.
같은 YOLO 인물·관절 검출을 고정한 비교에서는 평가에 `--pose-cache 'D:/runs/article/test_pose'`를 지정한다.
상품 탐지 평가는 신뢰도 0.35 이상, 정답 상자와 IoU 0.5 이상을 기준으로 하고,
흐름으로 연결한 시점 수와 실제 검출로 이어진 횟수는 별도 기록한다.

실험을 웹 워커에서 재현할 때는 서버 환경 변수 `CCTV_ARTICLE_MODEL`에 학습한 `article.pt`의 절대 경로를 넣고
워커를 시작한다. 이 값이 설정되어 있을 때만 상품 궤적을 관절 행동 분류에 결합한다.
가림은 영상에서 관측한 간접 증거이며, 결제나 주머니 내부는 별도로 검증할 수 없다.

정상·넘어짐·파손·방화·흡연·유기·절도·폭행의 오탐과 미탐을 함께 평가한다.
흡연이나 절도에만 유리하도록 임계값을 바꾸거나, 특정 클래스 예측을 다른 클래스로 강제 변경하지 않는다.

## 변경한 알고리즘

1. **정답 연결** — AI-Hub 71550 원본 MP4와 XML을 연결한다. 사건 시작/종료 인물 상자를 추적 인물과 연결하고, 겹치거나 뒤집힌 사건 구간은 제외 사유를 기록한다.
2. **촬영 그룹 분리** — 행동·시나리오·장소·날짜 기준으로 그룹을 구성한다. 다른 카메라의 같은 촬영이 학습과 평가에 동시에 들어가지 않도록 분리한다. 이미 살펴본 회귀 영상 그룹은 별도로 보관한다.
3. **운영과 같은 입력** — 17관절의 사람 상자 상대 좌표 34개를 사용한다. 3 FPS, 길이 30, stride 5, 관절 신뢰도 0.3, 포즈 해상도 480과 사람 검출 기준 0.4를 사용한다. 짧은 관절 누락은 0으로, 긴 누락은 새 시퀀스로 처리한다.
4. **포즈 입력 다양화** — 전체 YOLO11n 관절 자료에 YOLO11s 자료를 추가한다. 각 이상행동에서 동일한 수의 추가 영상 후보를 뽑고, 촬영 그룹을 순회해 특정 촬영에 치우치지 않게 한다. 기본 추가 후보는 학습 64개 × 7종, 검증 16개 × 7종이다. 정상은 해당 사건 전 구간에서 얻는 대체 정답이다.
5. **관절 증강** — 좌우 반전 시 COCO 좌우 관절을 함께 교환한다. 상자 오차를 모사한 확대/축소·이동, 관절 좌표 잡음, 관절 가림, 짧은 프레임 누락을 적용한다. 누락 관절은 0으로 유지한다.
6. **학습** — 앱과 같은 KeypointLSTM v2를 무작위 초기화한다. AdamW, gradient clipping, 역빈도 제곱근 클래스 가중치, label smoothing 0.03을 사용한다. 검증 정체 시 학습률을 낮춘다.
7. **모델 선택** — 각 포즈 도메인의 8개 클래스 Macro-F1을 계산하고 n/s 도메인을 동일한 비중으로 평균한다. 특정 행동의 우선 가중치는 없다. 12 epoch 동안 개선이 없으면 종료한다. 평가 자료는 모델 선택에 사용하지 않는다.
8. **중단·재개** — 원본 관절 캐시를 재사용한다. epoch마다 모델·optimizer·scheduler·난수 상태를 원자적으로 저장한다. 같은 출력 경로와 같은 데이터로 학습을 재실행하면 마지막 완료 epoch에서 이어간다. 데이터 해시가 달라지면 새 출력 폴더를 요구한다.
9. **평가** — 전체 n-pose 평가 시퀀스, 별도의 s-pose 원본 영상, 이미 알려진 회귀 영상을 구분한다. 클래스별 정밀도·재현율·F1·FP·FN·혼동행렬, 정상→이상 오탐, 이상→정상 미탐을 기록한다. 원본 평가는 EMA와 관절 수집 지연까지 포함하며 DB 사건 저장 규칙은 포함하지 않는다.
10. **웹 입력 시간 보정** — 배포용 체크포인트에 클래스 순서와 `sample_fps=3`을 넣는다. 수정된 웹 워커는 원본 타임스탬프를 사용해 고 FPS 영상도 같은 간격으로 샘플링한다. 기존 plain state_dict 체크포인트는 이전 동작을 유지한다.

## 실행

CUDA PyTorch, torchvision, ultralytics, deep-sort-realtime, OpenCV, NumPy가 필요하다.
프로젝트의 `backend/requirements.txt`도 설치한다. 현재 PC의 검증된 환경은 작업 폴더의 `work/gpu-env/Scripts/python.exe`다.

```powershell
python train/behavior_repair/run.py `
  --data-root 'D:/AIHub/238-2.실내(편의점, 매장) 사람 이상행동 데이터' `
  --prepared 'D:/experiments/aihub-prepared' `
  --output 'D:/experiments/behavior-repair-run1' `
  --previous-model 'train/1_behavior/kpt_behavior_local.pth' `
  --legacy-v1 'D:/models/old-v1.pth' `
  --pose-s 'D:/models/yolo11s-pose.pt'
```

`--legacy-v1`은 선택 사항이다. `--prepared`에 이전 전체 관절 자료가 있으면 재사용하며, 없으면 전체 원본에서 추출한다.
실제 실행 전 명령과 경로만 확인하려면 `--dry-run`을 추가한다.
실시간 진행 창을 함께 열려면 `--monitor`를 추가한다. 창을 닫아도 학습은 계속된다.
`--stage prepare`, `train`, `evaluate`, `diagnose`, `package`로 개별 단계를 실행할 수 있다.
중단 후 같은 명령을 다시 실행하면 완료된 단계와 관절 캐시를 재사용한다.
체크포인트의 모델·optimizer·scheduler·난수 상태 복원은 검사한다. 분리 실행과 연속 실행의 최종 가중치가 비트 단위로 완전히 같은 것은 보장하지 않는다.
모델 선택 기준을 바꾸는 실험은 별도 `--output` 경로를 사용한다.

## 산출물

- `progress.json` — 현재 단계와 진행 수량
- `mixed-dataset/preparation-report.json` — 추가 추출 영상, 유효 시퀀스 수, 제외 사유
- `trained/training.json`, `best-validation.json` — 모델 선정 근거와 클래스별 검증 수치
- `evaluation.json` — 전체 클래스 비교, 혼동행렬, 원본 영상 집계
- `raw-test-streams.json` — 원본 영상별 추적·분류 결과
- `kpt_behavior_repaired_3fps.pth` — 입력 규격이 포함된 웹 업로드용 모델
- `REPORT.md`, `pocket-before-after.mp4` — 전체 성능 보고서와 알려진 사례의 보조 회귀 영상

기존 기본 가중치와 서비스 DB는 자동 교체하지 않는다. 새 모델의 클래스별 평가를 확인한 뒤 웹에서 해당 가중치를 선택한다.
다른 PC에는 변경된 `core/kpt_behavior_classifier.py`, `core/tracker.py`, `backend/worker.py`도 함께 반영한다.

## YOLO11m 행동 모델 보정

`finetune_hand_medium.py`는 기존 YOLO11n 학습 캐시와 YOLO11m으로 추출한 학습·보정 캐시를 읽는다.
YOLO11m 학습에는 사건 전 장면과, 단일 인물 사건에서 정답 인물과 충분히 떨어진 관찰자의 관절 시퀀스를
정상 예시로 추가한다. 보정에는 학습에 참여하지 않은 촬영 그룹을 사용하고, 90% 기준의 클래스별 F1과
오탐으로 체크포인트를 선택한다. `--focus`는 파손과 유기의 학습 비중을 높인 후보를 별도 파일로 저장한다.

```powershell
python train/behavior_repair/finetune_hand_medium.py --base hand_fusion_3fps.pth --legacy-cache D:/runs/hand-n --medium-cache D:/runs/hand-m --focus
```

YOLO11m의 넓은 손목 영역에서 나온 도난 보조 신호는 마지막 상품 상자가 골반 가까이에 위치해야 한다.
짧은 이동으로 판정할 경우 골반 가까이에서 상품이 여러 시점에 관측돼야 한다.

## 해석 시 주의할 범위

정상 구간은 독립적으로 검수한 정상 구매 데이터가 아니라 사건 전 대체 정답이다.
추적·인물 연결 실패로 시퀀스를 얻지 못한 영상을 별도로 세어야 하며, 유효 시퀀스만의 높은 점수를 전체 영상 감지율로 해석하면 안 된다.
관절에는 담배나 상품 자체가 없으므로 모든 현장 행동의 의미를 구분할 수 있다고 보장하지 않는다.
보고서의 절도→흡연 사례는 전체 모델 선정 기준이 아니라 알려진 오류가 재발하는지 확인하는 보조 검사다.
