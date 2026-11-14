# AI 모델 포함 및 품질 검토

> 이 문서는 11월 13일 시점의 최초 감사 기록이다. 11월 14일의 실제 데모 추론 비교와 수정 결과는 [데모 회귀 검토](demo-regression-review.md)에 기록했다. 아래의 미실행·미수정 범위는 최초 감사 시점을 뜻한다.

검토 기준: **2026-11-13 KST (Asia/Seoul)**

- 개인 저장소 기준 커밋: [c7a7341](https://github.com/jgg-0305/Deu_Capstone26-personal/commit/c7a73419d48bfa861640838f833082487ddae6d1)
- 원본 저장소 기준 커밋: [9dfce34](https://github.com/peanutcorn/Deu_Capstone26/commit/9dfce34db59ca49442acfa284dd7eb02bc05b7ea)
- 원본 master/HEAD의 최신성은 검토 당일 git ls-remote로 별도 확인했다.
- 원본 모델, 애플리케이션 코드, 실행 설정은 수정하지 않았다.

## 1. 결론과 검증 범위

**가판대 이상행동 전용 학습 클래스는 확인되지 않았다.** `objects.pt`의 학습 클래스는 `stand`와 `pos`이고, 행동 분류 가중치의 클래스 6은 일반 `theft`다. 가판대에 특화된 도난 의심은 손목 이동과 POS 영역을 비교하는 `TheftMonitor` 규칙에서 생성한다.

**가판대·POS 감지와 이상행동 감시에 사용하는 필수 모델 세 개가 모두 포함되어 있다.**
작업 파일, 개인 저장소 main, 원본 upstream/master에서 각 모델의 Git blob이 일치한다.
개인 저장소에 옮기는 과정에서 모델이 누락되거나 변경된 흔적은 없다.

시스템은 단일 “도난 AI” 모델이 아니라 다음 구성의 조합이다.

1. YOLO11n Pose: 사람과 17개 관절 검출.
2. DeepSORT + MobileNetV2 임베더: 인물 추적.
3. KeypointLSTMv2: 30프레임 관절 시퀀스로 8개 행동 분류.
4. 커스텀 YOLO 객체 모델: 가판대와 POS 영역 검출.
5. TheftMonitor: 손목의 영역 진입과 시간 경과를 이용한 도난 의심 규칙.

모델 포함·구조·코드 연결은 검증했다. **실제 매장 영상의 정확도, 실제 결제 여부, 운영 환경 처리량은 이번 검토에서 검증하지 않았다.**

## 2. 모델 파일 무결성

| 파일 | 역할과 확인한 구조 | 크기(bytes) | 공통 Git blob |
|---|---|---:|---|
| [yolo11n-pose.pt](../../yolo11n-pose.pt) | PoseModel, person 1클래스, 17관절 × 3값 | 6,255,593 | f8e737050f8d2804e568e77627a538ae684b44d6 |
| [model/objects.pt](../../model/objects.pt) | DetectionModel, class 0=stand / class 1=pos | 5,473,924 | 4e4f216232664a0e0a16898498edf609d91efab1 |
| [train/1_behavior/output/kpt_behavior.pth](../../train/1_behavior/output/kpt_behavior.pth) | KeypointLSTMv2, 8개 행동 클래스 | 2,466,044 | 45cc03ce078147f6e29f9e65da374d31e61bf464 |

SHA-256과 기계 판독용 검증 범위는 [model-manifest.json](model-manifest.json)에 기록했다.

검사 방법:

- main:path와 upstream/master:path의 Git blob 및 작업 파일의 git hash-object 결과 비교.
- 작업 파일 전체의 SHA-256 계산.
- YOLO 파일은 ZIP 내부 data.pkl을 pickletools로 정적 분석하여 names, kpt_shape, 아키텍처와 저장 metadata 확인. 일반 pickle 역직렬화나 YOLO 모델 실행은 하지 않았다.
- 행동 모델은 torch.load(weights_only=True)로 읽고 기존 로더를 통해 strict state_dict 로딩 확인.
- 행동 모델에 영벡터 입력 (1, 30, 34)을 전달하여 유한한 (1, 8) 출력 확인. 이는 구조 점검이며 예측 성능 시험이 아니다.

## 3. 실행 경로 연결

| 항목 | 확인한 실행 코드 |
|---|---|
| 행동 모델 자동 선택 | [gui/main_window.py:50](../../gui/main_window.py#L50): train/1_behavior/output/kpt_behavior.pth 경로 확인 |
| 객체 모델 자동 선택 | [gui/main_window.py:57](../../gui/main_window.py#L57): objects.pt 우선, 구버전 stand.pt 대체 경로 |
| 작업 스레드에 모델 전달 | [gui/main_window.py:656](../../gui/main_window.py#L656) |
| 사람·추적·행동·객체 모델 생성 | [core/video_thread.py:201](../../core/video_thread.py#L201) |
| POS 클래스가 있으면 도난 규칙 활성화 | [core/video_thread.py:217](../../core/video_thread.py#L217) |
| LSTM v1/v2 자동 판별과 로딩 | [core/kpt_behavior_classifier.py:145](../../core/kpt_behavior_classifier.py#L145) |
| 8개 행동 클래스 | [core/labels.py:7](../../core/labels.py#L7): 정상, 전도, 파손, 방화, 흡연, 유기, 절도, 폭행 |

파일이 없을 때 일부 감지만 계속 수행하는 현재 동작은 데스크톱 편의 기능이다.
서버에서는 필수 모델 누락을 readiness 실패 또는 명시적인 기능 저하 상태로 표시해야 한다.
정상 가동으로 보이는 상태에서 감시 기능만 조용히 비활성화되면 운영자가 상태를 오해할 수 있다.

## 4. DeepSORT의 추가 가중치 의존성

[core/tracker.py:15](../../core/tracker.py#L15)는 DeepSort의 기본 embedder를 사용한다.
현재 설치된 deep-sort-realtime 1.3.2의 기본값은 mobilenet이며,
패키지 내부 embedder/weights/mobilenetv2_bottleneck_wts.pt를 불러온다.

- 현재 환경에서 파일 존재 확인: 9,084,095 bytes.
- SHA-256: 2f518e773d4402dde55f981ae3078a72ba95c3adccae1d55051a4be844d50197.
- 설치 패키지의 deepsort_tracker.py:38, embedder/embedder_pytorch.py:47–53에서 확인.
- 이 기본 경로는 실행 시 자동 다운로드가 아니라 패키지에 포함된 가중치 파일에 의존한다.
- 이 파일은 프로젝트 Git에 직접 포함된 세 모델과는 별개의 패키지 자산이다.

따라서 서버 배포에서는 프로젝트 모델 세 개뿐 아니라 이 임베더도 이미지 빌드 및 시작 점검 대상으로 삼아야 한다.
다른 YOLO 모델을 선택하거나 파일이 누락된 상황에서 일어날 수 있는 다운로드 의존성을 없애기 위해,
서버에는 허용 모델 목록과 고정 경로·해시를 설정하는 편이 좋다.

## 5. 우선 수정할 판정·전처리 문제

### 5.1 학습과 실행의 저신뢰 관절 표현이 다름 — 우선순위 높음

학습 코드 [kpt_common.py:53](../../train/1_behavior/kpt_common.py#L53)는 저신뢰 관절을 픽셀 (0, 0)으로 바꾼다.
이어 [encode_kpts:29](../../train/1_behavior/kpt_common.py#L29)에서 bbox 상대 정규화를 적용하므로 이 값이 음수가 될 수 있다.

실행 코드 [_encode:103](../../core/kpt_behavior_classifier.py#L103)는 저신뢰 관절을 최종 특징 (0, 0)으로 유지한다.

검토 시 합성 입력으로 재현했다.

| 조건 | 학습 특징 | 실행 특징 |
|---|---|---|
| bbox=[100,100,200,200], 신뢰도 미달 관절 | [-1.0, -1.0] | [0.0, 0.0] |

학습·실행이 하나의 전처리 함수를 사용하도록 통합하고 전처리 버전을 모델 manifest에 기록해야 한다.
이 문제를 수정한 뒤에는 기존 모델의 재평가가 필요하며, 데이터 특징이 바뀌므로 재학습도 검토해야 한다.

### 5.2 사라진 인물의 도난 의심이 반환되지 않음 — 우선순위 높음

[core/theft_monitor.py:100](../../core/theft_monitor.py#L100)는 사라진 트랙에 theft=True를 설정한다.
그러나 [반환 조건:110](../../core/theft_monitor.py#L110)은 현재 화면에 있는 tid만 포함한다.

손목 진입·이탈 후 LOST_FRAMES를 넘겨 인물을 제거한 합성 입력에서,
내부 theft=True 상태임에도 반환 결과가 빈 사전인 것을 재현했다.
상태 갱신과 이벤트 발행을 분리하고, 트랙 종료 이벤트를 별도로 발행해야 한다.

### 5.3 룰 판정과 모델 확률을 혼용함

[core/video_thread.py:324](../../core/video_thread.py#L324)는 도난 규칙이 성립하면 행동 결과를 (절도, 0.9)로 대체한다.
이 0.9는 모델이 산출한 보정된 확률이 아니다.

화면과 DB에는 최소한 아래를 구분해야 한다.

- 판정 출처: 행동 모델 / 도난 규칙.
- 모델 confidence와 룰 성립 여부.
- 적용한 모델·규칙 버전.
- 근거: 영역 진입·이탈 시각, POS 영역 경유 여부, 관측 종료 사유.
- 사용자 확인 상태: 미검토 / 확인 / 오탐.

현재 규칙은 실제 물품 소지나 결제 시스템의 거래를 확인하지 않으므로,
웹 화면에는 “도난 확정” 대신 “도난 의심”으로 표시해야 한다.

### 5.4 시간·프레임 기준이 실행 성능과 재생 조작에 영향을 받음

- [theft_monitor.py:53](../../core/theft_monitor.py#L53)는 벽시계 time.time()을 사용한다.
- [video_thread.py:297](../../core/video_thread.py#L297)는 detect_interval마다 행동 버퍼를 갱신한다.
- 학습 [kpt_common.py:63](../../train/1_behavior/kpt_common.py#L63)는 영상의 모든 프레임을 특징화한다.
- [video_thread.py:249](../../core/video_thread.py#L249), [263](../../core/video_thread.py#L263)의 seek는 캡처 위치만 변경한다.

느린 추론, 검출 간격 변경, 일시정지·탐색에 따라 4초 룰과 30프레임 시퀀스의 의미가 달라질 수 있다.
영상 timestamp 기반 판정, 고정 샘플링 FPS, seek 시 tracker/LSTM/룰 상태 초기화가 필요하다.

## 6. 학습·평가 근거와 누수 가능성

### 저장된 지표와 이번 측정의 구분

objects.pt에 다음 training metadata가 들어 있다.

| 지표 | 체크포인트에 저장된 값 |
|---|---:|
| Precision(B) | 0.58459 |
| Recall(B) | 0.54018 |
| mAP50(B) | 0.47681 |
| mAP50–95(B) | 0.38734 |

**위 수치는 이번 검토에서 독립적으로 재측정하지 않았다.**
검증 데이터 구성과 라벨 품질을 재현하지 못했으므로 실제 매장 운영 성능으로 일반화할 수 없다.
포즈 모델의 사전학습 지표 역시 이 프로젝트의 도난 감시 성능을 뜻하지 않는다.

현재 로컬에 없는 재현 자료:

- train/1_behavior/split/kpt_train.npz 및 kpt_val.npz.
- AI_성능평가표.docx.
- train/3_stand/output/stand/results.csv.
- train/3_stand/dataset/data.yaml.

평가·학습 스크립트는 존재하지만 모델의 실제 학습 실행 기록과 독립 테스트셋은 추가 확보가 필요하다.

### 데모 영상 학습과 평가의 중복 가능성

[demo_onset.py:61](../../train/1_behavior/demo_onset.py#L61)는 기본 학습 NPZ를 대상으로 하고,
[84](../../train/1_behavior/demo_onset.py#L84)에서 데모 윈도우를 기본 8회 반복하여 추가한다.
[evaluate_models.py:251](../../train/1_behavior/evaluate_models.py#L251)는 동일 DEMO_ONSET 목록을 평가에 사용한다.

최종 모델이 실제로 이 절차를 거쳤는지는 실행 기록이 없어 확정할 수 없다.
다만 이 절차로 학습한 모델의 동일 데모 결과는 독립적인 일반화 평가가 아니다.

### 객체 감지 데이터 분할의 중복 가능성

[auto_label.py:45](../../train/3_stand/auto_label.py#L45)의 stand/pos 소스에는 구매·반품 영상이 겹친다.
[collect_class_videos:103](../../train/3_stand/auto_label.py#L103)는 클래스마다 별도 seed로 분할하므로
동일 원본 영상이 한 클래스에서는 train, 다른 클래스에서는 val에 들어갈 가능성이 있다.
실제 데이터가 없어 이번 검토에서 특정 영상의 중복을 확정하지는 않았다.

클래스별 처리 전에 고유 원본 영상 목록을 만들고 전체 클래스에 공통 split을 적용해야 한다.
추출기 재실행 시 기존 NPZ에 다시 추가하는 [merge_npz:100](../../train/1_behavior/kpt_common.py#L100)도
실험 재현성과 중복 샘플 관리 측면에서 검토해야 한다.

## 7. 포트폴리오 개선 증거로 남길 항목

- 매장·촬영일·원본 영상 단위로 고정한 train/validation/test manifest.
- 전처리 수정 전후의 행동 클래스별 precision/recall/F1 및 confusion matrix.
- 가판대·POS 클래스별 AP와 수동 검수한 객체 라벨.
- 이벤트 단위 오탐/시간, 미탐률, 탐지 지연.
- 카메라 수·해상도·GPU/CPU별 FPS, P95 latency, 메모리 사용량.
- 영상 timestamp·프레임 건너뛰기·일시정지·트랙 유실에 대한 회귀 검증.
- DB model_artifacts와 분석 세션에 모델 해시·전처리·룰 버전을 연결한 판정 추적성.
- 가중치 라이선스와 데이터 사용 범위는 서비스 공개 전 별도 확인. 파일 내부 라이선스 문자열 확인만으로 전체 배포 적합성을 판정하지 않는다.

이번 검토에서는 라이브 카메라, 실제 카카오 발송, 외부 모델 다운로드를 수행하지 않았다.
