CSV 형식 (헤더 없음):
    frame1_path,frame2_path,frame3_path,label

경로:
    - config.py의 DATA_ROOT 기준 상대경로, 또는 절대경로
    - 슬래시/역슬래시 모두 허용

레이블:
    0 = 정상
    1 = 전도 (쓰러짐)
    2 = 파손
    3 = 방화
    4 = 흡연
    5 = 유기
    6 = 절도
    7 = 폭행

예시 (train.csv):
    videos/C_3_7_1/frame_000001.png,videos/C_3_7_1/frame_000002.png,videos/C_3_7_1/frame_000003.png,0
    videos/C_3_7_1/frame_000004.png,videos/C_3_7_1/frame_000005.png,videos/C_3_7_1/frame_000006.png,1
    ...

파일 목록:
    split/train.csv   ← 학습 데이터
    split/val.csv     ← 검증 데이터

source 데이터 변환:
    기존 source/1. 이상행동 분류/ConvLSTM/split/test.csv 형식과 동일하므로
    해당 파일을 train.csv/val.csv 로 복사하면 바로 사용 가능.
