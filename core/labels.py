"""이상행동 8종 클래스 라벨 (한글 / 화면 표시용 영문).

NIA read_data_from_nia.py 의 event_type 매핑과 동일:
    0 정상 / 1 전도 / 2 파손 / 3 방화 / 4 흡연 / 5 유기 / 6 절도 / 7 폭행
"""

BEHAVIOR_LABELS = [
    ("정상", "NORMAL"),
    ("전도(쓰러짐)", "FALL"),
    ("파손", "VANDALISM"),
    ("방화", "ARSON"),
    ("흡연", "SMOKING"),
    ("유기", "ABANDON"),
    ("절도", "THEFT"),
    ("폭행", "ASSAULT"),
]


def label_kr(class_id: int) -> str:
    return BEHAVIOR_LABELS[class_id][0]


def label_en(class_id: int) -> str:
    return BEHAVIOR_LABELS[class_id][1]
