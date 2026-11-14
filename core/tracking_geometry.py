"""경계 상자의 겹침 정도로 관절 검출 결과를 추적 대상과 연결한다."""

def _iou(a: list, b: list) -> float:
    """두 bbox [x1,y1,x2,y2] 의 IoU."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def match_keypoints(tracks: list, detections: list) -> dict:
    """track_id → keypoints(np.ndarray|None) 매핑. IoU 최대 detection 선택."""
    result = {}
    for track in tracks:
        if "keypoints" in track:
            # DeepSORT가 실제로 연결한 검출의 관절만 사용한다. 미관측 트랙은 None이다.
            result[track["track_id"]] = track["keypoints"]
            continue
        best_iou, best_points = 0.0, None
        for detection in detections:
            overlap = _iou(track["bbox"], detection["bbox"])
            if overlap > best_iou:
                best_iou, best_points = overlap, detection.get("keypoints")
        # 원본 GUI와 같은 연결 규칙을 사용해 행동 모델의 입력을 유지한다.
        result[track["track_id"]] = best_points if best_iou > 0.1 else None
    return result
