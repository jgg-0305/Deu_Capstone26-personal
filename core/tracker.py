from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import torch


class HumanTracker:
    """DeepSORT 래퍼. YOLO 감지 결과를 받아 트랙 ID를 부여.

    경량화/반응성:
      - n_init=2  : 2프레임만 관측되면 트랙 확정 (기본 3 → 박스 표시 지연 감소)
      - embedder_gpu : CUDA 가용 시 외형 임베더를 GPU 에서 실행
    """

    def __init__(self, max_age: int = 30, n_init: int = 2):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder_gpu=torch.cuda.is_available(),
        )

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """
        detections: HumanDetector.detect() 결과
        검출 결과 사전의 목록을 반환한다.
          { 'track_id': int, 'bbox': [x1,y1,x2,y2] }
        """
        # 추적기의 입력 형식은 좌상단 좌표, 너비, 높이, 신뢰도와 클래스 번호다.
        raw = []
        supplementary = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            raw.append(([x1, y1, w, h], d["conf"], 0))
            supplementary.append(d.get("keypoints"))

        # 검출이 잠시 없어도 확정된 트랙은 최대 소실 프레임까지 유지한다.
        tracks = self.tracker.update_tracks(raw, frame=frame, others=supplementary)
        results = []
        for t in tracks:
            # 연속 프레임에서 충분히 관측되지 않은 잠정 트랙은 제외
            if not t.is_confirmed():
                continue
            observed = t.time_since_update == 0
            # 관절과 정규화 상자는 같은 검출에서 가져와 다른 사람의 관절이 섞이지 않게 한다.
            ltrb = t.to_ltrb(orig=True) if observed else t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            results.append({
                # DeepSORT가 track_id를 문자열로 반환하므로 명시적 변환 필요 (f-string 외 % 연산 오류 방지)
                "track_id": int(t.track_id),
                "bbox": [x1, y1, x2, y2],
                "observed": observed,
                "missing_frames": t.time_since_update,
                "keypoints": t.get_det_supplementary() if observed else None,
            })
        return results
