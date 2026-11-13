import time
import os
import threading
import cv2
import numpy as np
import requests
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from core.detector import HumanDetector
from core.tracker import HumanTracker
from core.labels import label_en

_COLOR_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# COCO 17 관절 연결 쌍 (skeleton)
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),    # 얼굴
    (5, 6),                              # 양 어깨
    (5, 7), (7, 9),                      # 왼쪽 팔
    (6, 8), (8, 10),                     # 오른쪽 팔
    (5, 11), (6, 12),                    # 몸통 측면
    (11, 12),                            # 골반
    (11, 13), (13, 15),                  # 왼쪽 다리
    (12, 14), (14, 16),                  # 오른쪽 다리
]

_KPT_CONF_THRESHOLD = 0.3

# FastAPI 서버 연동 상수
_API_URL = "http://127.0.0.1:8000/api/alert"
_ALERT_COOLDOWN_SEC = 10
_EVENT_IMAGE_DIR = "event_captures"

# 이상행동 녹화 트리거 최소 신뢰도 (스무딩된 행동 확률 기준)
_RECORD_CONF_THR = 0.6

def _track_color(track_id) -> tuple:
    return _COLOR_PALETTE[int(track_id) % len(_COLOR_PALETTE)]


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


def _match_keypoints(tracks: list, detections: list) -> dict:
    """track_id → keypoints(np.ndarray|None) 매핑. IoU 최대 detection 선택."""
    result = {}
    for t in tracks:
        best_iou, best_kpts = 0.0, None
        for d in detections:
            iou = _iou(t["bbox"], d["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_kpts = d.get("keypoints")
        # IoU 0.1 미만이면 박스가 거의 겹치지 않으므로 잘못된 매핑으로 간주
        result[t["track_id"]] = best_kpts if best_iou > 0.1 else None
    return result


def _summarize_behavior(behavior_map: dict) -> tuple:
    """인물별 행동 결과를 화면 전체 요약 (class_id, conf) 로 압축한다.

    - 결과 없음            → (-1, 0.0)  분석 중
    - 이상행동 있음        → 신뢰도가 가장 높은 이상행동 (class_id != 0)
    - 전원 정상            → (0, 최대 정상 신뢰도)
    """
    if not behavior_map:
        return -1, 0.0
    best_abn = None       # (conf, class_id)
    best_normal_conf = 0.0
    for cid, conf in behavior_map.values():
        if cid != 0:
            if best_abn is None or conf > best_abn[0]:
                best_abn = (conf, cid)
        else:
            best_normal_conf = max(best_normal_conf, conf)
    if best_abn is not None:
        return best_abn[1], best_abn[0]
    return 0, best_normal_conf


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_updated = pyqtSignal(int, float)        # (감지 인원, fps)
    # 화면 전체에서 가장 두드러진 이상행동 요약: (class_id, confidence)
    #   class_id = -1 분석 중 / 0 정상 / 그 외 이상행동
    behavior_ready = pyqtSignal(int, float)
    # 영상 파일 재생 진행률: (현재 프레임, 전체 프레임). 파일 소스에서만 emit
    progress_updated = pyqtSignal(int, int)
    error_occurred = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, source, model_path: str, conf: float, max_age: int,
                 kpt_model_path: str = "", imgsz: int = 640,
                 detect_interval: int = 1, stand_model_path: str = "",
                 source_name: str = ""):
        super().__init__()
        self.source = source
        self.model_path = model_path  # 항상 YOLO pose 모델
        self.conf = conf
        self.max_age = max_age
        self.kpt_model_path = kpt_model_path  # 키포인트 행동 분류 모델 경로 (선택)
        self.stand_model_path = stand_model_path  # 물품 가판대 감지 모델 경로 (선택)
        self.source_name = source_name        # 녹화 기록에 남길 소스 이름
        self.imgsz = imgsz                    # 추론 해상도 (경량화)
        # 영상 파일 탐색(seek): 0.0~1.0 비율. run 루프가 적용 후 None 으로 초기화
        self._seek_fraction = None
        self.record_events = True             # 이상행동 발생 시 자동 녹화 여부
        # N프레임마다 YOLO 검출 실행, 사이 프레임은 직전 결과 재사용
        # (Raspberry Pi 등 CPU 환경에서 다중 카메라 구동 시 부하 분산)
        self.detect_interval = max(1, int(detect_interval))
        self._running = False
        self._paused = False
        self.show_bbox = True
        self.show_track_id = True
        self.show_keypoints = True
        self.show_kpt_behavior = True  # 키포인트 행동 분류 결과 표시 여부
        self.show_stand = True         # 물품 가판대(class 0) 표시 여부
        self.show_pos = True           # 결제기(class 1) 표시 여부
        
        # FastAPI 알림 연동: 쿨타임 관리 + 비동기 전송/서킷 브레이커
        self.last_alert_time = 0
        self._alert_enabled = True      # 연속 실패 시 자동 비활성화
        self._alert_fail_count = 0

        # 이벤트 이미지 저장 디렉토리 생성
        os.makedirs(_EVENT_IMAGE_DIR, exist_ok=True)

    @property
    def _is_live(self) -> bool:
        """RTSP/웹캠처럼 끊기면 재연결이 필요한 라이브 소스인지 여부."""
        if isinstance(self.source, int):
            return True
        s = str(self.source).lower()
        return s.startswith(("rtsp://", "rtmp://"))

    @property
    def is_file(self) -> bool:
        """탐색(seek)·진행률이 가능한 동영상 파일 소스인지 여부."""
        if isinstance(self.source, int):
            return False
        s = str(self.source).lower()
        return not s.startswith(("rtsp://", "rtmp://", "http://", "https://"))

    def seek_to_fraction(self, frac: float):
        """재생 위치를 0.0~1.0 비율로 이동 요청 (파일 소스 전용)."""
        self._seek_fraction = max(0.0, min(1.0, float(frac)))

    def _open_capture(self):
        """소스에 맞는 VideoCapture를 열고 반환한다."""
        if isinstance(self.source, int):
            # 웹캠 — 플랫폼별 최적 백엔드 사용, 실패 시 기본 백엔드 재시도
            #   Windows(개발 PC): DSHOW가 MSMF보다 열기 빠르고 다중 카메라 충돌이 적다
            #   Raspberry Pi OS(운영): V4L2 명시 — /dev/video{N} 직접 사용
            import sys
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
            cap = cv2.VideoCapture(self.source, backend)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.source)
        elif self._is_live:
            # RTSP: TCP 전송 강제 + 연결 타임아웃 5초
            # UDP 기본값은 패킷 손실 시 30초 타임아웃 경고를 유발하므로 TCP로 고정
            _prev_opts = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000"
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if _prev_opts:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _prev_opts
            else:
                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.source)
        return cap

    def run(self):
        self._running = True
        cap = None
        recorder = None
        try:
            detector = HumanDetector(self.model_path, self.conf, imgsz=self.imgsz)
            tracker = HumanTracker(self.max_age)
            # 키포인트 행동 분류: YOLO pose 모델 + 별도 .pth 지정 시 활성
            kpt_buf = None
            if self.kpt_model_path and detector.is_pose:
                from core.kpt_behavior_classifier import TrackBehaviorBuffer
                kpt_buf = TrackBehaviorBuffer(self.kpt_model_path)
            # 물품 가판대 감지기 (모델 지정 시 활성, show_stand 로 표시 토글)
            obj_detector = None
            theft_monitor = None
            if self.stand_model_path:
                from core.detector import ObjectDetector
                # 가판대/결제기는 정적 구조물 — 도난 규칙의 영역 캐싱을 위해 임계값을 낮춤
                obj_detector = ObjectDetector(self.stand_model_path,
                                              conf_threshold=0.25, imgsz=self.imgsz)
                # 결제기(pos) 클래스가 있으면 규칙 기반 도난 감지 활성
                if 1 in getattr(obj_detector, "names", {}):
                    from core.theft_monitor import TheftMonitor
                    theft_monitor = TheftMonitor()
            cap = self._open_capture()
            if not cap.isOpened():
                self.error_occurred.emit(f"영상 소스를 열 수 없습니다: {self.source}")
                return

            src_fps = cap.get(cv2.CAP_PROP_FPS)
            # 30 fps 상한: 처리 속도가 원본 fps를 초과하면 Qt 이벤트 큐가 과부하됨
            # RTSP는 fps가 0으로 반환될 수 있으므로 30fps로 고정
            target_interval = 1.0 / min(src_fps if src_fps > 0 else 30.0, 30.0)

            # 파일 소스 전체 프레임 수 (재생바용)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else 0

            # 이상행동 이벤트 녹화기 (행동 분류 모델이 있을 때만 의미 있음)
            recorder = None
            if self.record_events and self.kpt_model_path:
                from core.recorder import EventRecorder
                rec_fps = min(src_fps if src_fps > 0 else 20.0, 30.0)
                recorder = EventRecorder(self.source_name, rec_fps)

            _reconnect_delay = 2.0   # RTSP 끊김 후 재연결 대기 (초)
            _max_reconnects = 5      # 최대 재연결 시도 횟수
            _reconnect_count = 0

            prev_time = time.time()
            frame_idx = 0
            # detect_interval > 1 일 때 사이 프레임에 재사용할 직전 검출 결과
            tracks, kpt_map, behavior_map, objects = [], {}, {}, []
            while self._running:
                if self._paused:
                    # 일시정지 중에도 탐색(seek)은 적용해 미리보기 프레임을 갱신
                    if self._seek_fraction is not None and total_frames > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,
                                int(self._seek_fraction * total_frames))
                        self._seek_fraction = None
                        ret, frame = cap.read()
                        if ret:
                            self.frame_ready.emit(self._to_qimage(frame))
                            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            self.progress_updated.emit(cur, total_frames)
                    time.sleep(0.05)
                    continue

                # 재생 중 탐색 요청 적용
                if self._seek_fraction is not None and total_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,
                            int(self._seek_fraction * total_frames))
                    self._seek_fraction = None

                loop_start = time.time()
                ret, frame = cap.read()

                if not ret:
                    if not self._is_live:
                        break  # 파일 재생 종료

                    # RTSP 끊김 — 재연결 시도
                    _reconnect_count += 1
                    if _reconnect_count > _max_reconnects:
                        self.error_occurred.emit(
                            f"RTSP 스트림 재연결 실패 ({_max_reconnects}회 시도): {self.source}"
                        )
                        break
                    cap.release()
                    time.sleep(_reconnect_delay)
                    cap = self._open_capture()
                    if not cap.isOpened():
                        continue
                    _reconnect_count = 0
                    continue

                _reconnect_count = 0  # 정상 수신이면 카운터 초기화

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                # detect_interval 마다만 YOLO 검출 — 사이 프레임은 직전 결과 재사용
                if frame_idx % self.detect_interval == 0:
                    detector.set_confidence(self.conf)
                    detections = detector.detect(frame)
                    tracks = tracker.update(detections, frame)

                    kpt_map = {}
                    if detector.is_pose:
                        kpt_map = _match_keypoints(tracks, detections)

                    # 키포인트 행동 분류 (pose 모델 + kpt_model 지정 시)
                    behavior_map = {}
                    if kpt_buf is not None and self.show_kpt_behavior:
                        behavior_map = kpt_buf.update(tracks, kpt_map)

                    # 물품 가판대/결제기 감지 (모델 지정 시)
                    if obj_detector is not None:
                        objects = obj_detector.detect(frame)
                    else:
                        objects = []

                    # 규칙 기반 도난 감지 — 손이 가판대→(미경유 POS)면 도난
                    if theft_monitor is not None:
                        theft = theft_monitor.update(tracks, kpt_map, objects)
                        for tid in theft:
                            # LSTM 이 정상/불확실일 때만 도난으로 덮어씀
                            # (전도·파손·폭행 등 특정 이상행동은 LSTM 결과를 우선)
                            cur = behavior_map.get(tid, (-1, 0.0))
                            if cur[0] in (-1, 0):
                                behavior_map[tid] = (6, 0.9)
                frame_idx += 1

                # 가판대(0)/결제기(1) 표시는 각각의 토글로 필터 (감지 자체는 도난규칙 위해 항상 수행)
                shown_objects = [o for o in objects
                                 if (o.get("cls") == 0 and self.show_stand)
                                 or (o.get("cls") == 1 and self.show_pos)]
                self._draw(frame, tracks,
                           kpt_map if self.show_keypoints else {},
                           behavior_map,
                           shown_objects)
                self.stats_updated.emit(len(tracks), fps)
                cid, bconf = _summarize_behavior(behavior_map)
                self.behavior_ready.emit(cid, bconf)
                
                # ===== FastAPI 알림 연동 (비동기 — 처리 루프를 막지 않음) =====
                current_time = time.time()
                anomaly_score = float(bconf)
                if (self._alert_enabled and len(tracks) > 0 and cid != 0
                        and anomaly_score >= 0.90
                        and current_time - self.last_alert_time > _ALERT_COOLDOWN_SEC):
                    self.last_alert_time = current_time

                    # 1) 사진 저장 (로컬, 빠름)
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = f"evt_{timestamp_str}.jpg"
                    img_path = os.path.join(_EVENT_IMAGE_DIR, img_name)
                    cv2.imwrite(img_path, frame)

                    # 2) JSON 페이로드
                    anomaly_type = label_en(cid) if cid != -1 else "정상"
                    payload = {
                        "event_id": f"evt_{timestamp_str}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "anomaly_type": anomaly_type,
                        "image_path": os.path.abspath(img_path),
                        "anomaly_score": float(bconf),
                    }

                    # 3) POST 는 데몬 스레드로 비동기 전송 (네트워크 지연이 영상 처리를 막지 않음)
                    threading.Thread(target=self._post_alert, args=(payload,),
                                     daemon=True).start()
                # ===== FastAPI 알림 연동 끝 =====

                # ===== 이상행동 구간 녹화 =====
                if recorder is not None:
                    is_abn = (len(tracks) > 0 and cid not in (-1, 0)
                              and bconf >= _RECORD_CONF_THR)
                    recorder.push(frame, is_abn, cid if is_abn else 0, bconf)

                # 파일 재생 진행률 (재생바 갱신)
                if total_frames > 0:
                    cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.progress_updated.emit(cur, total_frames)

                self.frame_ready.emit(self._to_qimage(frame))

                elapsed = time.time() - loop_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

        except Exception:
            import traceback
            self.error_occurred.emit(traceback.format_exc())
        finally:
            if recorder is not None:
                recorder.close()
            if cap is not None:
                cap.release()
            self.finished_signal.emit()

    # ------------------------------------------------------------------ 그리기
    def _draw(self, frame: np.ndarray, tracks: list, kpt_map: dict,
              behavior_map: dict = None, stands: list = None):
        if behavior_map is None:
            behavior_map = {}
        # 물품 가판대 박스 먼저 그려 사람 박스가 위에 오도록 함
        if stands:
            self._draw_stands(frame, stands)
        for t in tracks:
            tid = t["track_id"]
            color = _track_color(tid)
            x1, y1, x2, y2 = t["bbox"]

            # 이상행동 감지 시 빨간 테두리 색상으로 덮어씀
            beh = behavior_map.get(tid)
            if beh is not None and beh[0] != 0:
                color = (0, 0, 255)   # BGR 빨강

            if self.show_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 레이블 조합: "ID:1 | FALL 92%"
                parts = []
                if self.show_track_id:
                    parts.append(f"ID:{tid}")
                if beh is not None:
                    cid, bconf = beh
                    parts.append(f"{label_en(cid)} {bconf*100:.0f}%")

                if parts:
                    label = " | ".join(parts)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            kpts = kpt_map.get(tid)
            if kpts is not None:
                self._draw_pose(frame, kpts, color)

    # 클래스별 색/라벨 (BGR): 0 가판대=주황, 1 결제기=파랑
    _OBJ_STYLE = {
        0: ((0, 165, 255), "STAND"),
        1: ((255, 160, 0), "POS"),
    }

    def _draw_stands(self, frame: np.ndarray, objects: list):
        """가판대/결제기를 박스가 아닌 반투명 색칠 영역으로 표시한다."""
        if not objects:
            return
        alpha = 0.35
        h, w = frame.shape[:2]

        def _clip(box):
            x1, y1, x2, y2 = box
            return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        overlay = frame.copy()
        for o in objects:
            color, _ = self._OBJ_STYLE.get(o.get("cls", 0), self._OBJ_STYLE[0])
            x1, y1, x2, y2 = _clip(o["bbox"])
            if x2 > x1 and y2 > y1:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        for o in objects:
            color, name = self._OBJ_STYLE.get(o.get("cls", 0), self._OBJ_STYLE[0])
            x1, y1, x2, y2 = _clip(o["bbox"])
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, name, (x1 + 3, y1 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _draw_pose(self, frame: np.ndarray, kpts: np.ndarray, color: tuple):
        """17개 관절 점과 skeleton 연결선을 그린다."""
        h, w = frame.shape[:2]
        points = []
        for kx, ky, kc in kpts:
            if kc >= _KPT_CONF_THRESHOLD and 0 <= int(kx) < w and 0 <= int(ky) < h:
                points.append((int(kx), int(ky)))
            else:
                points.append(None)

        # 연결선
        for a, b in _SKELETON:
            if points[a] and points[b]:
                cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)

        # 관절 점
        for pt in points:
            if pt:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ 유틸
    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # .copy() 필수: 없으면 QImage가 numpy 버퍼 포인터를 보유한 채 GC 발생 → dangling pointer crash (exit code 5)
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def _post_alert(self, payload: dict):
        """FastAPI 알림 POST (데몬 스레드에서 실행). 연속 실패 시 알림 비활성화."""
        try:
            requests.post(_API_URL, json=payload, timeout=2)
            self._alert_fail_count = 0
        except requests.exceptions.RequestException:
            self._alert_fail_count += 1
            if self._alert_fail_count >= 3:
                self._alert_enabled = False
                print("FastAPI 알림 서버에 연결할 수 없어 알림 전송을 비활성화합니다.")

    def set_conf(self, value: float):
        self.conf = value

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)
