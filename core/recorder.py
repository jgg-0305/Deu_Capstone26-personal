# -*- coding: utf-8 -*-
"""이상행동 이벤트 구간 녹화 + JSON 기록 관리.

DB 대신 단일 JSON(recordings/recordings.json)으로 간단히 관리한다.
각 기록에는 발생 시각(timestamp)·행동·신뢰도·소스·영상경로·길이가 들어간다.

설계:
  - 프리롤(pre-roll) 링버퍼로 이상행동 직전 수 초도 함께 저장
  - 이상행동이 멈춘 뒤 post_roll 초까지 녹화 후 종료(또는 max_duration 도달 시)
  - 여러 VideoThread 가 동시에 같은 JSON 에 기록하므로 파일 접근은 락으로 보호
"""

import os
import sys
import json
import time
import ctypes
import threading
from collections import deque
from datetime import datetime

import cv2

from core.labels import label_kr, label_en

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD_DIR = os.path.join(_ROOT, "recordings")
_JSON_PATH = os.path.join(RECORD_DIR, "recordings.json")
_JSON_LOCK = threading.Lock()


def _ascii_path(path: str) -> str:
    """Windows에서 한글 경로를 단축경로(ASCII)로 변환 — cv2.VideoWriter 우회용.

    (cv2.VideoWriter 는 비ASCII 경로에서 열기 실패. 단축경로는 부모 폴더가
     실제 존재해야 변환되므로 호출 전에 폴더를 생성해 둔다.)
    """
    if sys.platform != "win32":
        return path
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(d, buf, 1024)
    if n:
        return os.path.join(buf.value, os.path.basename(path))
    return path


# ─────────────────────────────────────────────── JSON 기록 입출력 (스레드 안전)
def _append_record(rec: dict):
    with _JSON_LOCK:
        os.makedirs(RECORD_DIR, exist_ok=True)
        data = []
        if os.path.exists(_JSON_PATH):
            try:
                with open(_JSON_PATH, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = []
        data.append(rec)
        with open(_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_records() -> list:
    """저장된 녹화 기록 목록(최신순)을 반환."""
    with _JSON_LOCK:
        if not os.path.exists(_JSON_PATH):
            return []
        try:
            with open(_JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return sorted(data, key=lambda r: r.get("timestamp", ""), reverse=True)


def delete_record(rec_id: str):
    """기록 1건 + 연결된 영상/썸네일 파일 삭제."""
    removed = []
    with _JSON_LOCK:
        if not os.path.exists(_JSON_PATH):
            return
        try:
            with open(_JSON_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        keep = [r for r in data if r.get("id") != rec_id]
        removed = [r for r in data if r.get("id") == rec_id]
        with open(_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False, indent=2)
    for r in removed:
        for key in ("video_path", "thumbnail"):
            p = r.get(key)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


# ─────────────────────────────────────────────── 이벤트 녹화기 (소스별 1개)
class EventRecorder:
    """이상행동 발생 시 해당 구간을 녹화하는 소스 단위 녹화기."""

    def __init__(self, source_name: str, fps: float,
                 pre_roll: float = 2.0, post_roll: float = 3.0,
                 max_duration: float = 30.0, max_width: int = 960):
        self.source_name = source_name or "소스"
        self.fps = max(1.0, min(float(fps), 30.0))
        self.pre_roll = pre_roll
        self.post_roll = post_roll
        self.max_duration = max_duration
        self.max_width = max_width

        self._buf = deque(maxlen=max(1, int(self.pre_roll * self.fps)))
        self._writer = None
        self._recording = False
        self._start_time = 0.0
        self._last_abn_time = 0.0
        self._event = None
        self._size = None
        self._frames_written = 0
        self._video_path = None
        self._thumb_path = None

    def _resize(self, frame):
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / w
        return cv2.resize(frame, (self.max_width, int(h * scale)))

    def push(self, frame, is_abnormal: bool, cid: int, conf: float):
        """매 프레임 호출 — 버퍼 적립 및 녹화 상태 관리."""
        small = self._resize(frame)
        if not self._recording:
            self._buf.append(small.copy())
            if is_abnormal:
                self._start(small, cid, conf)
        else:
            if (small.shape[1], small.shape[0]) != self._size:
                small = cv2.resize(small, self._size)
            self._writer.write(small)
            self._frames_written += 1
            now = time.time()
            if is_abnormal:
                self._last_abn_time = now
            dur = now - self._start_time
            if (now - self._last_abn_time) > self.post_roll or dur > self.max_duration:
                self._finalize()

    def _start(self, small, cid, conf):
        now = time.time()
        ts = datetime.now()
        h, w = small.shape[:2]
        self._size = (w, h)
        stamp = ts.strftime("%Y%m%d_%H%M%S")
        beh_en = label_en(cid)
        os.makedirs(RECORD_DIR, exist_ok=True)
        self._video_path = os.path.join(RECORD_DIR, f"rec_{stamp}_{beh_en}.mp4")
        self._thumb_path = os.path.join(RECORD_DIR, f"rec_{stamp}_{beh_en}.jpg")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            _ascii_path(self._video_path), fourcc, self.fps, (w, h))

        ok, jbuf = cv2.imencode(".jpg", small)
        if ok:
            jbuf.tofile(self._thumb_path)   # numpy tofile 은 한글 경로 안전

        # 프리롤 버퍼 기록
        for f in self._buf:
            if (f.shape[1], f.shape[0]) != self._size:
                f = cv2.resize(f, self._size)
            self._writer.write(f)
            self._frames_written += 1
        self._buf.clear()

        self._recording = True
        self._start_time = now
        self._last_abn_time = now
        self._event = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "behavior_kr": label_kr(cid),
            "behavior_en": beh_en,
            "confidence": round(float(conf), 3),
            "source": self.source_name,
        }

    def _finalize(self):
        if not self._recording:
            return
        if self._writer is not None:
            self._writer.release()
        dur = self._frames_written / self.fps if self.fps else 0.0
        rec = dict(self._event)
        rec["id"] = os.path.splitext(os.path.basename(self._video_path))[0]
        rec["video_path"] = os.path.abspath(self._video_path)
        rec["thumbnail"] = os.path.abspath(self._thumb_path) if self._thumb_path else ""
        rec["duration_sec"] = round(dur, 1)
        _append_record(rec)

        self._recording = False
        self._writer = None
        self._frames_written = 0
        self._event = None

    def close(self):
        """스레드 종료 시 진행 중인 녹화를 마무리."""
        self._finalize()
