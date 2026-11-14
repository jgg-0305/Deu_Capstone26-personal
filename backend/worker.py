"""저장소의 모델로 등록된 파일과 실시간 영상 소스를 분석한다."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Thread
from queue import Empty, Full, Queue
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
import signal
import time
from uuid import UUID

import cv2
import numpy as np
import requests
import torch

from backend.analysis_media import frame_path
from backend.behavior_upload import MODEL_DIR, UPLOAD_PREFIX
from backend.live import open_capture
from backend.samples import ROOT, source_path
from core.detector import HumanDetector, ObjectDetector
from core.kpt_behavior_classifier import TrackBehaviorBuffer
from core.labels import label_en
from core.rgb_behavior_classifier import RgbBehaviorClassifier
from core.theft_monitor import TheftMonitor
from core.tracker import HumanTracker
from core.tracking_geometry import match_keypoints

RULE_VERSION = "stand-pos-v1"
STOP = Event()


def limit_cpu_threads(predictor):
    """YOLO 초기화가 덮어쓴 CPU 병렬도를 추론 시작 시 복원한다."""
    if predictor.device.type == "cpu" and torch.get_num_threads() > 2:
        torch.set_num_threads(2)


class SourceError(RuntimeError):
    """카메라 주소를 담지 않는 영상 입력 오류."""


class WorkerError(RuntimeError):
    def __init__(self, path, status):
        self.status = status
        super().__init__(f"분석 API {path}: HTTP {status}")


class WorkerClient:
    def __init__(self, url, token):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def request(self, method, path, body=None):
        response = self.session.request(method, self.url + "/internal/v1" + path,
                                        json=body, timeout=(5, 15), allow_redirects=False)
        if not response.ok or response.is_redirect:
            raise WorkerError(path, response.status_code)
        return response.json() if response.content else None


@lru_cache(maxsize=16)
def model_digest(path, size, mtime_ns):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verified_models(api, behavior_model, pose_model="human_pose"):
    manifest = json.loads((ROOT / "docs/design/model-manifest.json").read_text(encoding="utf-8"))
    registered = {(item["name"], item["sha256"]): item for item in api.request("GET", "/models")}
    roles = {pose_model: "pose", "store_objects": "object", behavior_model: "behavior"}
    paths, ids = {}, {}
    for artifact in manifest["models"]:
        role = roles.get(artifact["id"])
        if role is None:
            continue
        path = ROOT / artifact["path"]
        stat = path.stat()
        digest = model_digest(path, stat.st_size, stat.st_mtime_ns)
        model = registered.get((artifact["id"], digest))
        if digest != artifact["sha256"] or model is None or model["role"] != role:
            raise RuntimeError(f"{role} 모델을 확인할 수 없습니다. 모델을 다시 등록하세요.")
        paths[role], ids[role] = str(path), model["id"]
    if behavior_model.startswith(UPLOAD_PREFIX):
        digest = behavior_model.removeprefix(UPLOAD_PREFIX)
        model = registered.get((behavior_model, digest))
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or model is None or model["role"] != "behavior"):
            raise RuntimeError("첨부한 행동 모델이 등록되어 있지 않습니다.")
        path = MODEL_DIR / f"{digest}.pth"
        if MODEL_DIR.is_symlink() or path.is_symlink() or not path.is_file():
            raise RuntimeError("첨부한 행동 모델 파일을 찾을 수 없습니다.")
        stat = path.stat()
        if model_digest(path, stat.st_size, stat.st_mtime_ns) != digest:
            raise RuntimeError("첨부한 행동 가중치의 해시가 일치하지 않습니다.")
        paths["behavior"], ids["behavior"] = str(path), model["id"]
    if paths.keys() != {"pose", "object", "behavior"}:
        raise RuntimeError("분석에 필요한 모델이 목록에 없습니다.")
    return paths, ids


@contextmanager
def source_frames(run, camera_id):
    source_type = run["source_type"]
    if source_type == "webcam":
        yield webcam_frames(camera_id)
        return
    if source_type == "rtsp":
        with rtsp_frames(run["rtsp_url"]) as frames:
            yield frames
        return
    capture = cv2.VideoCapture(str(source_path(run["source_storage_key"])))
    try:
        if not capture.isOpened():
            raise SourceError("영상 소스를 열 수 없습니다.")
        run["total_frames"] = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = capture.get(cv2.CAP_PROP_FPS)
        run["source_fps"] = fps if math.isfinite(fps) and fps > 0 else None
        yield capture_frames(capture)
    finally:
        capture.release()


def capture_frames(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        pts = capture.get(cv2.CAP_PROP_POS_MSEC)
        if not math.isfinite(pts) or pts < 0 or (index and pts == 0):
            pts = index * 1000 / fps
        yield frame, pts
        index += 1


@contextmanager
def rtsp_frames(url):
    # 추론이 느려도 카메라 입력을 계속 읽어 오래된 프레임이 밀리지 않게 한다.
    latest = Queue(maxsize=1)
    stopped = Event()

    def read():
        capture = None
        started = time.monotonic()
        try:
            capture = open_capture(url)
            while not stopped.is_set() and not STOP.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                try:
                    latest.put_nowait((frame, (time.monotonic() - started) * 1000))
                except Full:
                    try:
                        latest.get_nowait()
                    except Empty:
                        pass
                    latest.put_nowait((frame, (time.monotonic() - started) * 1000))
        finally:
            stopped.set()
            if capture is not None:
                capture.release()

    def consume():
        while not stopped.is_set() and not STOP.is_set():
            try:
                yield latest.get(timeout=0.25)
            except Empty:
                continue
        if not STOP.is_set():
            raise SourceError("RTSP 연결이 종료되었습니다.")

    reader = Thread(target=read, daemon=True)
    reader.start()
    try:
        yield consume()
    finally:
        stopped.set()
        reader.join(timeout=6)


def webcam_frames(camera_id):
    path = frame_path(camera_id, "webcam.jpg")
    last_stamp = 0
    started = time.monotonic()
    last_frame = started
    while not STOP.is_set() and time.monotonic() - last_frame < 25:
        try:
            with path.open("rb") as image:
                stat = os.fstat(image.fileno())
                if stat.st_mtime_ns != last_stamp and time.time() - stat.st_mtime < 10:
                    last_stamp = stat.st_mtime_ns
                    frame = cv2.imdecode(np.frombuffer(image.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        last_frame = time.monotonic()
                        yield frame, (last_frame - started) * 1000
                        continue
        except FileNotFoundError:
            pass
        time.sleep(0.08)
    if not STOP.is_set():
        raise SourceError("브라우저에서 웹캠 프레임을 받지 못했습니다.")


def publish_snapshot(api, path, lease, run_id, frame, tracks, keypoints, predictions,
                     objects, suspects, frame_count, pts, fps, scene, file_source, diagnostics):
    height, width = frame.shape[:2]

    def box(item):
        x1, y1, x2, y2 = item["bbox"]
        return [x1 / width, y1 / height, x2 / width, y2 / height]

    people = []
    for track in tracks:
        tid = track["track_id"]
        label, score = predictions.get(tid, (None, None))
        progress = diagnostics["buffers"].get(tid)
        if progress and progress["state"] == "uncertain":
            # 임계값 미달인 이상행동 후보를 확정된 정상 결과처럼 표시하지 않는다.
            label, score = None, None
        if tid in suspects and not label:
            label, score = 6, None
        points = keypoints.get(tid)
        if points is None:
            points = ()
        people.append({
            "id": tid, "box": box(track), "label": label, "score": score,
            "model_result": predictions.get(tid),
            "observed": track.get("observed", True), "missing_frames": track.get("missing_frames", 0),
            "method": "rule" if tid in suspects and score is None else "model",
            "buffer": progress,
            "points": [[float(x) / width, float(y) / height, float(conf)] for x, y, conf in points],
        })
    snapshot = {
        "run_id": run_id, "frame_count": frame_count, "time_ms": round(pts),
        "fps": round(fps, 1), "people": people, "scene": scene,
        "objects": [{"class_id": item["cls"], "box": box(item)} for item in objects],
        "width": width, "height": height, "updated_at_ms": round(time.time() * 1000),
        **{key: value for key, value in diagnostics.items() if key != "buffers"},
    }
    if not file_source:
        scale = min(1, 640 / width, 480 / height)
        if scale < 1:
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if not ok:
            return
        snapshot["image"] = base64.b64encode(image.tobytes()).decode("ascii")
    api.request("POST", path + "/snapshot", {**lease, "snapshot": snapshot, "timeline": file_source})


@torch.inference_mode()
def analyze(api, camera_id, threshold, job=None):
    settings = {
        "resolution": 480, "detection_interval": 1, "detection_confidence": 0.4,
        "max_age": 30, "event_threshold": 0.6, **(job.get("settings", {}) if job else {}),
    }
    behavior_model = settings.pop("behavior_model", "behavior")
    pose_model = settings.pop("pose_model", "human_pose")
    paths, ids = verified_models(api, behavior_model, pose_model)
    claim = {"job_id": job["id"], "claim_token": job["claim_token"]} if job else {}
    run = api.request("POST", "/runs", {
        **claim, **settings, "camera_id": str(camera_id), "worker_id": "video-worker",
        "models": ids, "rule_version": RULE_VERSION, "behavior_threshold": threshold,
    })
    path = f"/runs/{run['id']}"
    lease = {"lease_token": run["lease_token"]}
    file_source = run["source_type"] in {"file", "sample"}
    status = "failed"
    count = 0
    frame_count = 0
    error_code = "model_load_failed"
    try:
        api.request("POST", path + "/snapshot", {**lease, "snapshot": {
            "phase": "loading", "people": [], "objects": [], "frame_count": 0,
            "pose_model": pose_model, "behavior_model": behavior_model,
            "inference_count": 0, "pose_count": 0, "fps": 0,
            "updated_at_ms": round(time.time() * 1000),
        }})
        pose = HumanDetector(paths["pose"], conf_threshold=settings["detection_confidence"],
                             imgsz=settings["resolution"])
        detector = ObjectDetector(paths["object"], conf_threshold=0.25, imgsz=settings["resolution"])
        pose.model.add_callback("on_predict_start", limit_cpu_threads)
        detector.model.add_callback("on_predict_start", limit_cpu_threads)
        rgb = RgbBehaviorClassifier(paths["behavior"]) if behavior_model == "behavior_rgb" else None
        behavior = None if rgb else TrackBehaviorBuffer(paths["behavior"], abn_conf_thr=threshold,
                                                       max_missing=settings["max_age"])
        scene = None
        predictions = {}
        objects = []
        suspects = set()
        last_frame_pts = -math.inf
        pose_count = 0
        tracks, keypoints = [], {}
        tracker = HumanTracker(max_age=settings["max_age"])
        rule = TheftMonitor()
        if not pose.is_pose or detector.names != {0: "stand", 1: "pos"}:
            raise RuntimeError("등록된 모델의 분류 구성이 올바르지 않습니다.")
        error_code = "inference_failed"

        def diagnostics():
            classifier = rgb or behavior
            return {
                "phase": "classifying" if classifier.inference_count else "collecting",
                "pose_model": pose_model, "behavior_model": behavior_model,
                "pose_count": pose_count, "inference_count": classifier.inference_count,
                "total_frames": run.get("total_frames", 0), "device": str(classifier.device),
                "cpu_threads": torch.get_num_threads(),
                "source_fps": run.get("source_fps"),
                "frame_duration_ms": round(1000 * settings["detection_interval"] / run["source_fps"], 3) if run.get("source_fps") else None,
                "buffer_reset_count": behavior.reset_count if behavior else 0,
                "buffers": {t["track_id"]: behavior.progress(t["track_id"]) for t in tracks} if behavior else {},
                "scene_buffer": {"collected": len(rgb.features), "required": 3} if rgb else None,
                "behavior_threshold": threshold, "event_threshold": settings["event_threshold"],
            }
        previous = set()
        last_heartbeat = started = time.monotonic()
        last_snapshot = 0
        with source_frames(run, camera_id) as frames:
            for frame, pts in frames:
                if STOP.is_set():
                    status = "stopped"
                    return
                now = time.monotonic()
                if now - last_heartbeat >= 8:
                    api.request("POST", path + "/heartbeat", lease)
                    last_heartbeat = now
                if frame_count % settings["detection_interval"] == 0:
                    detections = pose.detect(frame)
                    pose_count += 1
                    tracks = tracker.update(detections, frame)
                    keypoints = match_keypoints(tracks, detections)
                    # 원본과 같은 검출 시점에 가판대와 결제기 영역을 갱신한다.
                    objects = detector.detect(frame)
                    if rgb:
                        scene = rgb.update(frame)
                    else:
                        predictions = behavior.update(tracks, keypoints)
                    observed_tracks = [t for t in tracks if t.get("observed", True)]
                    suspects = rule.update(observed_tracks, keypoints, objects, source_seconds=pts / 1000, include_lost=True)
                present = {track["track_id"] for track in tracks if track.get("observed", True)}
                candidates = {
                    (tid, label_en(label).lower().replace("theft", "theft_suspected"), "model"): {
                        "model_score": score, "score_model_role": "behavior",
                    }
                    for tid, (label, score) in predictions.items()
                    if label and score >= max(threshold, settings["event_threshold"])
                }
                if scene and scene[0] and scene[1] >= max(threshold, settings["event_threshold"]):
                    kind = label_en(scene[0]).lower().replace("theft", "theft_suspected")
                    candidates[(None, kind, "model")] = {"model_score": scene[1], "score_model_role": "behavior"}
                for tid in suspects:
                    candidates[(tid, "theft_suspected", "rule")] = {
                        "rule_code": "stand_pos", "rule_version": RULE_VERSION,
                        "evidence": {"reason": "pos_timeout" if tid in present else "track_lost"},
                    }
                for (tid, kind, method), details in candidates.items():
                    if (tid, kind, method) in previous:
                        continue
                    api.request("POST", path + "/events", {
                        **lease, **details, "event_type": kind, "detection_method": method,
                        "track_id": str(tid) if tid is not None else None, "producer_event_key": f"{frame_count}:{tid}:{kind}:{method}",
                        "detected_at": datetime.now(timezone.utc).isoformat(),
                        "source_time_ms": round(pts), "source_frame_index": frame_count,
                    })
                    count += 1
                previous = set(candidates)
                frame_count += 1
                publish_due = (frame_count - 1) % settings["detection_interval"] == 0 if file_source else now - last_snapshot >= 0.20
                if publish_due:
                    publish_snapshot(api, path, lease, run["id"], frame, tracks, keypoints,
                                     predictions, objects, suspects, frame_count, pts,
                                     frame_count / max(0.01, time.monotonic() - started), scene, file_source, diagnostics())
                    last_snapshot = now
                    last_frame_pts = pts
        if STOP.is_set():
            status = "stopped"
            return
        if frame_count == 0:
            raise SourceError("해석 가능한 영상 프레임이 없습니다.")
        publish_snapshot(api, path, lease, run["id"], frame, tracks, keypoints,
                         predictions, objects, suspects, frame_count, pts,
                         frame_count / max(0.01, time.monotonic() - started), scene, file_source,
                         {**diagnostics(), "phase": "completed"})
        status = "completed"
    except SourceError:
        error_code = "source_unavailable"
        raise
    except WorkerError as error:
        if error.status != 409:
            raise
        status = "stopped"
    except KeyboardInterrupt:
        status = "stopped"
        raise
    finally:
        try:
            api.request("POST", path + "/finish", {
                **lease, "status": status, "error_code": error_code if status == "failed" else None,
            })
        except WorkerError as error:
            if error.status != 409:
                raise
    print(f"분석 {run['id']} {status}: {frame_count}프레임, 사건 {count}건", flush=True)


def run_job(args, token, job):
    api = WorkerClient(args.api_url, token)
    try:
        analyze(api, job["camera_id"], job["threshold"], job)
    except Exception as error:
        try:
            api.request("POST", f"/jobs/{job['id']}/fail", {"claim_token": job["claim_token"]})
        except (WorkerError, requests.RequestException):
            pass
        # 연결 주소나 인증 정보가 들어갈 수 있는 외부 라이브러리 오류는 기록하지 않는다.
        detail = str(error) if isinstance(error, (SourceError, WorkerError)) else type(error).__name__
        print(f"분석 {job['id']} 실패: {detail}", flush=True)
    finally:
        api.session.close()


def serve(api, args, token):
    running = {"file": [], "live": []}
    executor = ThreadPoolExecutor(max_workers=1 + args.live_slots)
    try:
        print("관제실 영상 분석 요청을 기다립니다.", flush=True)
        while not STOP.is_set():
            for kind, limit in (("file", 1), ("live", args.live_slots)):
                running[kind] = [job for job in running[kind] if not job.done()]
                if len(running[kind]) < limit:
                    try:
                        job = api.request("POST", f"/jobs/claim?kind={kind}")
                    except (WorkerError, requests.RequestException) as error:
                        print(f"분석 대기열 연결 실패: {error}", flush=True)
                        break
                    if job:
                        running[kind].append(executor.submit(run_job, args, token, job))
            time.sleep(1)
    finally:
        STOP.set()
        executor.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--camera-id", type=UUID)
    mode.add_argument("--serve", action="store_true", help="관제실의 분석 요청을 처리합니다.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--live-slots", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()
    token = os.environ.get("CCTV_WORKER_TOKEN", "")
    if len(token) < 32 or not 0.5 <= args.threshold <= 1:
        parser.error("Set CCTV_WORKER_TOKEN and use a threshold between 0.5 and 1.")
    api = WorkerClient(args.api_url, token)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    try:
        torch.set_num_threads(min(2, torch.get_num_threads()))
        cv2.setNumThreads(1)
        if args.serve:
            serve(api, args, token)
        else:
            analyze(api, args.camera_id, args.threshold)
    except KeyboardInterrupt:
        pass
    finally:
        api.session.close()


if __name__ == "__main__":
    main()
