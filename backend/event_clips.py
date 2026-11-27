"""실시간 영상의 감지 전후 프레임만 보관하고 브라우저 재생용 MP4로 변환한다."""

from bisect import bisect_right
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from shutil import which
import subprocess
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore

import cv2

from backend.samples import MAX_CLIP_BYTES

PRE_MS = 10_000
POST_MS = 5_000
FRAME_MS = 200


def encode_clip(ffmpeg, frames, start, end):
    timestamps = [timestamp for timestamp, _ in frames]
    images = b"".join(frames[bisect_right(timestamps, timestamp) - 1][1]
                      for timestamp in range(start, end + 1, FRAME_MS))
    with TemporaryDirectory() as directory:
        output = Path(directory) / "event.mp4"
        subprocess.run([
            ffmpeg, "-loglevel", "error", "-f", "image2pipe", "-framerate", "5",
            "-vcodec", "mjpeg", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "ultrafast", "-vf", "scale=in_range=full:out_range=tv",
            "-pix_fmt", "yuv420p", "-color_range", "tv", "-movflags", "+faststart", str(output),
        ], input=images, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True, timeout=20)
        if output.stat().st_size > MAX_CLIP_BYTES:
            raise ValueError("사건 클립 크기가 제한을 넘었습니다.")
        return output.read_bytes()


class LiveEventClips:
    """최근 영상만 메모리에 두고 사건이 생기면 앞뒤 구간을 별도 스레드에서 저장한다."""

    def __init__(self, publish):
        self.publish = publish
        self.ffmpeg = which("ffmpeg")
        self.frames = deque(maxlen=(PRE_MS + POST_MS) // FRAME_MS + 16)
        self.pending = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.slots = BoundedSemaphore(3)
        if not self.ffmpeg:
            print("사건 영상 저장에는 H.264 인코더가 포함된 FFmpeg가 필요합니다.", flush=True)

    def push(self, frame, timestamp):
        if not self.ffmpeg:
            return
        if not self.frames or timestamp - self.frames[-1][0] >= FRAME_MS:
            height, width = frame.shape[:2]
            scale = min(1, 640 / width, 480 / height)
            size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
            try:
                image = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.frames.append((timestamp, jpeg.tobytes()))
            except cv2.error:
                pass
        while self.pending and self.pending[0][0] + POST_MS <= timestamp:
            self.submit(*self.pending.pop(0))

    def start(self, timestamp, event_ids):
        if self.ffmpeg and event_ids and self.frames:
            self.pending.append((timestamp, event_ids))

    def submit(self, timestamp, event_ids):
        start = max(self.frames[0][0], timestamp - PRE_MS)
        end = min(self.frames[-1][0], timestamp + POST_MS)
        frames = tuple((time, image) for time, image in self.frames if time <= end)
        if not self.slots.acquire(blocking=False):
            print("사건 클립 저장 대기열이 가득 찼습니다.", flush=True)
            return
        self.executor.submit(self.save, frames, start, end, event_ids)

    def save(self, frames, start, end, event_ids):
        try:
            clip = encode_clip(self.ffmpeg, frames, start, end)
            for event_id in event_ids:
                try:
                    self.publish(event_id, clip, ((end - start) // FRAME_MS + 1) * FRAME_MS)
                except (RuntimeError, OSError) as error:
                    print(f"사건 클립 등록 실패: {type(error).__name__}", flush=True)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            print(f"사건 클립 인코딩 실패: {type(error).__name__}", flush=True)
        finally:
            self.slots.release()

    def reset(self):
        for timestamp, event_ids in self.pending:
            self.submit(timestamp, event_ids)
        self.pending.clear()
        self.frames.clear()

    def close(self):
        self.reset()
        self.executor.shutdown()
