"""분석 워커와 재생 API에서 공통으로 허용하는 영상 파일."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = frozenset({"demo-1.mp4", "demo-2.mp4"})


def sample_path(name: str) -> Path:
    if name not in SAMPLES:
        raise ValueError("Unknown sample video")
    path = ROOT / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Sample video is unavailable")
    return path


# 업로드 영상은 저장소 밖으로 노출되지 않는 경로에 서버가 부여한 이름으로 보관한다.
UPLOAD_DIR = ROOT / ".local-videos"
MAX_VIDEO_BYTES = 256 * 1024 * 1024


def source_path(key: str) -> Path:
    if key in SAMPLES:
        return sample_path(key)
    from uuid import UUID

    prefix, _, name = key.partition("/")
    if prefix != "uploads" or not name.endswith(".mp4"):
        raise ValueError("등록된 영상 경로가 아닙니다.")
    identifier = UUID(name[:-4])
    if name != f"{identifier}.mp4":
        raise ValueError("등록된 영상 경로가 아닙니다.")
    path = UPLOAD_DIR / name
    if UPLOAD_DIR.is_symlink() or path.is_symlink() or not path.is_file():
        raise FileNotFoundError("영상 파일을 찾을 수 없습니다.")
    return path


def asset_path(key: str) -> Path:
    parts = key.split("/")
    if len(parts) != 3:
        raise ValueError("등록된 영상 경로가 아닙니다.")
    kind, _, name = parts
    if kind == "samples":
        return sample_path(name)
    if kind == "uploads":
        return source_path(f"uploads/{name}")
    raise ValueError("등록된 영상 경로가 아닙니다.")
