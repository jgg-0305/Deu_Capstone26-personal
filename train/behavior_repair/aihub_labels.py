"""AI-Hub 71550 압축 자료의 영상과 CVAT 사건 주석을 연결한다. 같은 촬영의 카메라 영상이 분할에 섞이지 않도록 촬영 그룹 정보를
구성한다.
"""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
from xml.etree import ElementTree
from zipfile import ZipFile

LABELS = ("normal", "fall", "vandalism", "arson", "smoking", "abandon", "theft", "assault")
MARKERS = ("fall", "broken", "fire", "smoke", "abandon", "theft", "fight")
ARCHIVE_NAME = re.compile(r"^(TS|TL|VS|VL)_.*_(0[7-9]|1[0-3])\..+\.zip$")
GROUP_NAME = re.compile(r"^(C_3_\d+_\d+_BU_[A-Z]+_\d{2}[-_]\d{2})_")


@dataclass(frozen=True)
class Clip:
    name: str
    group: str
    source_split: str
    label: int
    video_zip: Path
    video: str
    video_identity: tuple[int, int]
    label_zip: Path
    xml: str


def group_of(name):
    match = GROUP_NAME.match(name)
    if match is None:
        raise ValueError(f"Unknown AI-Hub filming group: {name}")
    return match[1].replace("-", "_")


def members(path, extension):
    entries = {}
    with ZipFile(path) as archive:
        for item in archive.infolist():
            name = PurePosixPath(item.filename)
            if not item.is_dir() and name.suffix.lower() == extension:
                if name.stem in entries:
                    raise ValueError(f"Duplicate clip in {path}: {name.stem}")
                entries[name.stem] = item
    return entries


def index_archives(root):
    files = {}
    for path in root.rglob("*.zip"):
        match = ARCHIVE_NAME.fullmatch(path.name)
        if match:
            kind, code = match.groups()
            key = kind, int(code)
            if key in files:
                raise ValueError(f"Multiple archives for {key}: {path} / {files[key]}")
            files[key] = path
    clips = []
    audit = {}
    for source, raw_kind, xml_kind in (("Training", "TS", "TL"), ("Validation", "VS", "VL")):
        for index, label in enumerate(LABELS[1:], 1):
            code = index + 6
            required = [(raw_kind, code), (xml_kind, code)]
            missing = [key for key in required if key not in files or not files[key].stat().st_size]
            if missing:
                raise ValueError(f"Missing or empty AI-Hub archives: {missing}")
            video_zip, label_zip = (files[key] for key in required)
            raw, labels = members(video_zip, ".mp4"), members(label_zip, ".xml")
            paired = raw.keys() & labels.keys()
            if not paired:
                raise ValueError(f"No paired MP4 and XML members: {video_zip}")
            audit[f"{source}/{label}"] = {
                "paired": len(paired),
                "only_video": sorted(raw.keys() - labels.keys()),
                "only_xml": sorted(labels.keys() - raw.keys()),
            }
            clips.extend(Clip(name, group_of(name), source, index, video_zip, raw[name].filename,
                              (raw[name].CRC, raw[name].file_size), label_zip, labels[name].filename)
                         for name in sorted(paired))
    return clips, audit


def split_clips(clips, examined):
    # 클래스, 시나리오, 매장과 날짜로 묶는다. 시간과 카메라 접미사는 구분에 쓰지 않는다.
    examined = {group_of(PurePosixPath(name).stem) for name in examined}
    official_test = {clip.group for clip in clips if clip.source_split == "Validation"}
    validation = set()
    for label in range(1, len(LABELS)):
        available = {clip.group for clip in clips if clip.label == label}
        available -= official_test | examined
        ranked = sorted(available, key=lambda group: sha256(f"42:{group}".encode()).digest())
        if len(ranked) < 2:
            raise ValueError(f"Too few training groups for {LABELS[label]}")
        validation.update(ranked[:max(1, round(len(ranked) * 0.15))])
    assigned = {split: [] for split in ("train", "val", "test", "diagnostic")}
    for clip in clips:
        if clip.group in examined:
            assigned["diagnostic"].append(clip)
        elif clip.source_split == "Validation":
            assigned["test"].append(clip)
        elif clip.group not in official_test:
            assigned["val" if clip.group in validation else "train"].append(clip)
    return assigned


def remove_cross_split_duplicates(assigned):
    owners = {}
    excluded = {}
    for split in ("diagnostic", "test", "val", "train"):
        conflicts = {
            clip.group for clip in assigned[split]
            if owners.get(clip.video_identity, split) != split
        }
        excluded[split] = [clip.name for clip in assigned[split] if clip.group in conflicts]
        assigned[split] = [clip for clip in assigned[split] if clip.group not in conflicts]
        for clip in assigned[split]:
            owners[clip.video_identity] = split
    return excluded


def event_markers(root, label):
    prefix = MARKERS[label - 1]
    task = root.find("meta/task")
    if task is None or task.findtext("start_frame") != "0" or task.findtext("frame_filter"):
        raise ValueError("Unexpected CVAT frame origin or frame filter")
    frames = int(task.findtext("size"))
    width = int(task.findtext("original_size/width"))
    height = int(task.findtext("original_size/height"))
    starts, ends = [], []
    for track in root.findall("track"):
        kind = track.get("label")
        if kind not in (prefix + "_start", prefix + "_end"):
            continue
        visible = [box for box in track.findall("box") if box.get("outside") == "0"]
        if len(visible) != 1:
            raise ValueError(f"Expected one visible marker for {kind}")
        box = visible[0]
        frame = int(box.get("frame"))
        coords = tuple(float(box.get(key)) for key in ("xtl", "ytl", "xbr", "ybr"))
        if not (0 <= frame < frames and 0 <= coords[0] < coords[2] <= width
                and 0 <= coords[1] < coords[3] <= height):
            raise ValueError(f"Invalid event marker at {frame}")
        (starts if kind.endswith("_start") else ends).append((frame, coords))
    if not starts or len(starts) != len(ends):
        raise ValueError("Missing or unpaired start/end markers")
    events = sorted(zip(sorted(starts), sorted(ends)))
    # 사건이 겹치면 종료 상자에 인물 ID가 없으므로 시간 순서만으로 연결할 수 없다.
    if any(start[0] > end[0] for start, end in events) or any(
        left[1][0] >= right[0][0] for left, right in zip(events, events[1:])
    ):
        raise ValueError("Ambiguous overlapping or inverted events")
    return (frames, width, height), events


def read_annotation(archive, clip):
    root = ElementTree.fromstring(archive.read(clip.xml))
    return event_markers(root, clip.label)


def inventory(groups):
    return {split: dict(Counter(LABELS[item.label] for item in clips))
            for split, clips in groups.items()}
