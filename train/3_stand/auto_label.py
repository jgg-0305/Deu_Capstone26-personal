# -*- coding: utf-8 -*-
"""매장 영상에서 '물품 가판대(stand)'와 '결제기(pos)'를 자동 라벨링해 YOLO 데이터셋 생성.

수동 주석이 없으므로 개방어휘 감지기(YOLO-World)로 의사 라벨링한다.
클래스별로 여러 영어 프롬프트를 두고, 검출 결과를 해당 클래스로 통합한다.

  class 0 = stand (물품 가판대/진열대)
  class 1 = pos   (결제기/계산대 단말기)

처리 흐름:
    1. 클래스별 소스(폴더/파일)에서 프레임 추출
    2. YOLO-World 추론 → 프롬프트→클래스 매핑 → 클래스별 NMS → conf 필터
    3. 검출이 있는 프레임만 이미지 + YOLO 라벨(.txt)로 저장
    4. data.yaml 생성

출력:
    train/3_stand/dataset/images|labels/{train,val}/...
    train/3_stand/dataset/data.yaml
"""

import argparse
import os
import random

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BEHAVIOR_DATA = os.path.normpath(
    os.path.join(_HERE, "..", "1_behavior", "split", "data"))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))

CLASS_NAMES = ["stand", "pos"]

# 클래스별 개방어휘 프롬프트
CLASS_PROMPTS = {
    0: ["shelf", "store shelf", "display shelf", "merchandise shelf",
        "product display rack", "display stand", "goods rack"],
    1: ["cash register", "pos terminal", "point of sale device",
        "card payment terminal", "checkout monitor", "desktop monitor screen"],
}
# 클래스별 채택 최소 신뢰도 (POS 는 약하게 잡혀 낮춤)
CLASS_CONF = {0: 0.12, 1: 0.05}

# 클래스별 소스 — data 하위 폴더명 또는 (프로젝트 루트 기준) 파일 경로
CLASS_SOURCES = {
    0: ["01.매장이동", "04.구매", "05.반품"],
    1: ["04.구매", "05.반품",
        os.path.join(_ROOT, "도난.mp4"), os.path.join(_ROOT, "1도난2.mp4")],
}

FRAME_STRIDE = 20
NMS_IOU = 0.5
MAX_BOXES = 12
TRAIN_RATIO = 0.8
SPLIT_SEED = 42


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def nms(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def resolve_videos(entry):
    """소스 항목(폴더명 또는 파일경로) → mp4 경로 리스트."""
    if entry.lower().endswith(".mp4"):
        return [entry] if os.path.isfile(entry) else []
    folder = os.path.join(_BEHAVIOR_DATA, entry)
    if not os.path.isdir(folder):
        print(f"  [경고] 소스 없음: {folder}")
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(".mp4")]


def collect_class_videos():
    """클래스별 (train_videos, val_videos) — 각 비디오는 (path, class_id)."""
    train, val = [], []
    for cid, sources in CLASS_SOURCES.items():
        vids = []
        for entry in sources:
            vids.extend(resolve_videos(entry))
        rng = random.Random(SPLIT_SEED + cid)
        rng.shuffle(vids)
        n_tr = max(1, int(len(vids) * TRAIN_RATIO))
        for i, v in enumerate(vids):
            (train if i < n_tr else val).append((v, cid))
    return train, val


def build_prompt_index(model):
    """모든 클래스 프롬프트를 한 번에 set_classes 하고, 프롬프트→클래스 매핑 반환."""
    all_prompts, prompt_cls = [], []
    for cid, prompts in CLASS_PROMPTS.items():
        for p in prompts:
            all_prompts.append(p)
            prompt_cls.append(cid)
    model.set_classes(all_prompts)
    return np.array(prompt_cls, dtype=np.int64)


def process(videos, split, model, prompt_cls, img_dir, lbl_dir, frame_stride):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    # 같은 영상이 두 클래스 소스에 모두 있으면 프레임을 합쳐 라벨링하기 위해
    # (video_path) → set(class_ids) 로 묶는다
    by_video = {}
    for vpath, cid in videos:
        by_video.setdefault(vpath, set()).add(cid)

    saved = 0
    min_conf = min(CLASS_CONF.values())
    for vi, (vpath, cids) in enumerate(by_video.items()):
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"  [오류] 열기 실패: {vpath}")
            continue
        stem = os.path.splitext(os.path.basename(vpath))[0]
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_stride != 0:
                idx += 1
                continue
            h, w = frame.shape[:2]
            r = model.predict(frame, conf=min_conf, verbose=False)[0]
            lines = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                pcls = r.boxes.cls.cpu().numpy().astype(int)
                cls = prompt_cls[pcls]               # 프롬프트→클래스
                for c in cids:                        # 이 영상에서 라벨링할 클래스만
                    sel = np.where((cls == c) & (conf >= CLASS_CONF[c]))[0]
                    if len(sel) == 0:
                        continue
                    keep = nms(xyxy[sel], conf[sel], NMS_IOU)[:MAX_BOXES]
                    for k in keep:
                        x1, y1, x2, y2 = xyxy[sel][k]
                        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                        bw, bh = (x2 - x1) / w, (y2 - y1) / h
                        if bw > 0 and bh > 0:
                            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                name = f"{stem}_{idx:05d}"
                imwrite_unicode(os.path.join(img_dir, name + ".jpg"), frame)
                with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                    f.write("\n".join(lines))
                saved += 1
            idx += 1
        cap.release()
        print(f"  [{split}] {vi+1}/{len(by_video)} {stem}  누적 {saved}장")
    print(f"[{split}] 저장 {saved}장")
    return saved


def main():
    ap = argparse.ArgumentParser(description="가판대+결제기 자동 라벨링")
    ap.add_argument("--model", default="yolov8x-worldv2.pt")
    ap.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    args = ap.parse_args()

    from ultralytics import YOLOWorld
    model = YOLOWorld(args.model)
    prompt_cls = build_prompt_index(model)

    dataset = os.path.join(_HERE, "dataset")
    train_v, val_v = collect_class_videos()
    print(f"소스 영상(클래스별 중복 포함): train {len(train_v)} / val {len(val_v)}")
    print(f"클래스: {CLASS_NAMES}\n")

    n_tr = process(train_v, "train", model, prompt_cls,
                   os.path.join(dataset, "images", "train"),
                   os.path.join(dataset, "labels", "train"), args.frame_stride)
    n_va = process(val_v, "val", model, prompt_cls,
                   os.path.join(dataset, "images", "val"),
                   os.path.join(dataset, "labels", "val"), args.frame_stride)

    yaml_path = os.path.join(dataset, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"\n완료: train {n_tr}장 / val {n_va}장\ndata.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
