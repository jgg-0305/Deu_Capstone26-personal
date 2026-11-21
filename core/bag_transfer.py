"""Associate a visible article with a carried bag across brief wrist occlusion."""

from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

from core.article_tracking import article_appearance, intersection_over_union


def center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def distance(first, second):
    return hypot(first[0] - second[0], first[1] - second[1])


class BagDetector:
    def __init__(self, path):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.classes = [index for index, name in self.model.names.items()
                        if name in {"backpack", "handbag", "suitcase"}]

    def warmup_live(self):
        if self.device == 0:
            crop = np.zeros((480, 480, 3), dtype=np.uint8)
            self.model.predict([crop, crop], classes=self.classes, conf=.18, imgsz=480,
                               device=self.device, verbose=False)

    def detect(self, frame, people):
        height, width = frame.shape[:2]
        views = []
        for person in people:
            x1, y1, x2, y2 = person["bbox"]
            scale = max(1., y2 - y1)
            for top, bottom, side in ((-.18, 1.18, .35), (.21, .90, .30)):
                left = max(0, int(x1 - side * scale))
                right = min(width, int(x2 + side * scale))
                start = max(0, int(y1 + top * scale))
                end = min(height, int(y1 + bottom * scale))
                if right - left >= 16 and end - start >= 16:
                    views.append((person["track_id"], left, start, frame[start:end, left:right]))
        candidates = {}
        if not views:
            return candidates
        results = self.model.predict([view[3] for view in views], classes=self.classes,
                                     conf=.18, imgsz=480, device=self.device, verbose=False)
        for (tid, left, top, _), result in zip(views, results):
            for x1, y1, x2, y2, score, _ in result.boxes.data.cpu().tolist():
                box = (x1 + left, y1 + top, x2 + left, y2 + top)
                candidates.setdefault(tid, []).append((box, score))
        return candidates


@dataclass
class ArticleMotion:
    origin: tuple
    box: tuple
    appearance: object
    slot: int
    wrist: int
    wrist_position: tuple
    observations: int = 1
    entered_bag: tuple | None = None
    wrist_bridge: bool = False


class BagTransferTracker:
    def __init__(self, *, recover_compressed=False):
        self.recover_compressed = recover_compressed
        self.reset()

    def reset(self):
        self.items = {}
        self.active = {}
        self.checked_until = {}
        self.slot = None

    def update(self, people, detections, now, frame, detector):
        slot = int(now * 3 + 1e-6)
        if self.slot is not None and slot < self.slot:
            self.reset()
        if slot == self.slot:
            return set(self.active)
        self.slot = slot
        self.active = {tid: end for tid, end in self.active.items() if now <= end}
        self.checked_until = {tid: end for tid, end in self.checked_until.items() if now <= end}
        observed = {person["track_id"]: person for person in people if person.get("observed", True)}
        max_gap = 3 if self.recover_compressed else 2
        self.items = {tid: item for tid, item in self.items.items()
                      if tid in observed and slot - item.slot <= max_gap}
        moving = {}
        matches = {}
        for tid, person in observed.items():
            points = person.get("keypoints")
            if points is None or tid in self.checked_until:
                continue
            scale = max(1., person["bbox"][3] - person["bbox"][1])
            item = self.items.get(tid)
            best = None
            if item is not None:
                last = center(item.box)
                gap = slot - item.slot
                for box, score, appearance in detections:
                    pos = center(box)
                    step = distance(pos, last) / scale
                    visual = cv2.compareHist(item.appearance, appearance, cv2.HISTCMP_BHATTACHARYYA)
                    # A weak detection may continue a visually consistent article.
                    # Its path remains available when the seeding wrist is hidden.
                    wrist = points[item.wrist]
                    if (score < .28 or step > .17 * gap or visual > .52
                            or (wrist[2] >= .55 and distance(pos, wrist) > .20 * scale)):
                        continue
                    wrist_bridge = False
                    if gap > 2:
                        old_x, old_y = item.wrist_position
                        movement = (float(wrist[0]) - old_x, float(wrist[1]) - old_y)
                        prediction = (last[0] + movement[0], last[1] + movement[1])
                        old_area = (item.box[2] - item.box[0]) * (item.box[3] - item.box[1])
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        wrist_bridge = (wrist[2] >= .70 and visual <= .42
                                        and distance(prediction, pos) <= .10 * scale
                                        and .50 * old_area <= area <= 1.80 * old_area)
                        if not wrist_bridge:
                            continue
                    cost = step + .30 * visual - .20 * score
                    if best is None or cost < best[0]:
                        best = (cost, box, wrist_bridge)
                if best is not None:
                    item.box, item.slot = best[1], slot
                    item.observations += 1
                    item.wrist_bridge |= best[2]
                    if points[item.wrist, 2] >= .55:
                        item.wrist_position = tuple(points[item.wrist, :2])
                    matches[tid] = item
                if item.observations >= 2 and distance(center(item.box), item.origin) >= .12 * scale:
                    moving[tid] = (person, scale)
                    continue
                if best is not None or slot == item.slot:
                    continue
            seeds = [(score, box, appearance, wrist) for box, score, appearance in detections
                     for wrist in (9, 10) if points[wrist, 2] >= .55 and score >= .42
                     and distance(center(box), points[wrist]) <= .10 * scale]
            if seeds:
                _, box, appearance, wrist = max(seeds, key=lambda candidate: candidate[0])
                self.items[tid] = ArticleMotion(center(box), box, appearance, slot, wrist, tuple(points[wrist, :2]))

        bags = detector.detect(frame, [person for person, _ in moving.values()]) if moving else {}
        for tid, (person, scale) in moving.items():
            item = self.items[tid]
            for bag, score in bags.get(tid, ()):
                x1, y1, x2, y2 = bag
                bx, by = center(bag)
                px1, py1, px2, py2 = person["bbox"]
                if (not px1 - .20 * scale <= bx <= px2 + .20 * scale
                        or not py1 + .35 * scale <= by <= py2):
                    continue
                ix, iy = center(item.box)
                upper = y1 + .66 * (y2 - y1)
                outside = not (x1 - .04 * scale <= item.origin[0] <= x2 + .04 * scale
                               and y1 <= item.origin[1] <= y2)
                continuous = item.observations >= 3 or (item.wrist_bridge and
                    distance(center(item.box), item.origin) >= .18 * scale)
                inside_transfer = (continuous and outside and x1 <= ix <= x2 and y1 <= iy <= upper
                                   and (item.box[2] - item.box[0]) * (item.box[3] - item.box[1])
                                   <= .30 * (x2 - x1) * (y2 - y1))
                if tid in matches and inside_transfer:
                    item.entered_bag = bag
                confirmed_bag = (item.entered_bag is not None and score >= .32
                                 and intersection_over_union(bag, item.entered_bag) >= .35)
                bag_just_detected = (item.wrist_bridge and score >= .40 and inside_transfer
                                     and slot - item.slot == 1)
                if (tid not in matches and 1 <= slot - item.slot <= max_gap - 1
                        and (confirmed_bag or bag_just_detected)):
                    # A one-frame detector miss is insufficient if the item
                    # still looks the same at its last visible position.
                    remaining = article_appearance(frame, item.box)
                    self.checked_until[tid] = now + 2.
                    if (remaining is not None and
                            cv2.compareHist(item.appearance, remaining, cv2.HISTCMP_BHATTACHARYYA) >= .60):
                        self.active[tid] = now + 2.
                    self.items.pop(tid)
                    break
        return set(self.active)
