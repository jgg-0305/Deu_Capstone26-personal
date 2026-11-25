"""손목 주변 상품을 검출하고 같은 사람의 손에 연결한다. 관측 시각, 손과 상품의 상대 이동, 상품 외관을 함께 확인해 짧은 검출 누락을
이어간다. 움직인 상품이 가려졌을 때 주변 증거를 검토하고 흡연 후보와의 충돌을 조정한다.
"""
from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np


def wrist_roi(wrist, elbow, scale, width, height):
    x, y = wrist[:2]
    if elbow is not None:
        offset = (np.asarray(wrist[:2]) - np.asarray(elbow[:2])) * .25
        x, y = np.asarray((x, y)) + np.clip(offset, -.1 * scale, .1 * scale)
    radius = int(np.clip(scale * .38, 38, 230))
    return max(0, int(x)-radius), max(0, int(y)-radius), min(width, int(x)+radius), min(height, int(y)+radius)


def article_appearance(frame, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if not crop.size:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).flatten()
    return histogram / max(1., float(histogram.sum()))


@dataclass
class Carry:
    origin: tuple
    offset: tuple
    last: float
    appearance: np.ndarray
    box: tuple
    wrist: tuple
    maximum_hip_distance: float | None
    last_hip_distance: float | None
    box_slot: int
    maximum_item_hip_distance: float | None = None
    last_item_hip_distance: float | None = None
    minimum_item_hip_distance: float | None = None
    close_item_hits: int = 0
    hits: int = 1
    distance: float = 0.
    bridge_frames: int = 0
    pocket_reacquired: bool = False
    resumed_close_hits: int | None = None
    last_appearance: np.ndarray | None = None


def follow_box(previous_gray, gray, box, wrist_delta, scale):
    """주변 영상 특징과 손목 이동이 일치할 때만 마지막 상품 상자의 위치를 이어 간다."""
    x1, y1, x2, y2 = map(int, box)
    h, w = gray.shape
    x1, x2 = max(0, x1), min(w, x2+1)
    y1, y2 = max(0, y1), min(h, y2+1)
    patch = previous_gray[y1:y2, x1:x2]
    if min(patch.shape, default=0) < 5:
        return None
    corners = cv2.goodFeaturesToTrack(patch, maxCorners=18, qualityLevel=.025, minDistance=2)
    if corners is None or len(corners) < 4:
        return None
    old = corners + np.array([x1, y1], dtype=np.float32)
    predicted = old + np.array(wrist_delta, dtype=np.float32)
    options = dict(winSize=(15, 15), maxLevel=2,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03),
                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    new, status1, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, old, predicted.copy(), **options)
    if new is None:
        return None
    back, status2, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, new, old.copy(), **options)
    if back is None:
        return None
    valid = status1.ravel().astype(bool) & status2.ravel().astype(bool)
    valid &= np.linalg.norm((back-old).reshape(-1, 2),axis=1) <= 2.0
    if valid.sum() < 4 or valid.mean() < .55:
        return None
    shifts = (new-old).reshape(-1,2)[valid]
    delta = np.median(shifts, axis=0)
    if (np.median(np.linalg.norm(shifts-delta,axis=1)) > max(2., .025*scale)
            or np.linalg.norm(delta-wrist_delta) > max(4., .085*scale)):
        return None
    dx,dy = delta.tolist()
    shifted = (box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy)
    return shifted if 0 <= shifted[0] < shifted[2] <= w and 0 <= shifted[1] < shifted[3] <= h else None


class ArticleTracker:
    """같은 손에 연결된 상품이 움직인 뒤 골반 가까이에서 짧게 사라졌는지 확인한다."""
    def __init__(self, *, pocket_recovery=False, continuous_carry=False, strict_concealment=False, recover_compressed=False):
        self.pocket_recovery = pocket_recovery
        self.continuous_carry = continuous_carry
        self.strict_concealment = strict_concealment
        self.recover_compressed = recover_compressed
        self.tracks = {}
        self.active = {}
        self.slot = None
        self.gray = None
        self.bridged_frames = 0
        self.weak_bridged_frames = 0
        self.reacquired_tracks = 0
        self.trace = None

    def needs_frame(self, now):
        return self.slot != int(now * 3 + 1e-6)

    def reset(self):
        self.tracks.clear()
        self.active.clear()
        self.slot = None
        self.gray = None
        self.trace = None

    def update(self, tracks, detections, now, frame=None):
        # 상품 학습 자료의 초당 3프레임에 맞춰 추적 상태를 갱신한다.
        slot = int(now * 3 + 1e-6)
        if self.slot is not None and slot < self.slot:
            self.reset()
        if self.slot == slot:
            return dict(self.active)
        consecutive = self.slot is not None and slot == self.slot+1
        self.slot = slot
        previous_gray = self.gray if consecutive else None
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame is not None else None
        if gray is not None and previous_gray is not None and gray.shape != previous_gray.shape:
            previous_gray = None
        self.gray = gray
        self.active = {tid: expiry for tid, expiry in self.active.items() if now <= expiry}
        strict_pocket = self.pocket_recovery and self.strict_concealment
        recover_compressed = strict_pocket and self.recover_compressed
        reacquire_score = .42 if recover_compressed else .45
        carry_minimum = 3 if recover_compressed else 4
        carry_deadline = 5.35 if recover_compressed else 5.
        minimum_score = .29 if recover_compressed else .30
        visible_hands = {(track['track_id'], wrist) for track in tracks
                         if track.get('observed', True) and track.get('keypoints') is not None
                         for wrist in (9, 10) if track['keypoints'][wrist, 2] >= .55} if strict_pocket else set()
        previous_keys = set(self.tracks)
        self.tracks = {key: state for key, state in self.tracks.items()
                       if now - state.last <= 1.34 or
                       (key in visible_hands and state.hits >= carry_minimum and state.last_item_hip_distance is not None
                        and state.last_item_hip_distance >= .23 and now - state.last <= carry_deadline)}
        expired = sorted(previous_keys - self.tracks.keys())[:32]
        recovery_hits = 2 if self.continuous_carry else 3
        recovery_gap = .50 if self.continuous_carry else .70
        candidates = []
        weak_candidates = []
        hands = {}
        rejections = dict.fromkeys(("distance", "score", "drift", "appearance", "continuity"), 0)
        for track in tracks:
            if not track.get('observed', True) or track.get('keypoints') is None:
                continue
            tid, pts = track['track_id'], track['keypoints']
            scale = max(1, track['bbox'][3]-track['bbox'][1])
            for wrist, hip in ((9, 11), (10, 12)):
                if pts[wrist, 2] < .3:
                    continue
                key = (tid, wrist)
                x, y = pts[wrist, :2]
                hip_distance = hypot(x-pts[hip, 0], y-pts[hip, 1])/scale if pts[hip, 2] >= .3 else None
                hip_center = pts[hip, :2] if hip_distance is not None else None
                hands[key] = (x, y, scale, hip_distance, hip_center, pts[wrist, 2] >= .55)
                for index, (box, score, appearance) in enumerate(detections):
                    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                    offset = ((cx-x)/scale, (cy-y)/scale)
                    distance = hypot(*offset)
                    if distance > .22 or score < minimum_score:
                        rejections["distance" if distance > .22 else "score"] += 1
                        continue
                    state = self.tracks.get(key)
                    drift = hypot(offset[0]-state.offset[0], offset[1]-state.offset[1]) if state else 0.
                    if state and drift > .19:
                        rejections["drift"] += 1
                        continue
                    visual = float(cv2.compareHist(state.appearance, appearance, cv2.HISTCMP_BHATTACHARYYA)) if state else 0.
                    latest_visual = (float(cv2.compareHist(state.last_appearance, appearance, cv2.HISTCMP_BHATTACHARYYA))
                                     if recover_compressed and state and state.last_appearance is not None else visual)
                    if visual > .80:
                        rejections["appearance"] += 1
                        continue
                    overlap = 0.
                    pocket_reacquired = False
                    resumed_carry = False
                    if state:
                        dx,dy=x-state.wrist[0],y-state.wrist[1]
                        predicted = tuple(c + (dx if j%2==0 else dy) for j,c in enumerate(state.box))
                        overlap = intersection_over_union(predicted,box)
                        if self.pocket_recovery and hip_center is not None:
                            item_hip = hypot(cx-hip_center[0], cy-hip_center[1])/scale
                            prior_area = (state.box[2]-state.box[0])*(state.box[3]-state.box[1])
                            area = (box[2]-box[0])*(box[3]-box[1])
                            pocket_reacquired = (
                                state.hits >= recovery_hits and recovery_gap < now-state.last <= 1.06
                                and score >= .45 and pts[wrist, 2] >= .55
                                and distance <= .13 and drift <= .14 and visual <= .58
                                and hip_distance <= .22 and item_hip <= .15
                                and state.last_item_hip_distance is not None
                                and (state.last_item_hip_distance-item_hip >= .15
                                     or (self.continuous_carry and state.last_item_hip_distance >= .17
                                         and state.maximum_item_hip_distance-item_hip >= .16))
                                and .08*prior_area <= area <= .65*prior_area
                            )
                            resumed_carry = (
                                strict_pocket and state.hits >= carry_minimum and 1.34 < now - state.last <= carry_deadline
                                and state.last_item_hip_distance is not None
                                and state.last_item_hip_distance >= .23 and item_hip <= .17
                                and hip_distance <= .19 and distance <= .13
                                and score >= reacquire_score and min(visual, latest_visual) <= .55 and pts[wrist, 2] >= .70
                                and .50 * prior_area <= area <= 2. * prior_area
                            )
                        if state.bridge_frames and overlap < .12 and visual > .45 and not (pocket_reacquired or resumed_carry):
                            rejections["continuity"] += 1
                            continue
                    supported_carry = (
                        recover_compressed and state is not None and state.hits >= 2 and now - state.last <= .70
                        and pts[wrist, 2] >= .70 and distance <= .14 and drift <= .10
                        and latest_visual <= (.33 if score < .33 else .40) and hip_center is not None
                        and hypot(cx - hip_center[0], cy - hip_center[1]) / scale >= .23
                    )
                    if score < .35 and not supported_carry:
                        # 신뢰도가 낮은 검출은 같은 손에서 최근에 확인한 상품의 궤적만 이어 준다.
                        # 새 추적을 시작하거나 검출 횟수 및 마지막 확실한 검출 시각을 갱신하지 않는다.
                        if (score >= .30 and state and now-state.last <= .70 and state.bridge_frames < 2
                                and distance <= .12 and drift <= .08 and visual <= .45):
                            weak_candidates.append((distance+drift+visual, key, index))
                        else:
                            rejections["score"] += 1
                        continue
                    # 한 프레임에서 다른 손목에 연결하면 정상적인 상품 궤적이 둘로 갈릴 수 있다.
                    # 외관이 일치하는지 확인하면서 확실한 기존 연결을 우선한다.
                    continuation = .07 if (state and state.hits >= 2 and now-state.last <= .70
                                            and (overlap >= .25 or visual <= .35)) else 0.
                    wrist_penalty = 0.
                    if self.continuous_carry:
                        wrist_penalty = .08 * (1. - float(pts[wrist, 2]))
                        if (state and now-state.last <= .70 and pts[wrist, 2] >= .55
                                and overlap >= .25 and visual <= .45):
                            continuation = .10
                    recovery_preference = .18 if pocket_reacquired or resumed_carry else 0.
                    cost = (distance + .7*drift + .16*visual - .10*score - .10*overlap
                            - continuation - recovery_preference + wrist_penalty)
                    candidates.append((cost, key, index, (cx,cy), offset, pocket_reacquired, resumed_carry))
        matches, used = {}, set()
        for _, key, index, center, offset, pocket_reacquired, resumed_carry in sorted(candidates):
            if key not in matches and index not in used:
                used.add(index)
                matches[key] = (center, offset, detections[index][2], detections[index][0], pocket_reacquired, resumed_carry)
        weak_matches = {}
        for _, key, index in sorted(weak_candidates):
            if key not in matches and key not in weak_matches and index not in used:
                used.add(index)
                weak_matches[key] = detections[index][0]
        hand_states = []
        for key, (x, y, scale, hip_distance, hip_center, reliable_wrist) in hands.items():
            state = self.tracks.get(key)
            hip_limit = .15 if strict_pocket else .24
            near_hip = (hip_distance is not None and hip_distance <= hip_limit
                        and (not strict_pocket or reliable_wrist))
            trace = {"track_id": key[0], "wrist": key[1], "near_hip": bool(near_hip),
                     "result": "unlinked"}
            if len(hand_states) < 32:
                hand_states.append(trace)
            if key in matches:
                trace["result"] = "linked"
                center, offset, appearance, box, pocket_reacquired, resumed_carry = matches[key]
                item_hip = hypot(center[0]-hip_center[0],center[1]-hip_center[1])/scale if hip_center is not None else None
                if state is None or (not (pocket_reacquired or resumed_carry)
                                     and now-state.last > (.70 if not state.bridge_frames else 1.06)):
                    self.tracks[key] = Carry(center, offset, now, appearance, box, (x,y),
                                            hip_distance, hip_distance, slot, item_hip, item_hip,
                                            item_hip, int(item_hip is not None and item_hip <= .10),
                                            last_appearance=appearance)
                    continue
                if state.bridge_frames:
                    self.reacquired_tracks += 1
                state.offset, state.last = offset, now
                state.box, state.wrist = box, (x,y)
                state.box_slot = slot
                state.bridge_frames = 0
                state.pocket_reacquired |= pocket_reacquired
                if resumed_carry:
                    state.resumed_close_hits = 0
                if state.resumed_close_hits is not None:
                    state.resumed_close_hits = (state.resumed_close_hits + 1 if item_hip is not None and item_hip <= .17 else 0)
                state.appearance = .75 * state.appearance + .25 * appearance
                state.last_appearance = appearance
                state.hits += 1
                state.distance = max(state.distance, hypot(center[0]-state.origin[0], center[1]-state.origin[1])/scale)
                if hip_distance is not None:
                    state.maximum_hip_distance = max(state.maximum_hip_distance or 0.,hip_distance)
                state.last_hip_distance = hip_distance
                if item_hip is not None:
                    state.maximum_item_hip_distance = max(state.maximum_item_hip_distance or 0.,item_hip)
                    state.minimum_item_hip_distance = (item_hip if state.minimum_item_hip_distance is None
                                                      else min(state.minimum_item_hip_distance, item_hip))
                state.close_item_hits = (state.close_item_hits + 1 if item_hip is not None and item_hip <= .10
                                         else 0)
                state.last_item_hip_distance = item_hip
            elif key in weak_matches:
                trace["result"] = "weak_link"
                state.box, state.wrist = weak_matches[key], (x, y)
                state.box_slot = slot
                state.bridge_frames += 1
                self.weak_bridged_frames += 1
            elif state:
                short_gap = state.bridge_frames < 2 and now-state.last <= .70
                verify_visibility = state.bridge_frames >= 2
                if (previous_gray is not None and gray is not None and state.box_slot == slot-1
                        and (short_gap or verify_visibility)):
                    shifted = follow_box(previous_gray,gray,state.box,(x-state.wrist[0],y-state.wrist[1]),scale)
                    if shifted is not None and verify_visibility:
                        # 상품이 사라진 뒤에도 손목의 이동만으로 추정 위치가 손을 따라갈 수 있다.
                        # 짧은 연결 허용 시간을 넘으면 실제 검출에서 상품의 외관을 확인한다.
                        appearance = article_appearance(frame, shifted)
                        if (appearance is None or
                                cv2.compareHist(state.appearance, appearance, cv2.HISTCMP_BHATTACHARYYA) > .35):
                            shifted = None
                    if shifted is not None:
                        trace["result"] = "optical_flow_bridge"
                        state.box, state.wrist = shifted,(x,y)
                        state.box_slot = slot
                        state.bridge_frames += 1
                        self.bridged_frames += 1
                        continue
                wrist_approached = (state.last_hip_distance is not None and state.last_hip_distance <= .28
                                    and state.maximum_hip_distance - state.last_hip_distance >= .12)
                item_approached = (state.hits >= 4 and state.last_item_hip_distance is not None
                                   and state.last_item_hip_distance <= .15
                                   and state.maximum_item_hip_distance-state.last_item_hip_distance >= .10)
                # 상품의 절대 이동 거리가 작거나 골반 쪽 손목 좌표가 흔들려도
                # 인물에 대한 상대 위치의 변화로 짧은 이동을 확인할 수 있다.
                minimum_close_hits = 3 if self.strict_concealment else 2
                short_item_carry = (state.hits >= 4 and state.close_item_hits >= minimum_close_hits
                                    and state.distance >= .12
                                    and state.maximum_item_hip_distance is not None
                                    and state.minimum_item_hip_distance is not None
                                    and state.maximum_item_hip_distance-state.minimum_item_hip_distance >= .10)
                conventional_carry = (state.hits >= 3 and state.distance >= .20
                                      and (wrist_approached or item_approached))
                if self.pocket_recovery:
                    # 주머니 근처에서 상품의 일부가 같은 손에 다시 작게 보였다.
                    # 손목이 주변에 있는 동안 이 상품이 다시 사라지는지 확인한다.
                    concealment = (
                        state.pocket_reacquired and state.hits > recovery_hits and state.distance >= .20
                        and state.last_item_hip_distance is not None
                        and state.last_item_hip_distance <= .15
                        and state.maximum_item_hip_distance-state.last_item_hip_distance >= .16
                        and slot-state.box_slot >= 1
                    )
                    if self.continuous_carry:
                        concealment |= (
                            state.hits >= 5 and state.distance >= .14 and now-state.last >= .55
                            and state.last_item_hip_distance is not None
                            and state.last_item_hip_distance <= .12
                            and state.maximum_item_hip_distance-state.last_item_hip_distance >= .12
                        )
                    if state.resumed_close_hits is not None:
                        concealment = (
                            state.resumed_close_hits >= 3 and now - state.last >= .55
                            and state.last_item_hip_distance is not None and state.last_item_hip_distance <= .14
                        )
                else:
                    concealment = now-state.last >= .55 and (conventional_carry or short_item_carry)
                    if self.strict_concealment:
                        concealment &= (state.last_item_hip_distance is not None
                                        and state.last_item_hip_distance <= .12)
                if near_hip and concealment:
                    self.active[key[0]] = now + 2.0
                    self.tracks.pop(key)
                trace.update(result="confirmed" if near_hip and concealment else "concealment_rejected",
                             carry_conditions=bool(concealment), hits=state.hits,
                             travel=round(state.distance, 3), gap_seconds=round(now-state.last, 3),
                             pocket_reacquired=bool(state.pocket_reacquired),
                             last_item_hip_distance=(None if state.last_item_hip_distance is None
                                                     else round(state.last_item_hip_distance, 3)))
        self.trace = {"slot": slot, "detections": len(detections), "visible_hands": len(hands),
                      "top_scores": sorted((round(float(score), 3) for _, score, _ in detections), reverse=True)[:8],
                      "strong_candidates": len(candidates), "linked": len(matches), "weak_links": len(weak_matches),
                      "rejections": rejections, "expired_hands": expired, "hands": hand_states}
        return dict(self.active)


class ArticleDetector:
    def __init__(self, path):
        from ultralytics import YOLO
        from core.detector import fp16_predict_options
        import torch
        self.model = YOLO(str(path))
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.size = int(self.model.overrides.get('imgsz',160))
        self.fast_precision = fp16_predict_options()

    def detect(self, frame, tracks, *, compact=False):
        return self.detect_batch([frame], [tracks], compact=compact)[0]

    def detect_batch(self, frames, tracks_per_frame, *, compact=False):
        views = ((compact, .60 if compact else 1.),)
        return self._detect_views(frames, tracks_per_frame, views)[compact]

    def warmup_live(self):
        if self.device == 0:
            crop = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            self._predict([crop] * 6, fast=True)

    def detect_with_compact_views(self, frame, tracks):
        views = ((False, 1.), (True, .60), (True, .55))
        result = self._detect_views([frame], [tracks], views, fast=True)
        return result[False][0], result[True][0]

    def _detect_views(self, frames, tracks_per_frame, views, *, fast=False):
        rois = []
        for i, (frame, tracks) in enumerate(zip(frames, tracks_per_frame)):
            h, w = frame.shape[:2]
            for track in tracks:
                pts = track.get('keypoints')
                if pts is None or not track.get('observed', True):
                    continue
                scale = max(1, track['bbox'][3]-track['bbox'][1])
                for wrist, elbow in ((9, 7), (10, 8)):
                    if pts[wrist, 2] < .3:
                        continue
                    seen = set()
                    for compact, radius_factor in views:
                        elbow_point = pts[elbow] if not compact and pts[elbow, 2] >= .3 else None
                        box = wrist_roi(pts[wrist], elbow_point, scale * radius_factor, w, h)
                        left, top, right, bottom = box
                        key = (compact, box)
                        if key in seen or right-left < 16 or bottom-top < 16:
                            continue
                        seen.add(key)
                        crop = cv2.resize(frame[top:bottom, left:right], (self.size, self.size))
                        rois.append((compact, i, box, crop))
        all_boxes = {compact: [[] for _ in frames] for compact, _ in views}
        if not rois:
            return all_boxes
        predictions = self._predict([roi[-1] for roi in rois], fast=fast)
        for (compact, i, (left, top, right, bottom), _), result in zip(rois, predictions):
            sx, sy = (right-left)/self.size, (bottom-top)/self.size
            for x1, y1, x2, y2, score, _ in result.boxes.data.cpu().tolist():
                box = (left+sx*x1, top+sy*y1, left+sx*x2, top+sy*y2)
                all_boxes[compact][i].append((box, score))
        for groups in all_boxes.values():
            for i, (frame, boxes) in enumerate(zip(frames, groups)):
                kept = []
                for box, score in sorted(boxes, key=lambda item: item[1], reverse=True):
                    if any(intersection_over_union(box, other) > .45 for other, _, _ in kept):
                        continue
                    appearance = article_appearance(frame, box)
                    if appearance is not None:
                        kept.append((box, score, appearance))
                groups[i] = kept
        return all_boxes

    def _predict(self, crops, *, fast):
        precision = self.fast_precision if fast and self.device == 0 else {}
        return self.model.predict(crops, imgsz=self.size, conf=.25, device=self.device,
                                  verbose=False, **precision)


def intersection_over_union(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union else 0.


def hand_near_face(track):
    points = track.get('keypoints')
    if points is None or points[0,2] < .4:
        return False
    scale = max(1,track['bbox'][3]-track['bbox'][1])
    return any(points[wrist,2] >= .4 and np.linalg.norm(points[wrist,:2]-points[0,:2]) <= .18*scale
               for wrist in (9,10))


class SmokingMotionEvidence:
    """같은 사람을 서로 다른 실시간 프레임에서 관측해 손과 얼굴의 접촉 동작을 확인한다."""

    def __init__(self):
        self.slot = None
        self.last_near = {}
        self.confirmed_at = {}

    def reset(self):
        self.slot = None
        self.last_near.clear()
        self.confirmed_at.clear()

    def update(self, tracks, now):
        slot = int(now * 3 + 1e-6)
        if self.slot is not None and slot < self.slot:
            self.reset()
        if slot != self.slot:
            current = {track["track_id"] for track in tracks if hand_near_face(track)}
            self.confirmed_at = {tid: stamp for tid, stamp in self.confirmed_at.items() if now-stamp <= 3.}
            for tid in current:
                if now-self.last_near.get(tid, -float("inf")) <= .70:
                    self.confirmed_at[tid] = now
            self.last_near = dict.fromkeys(current, now)
            self.slot = slot
        return self.confirmed_at.keys()


def applicable_article_evidence(predictions, candidates, tracks):
    """상품 은닉 후보를 반영할 때 흡연 이외의 이상행동 예측은 유지한다."""
    applicable = set()
    for track in tracks:
        tid = track['track_id']
        if tid not in candidates:
            continue
        label = predictions.get(tid, (0, 0))[0]
        if label not in (0, 4):
            continue
        # 은닉 경보는 잠시 유지된다. 이때 새롭게 관측된 손과 얼굴의 근접 동작은
        # 은닉 경보와 동시에 나타날 수 있으며 흡연 추론의 근거가 된다.
        if label == 4 and hand_near_face(track):
            continue
        applicable.add(tid)
    return applicable


class ArticleSmokingGate:
    """은닉 근거가 만료된 뒤에도 이미 보류한 흡연 결과를 다시 사용하지 않는다."""
    def __init__(self):
        self.blocked = {}

    def update(self, predictions, article_suspects, tracks, origins, threshold):
        self.blocked = {tid: revision for tid, revision in self.blocked.items() if tid in origins}
        deferred = set()
        for track in tracks:
            tid = track['track_id']
            prediction = predictions.get(tid)
            origin = origins.get(tid)
            if prediction is None or origin is None:
                continue
            label, _ = prediction
            revision, raw_label, raw_score = origin
            if label != 4:
                self.blocked.pop(tid, None)
            elif tid in article_suspects:
                self.blocked[tid] = revision
            elif tid in self.blocked:
                # 지수 이동 평균에 이전 흡연 점수가 한동안 남을 수 있다.
                # 새 추론 창과 현재 손·얼굴 근접 근거가 모두 있을 때만 흡연 판정을 복원한다.
                if (revision > self.blocked[tid] and raw_label == 4 and raw_score >= threshold
                        and hand_near_face(track)):
                    self.blocked.pop(tid)
                else:
                    deferred.add(tid)
        return deferred
