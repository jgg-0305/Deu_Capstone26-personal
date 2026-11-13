# -*- coding: utf-8 -*-
"""규칙 기반 도난(절도) 감지.

규칙(사용자 정의):
  손(손목)이 '물품 가판대(stand)' 영역 위를 지나 물품을 든 뒤,
  다시 손이 '결제기(pos)' 영역 위로 오지 않으면 → 도난.

구현(트랙별 상태 기계):
  - 손목 키포인트(좌 9 / 우 10)가 stand bbox 안에 들어오면 '물품 집음(took)'.
  - took 이후 손목이 pos bbox 안에 들어오면 '결제(paid)' → 도난 아님.
  - took 이고 paid 가 아닌 채로 PAY_TIMEOUT 초 경과하거나 트랙이 사라지면 → 도난 확정(latched).

stand/pos 영역은 정적이므로 마지막으로 검출된 박스를 캐시해 사용한다.
"""

import time

_WRIST_IDS = (9, 10)        # COCO: 왼손목, 오른손목
_KPT_CONF = 0.3
TAKE_MIN_FRAMES = 2          # stand 위 손 검출이 이만큼 누적되면 '집음' 인정
PAY_TIMEOUT_SEC = 4.0        # 집은 뒤 이 시간 안에 POS 접촉 없으면 도난
LOST_FRAMES = 15             # 트랙이 이 프레임 수만큼 안 보이면 사라짐 처리


def _point_in_box(x, y, box, margin=0.0):
    x1, y1, x2, y2 = box
    mw = (x2 - x1) * margin
    mh = (y2 - y1) * margin
    return (x1 - mw) <= x <= (x2 + mw) and (y1 - mh) <= y <= (y2 + mh)


class TheftMonitor:
    def __init__(self, pay_timeout=PAY_TIMEOUT_SEC):
        self.pay_timeout = pay_timeout
        self._stand_boxes = []      # 캐시된 가판대 영역
        self._pos_boxes = []        # 캐시된 결제기 영역
        self._state = {}            # tid -> dict(took, took_time, paid, take_cnt, theft, last_seen_frame)
        self._frame = 0

    def _wrists(self, kpts):
        """키포인트(17,3)에서 신뢰도 충족 손목 좌표 리스트."""
        pts = []
        if kpts is None:
            return pts
        for i in _WRIST_IDS:
            if i < len(kpts) and kpts[i][2] >= _KPT_CONF:
                pts.append((float(kpts[i][0]), float(kpts[i][1])))
        return pts

    def update(self, tracks: list, kpt_map: dict, objects: list) -> dict:
        """매 프레임 호출. 반환: { track_id: True } (도난으로 판정된 트랙)."""
        self._frame += 1
        now = time.time()

        # stand/pos 영역 갱신 (정적이므로 검출되면 캐시)
        stands = [o["bbox"] for o in objects if o.get("cls") == 0]
        poss = [o["bbox"] for o in objects if o.get("cls") == 1]
        if stands:
            self._stand_boxes = stands
        if poss:
            self._pos_boxes = poss

        # stand·pos 영역이 모두 확보된 매장 장면에서만 도난 규칙 적용
        # (가판대/결제기가 없는 일반 이상행동 영상의 오탐 방지)
        if not self.has_regions:
            return {}

        present = set()
        for t in tracks:
            tid = t["track_id"]
            present.add(tid)
            st = self._state.setdefault(tid, dict(
                took=False, took_time=0.0, paid=False, take_cnt=0,
                released=False, theft=False, last_seen=self._frame))
            st["last_seen"] = self._frame

            wrists = self._wrists(kpt_map.get(tid))
            on_stand = any(_point_in_box(x, y, b)
                           for (x, y) in wrists for b in self._stand_boxes)
            on_pos = any(_point_in_box(x, y, b)
                         for (x, y) in wrists for b in self._pos_boxes)

            if on_stand:
                st["take_cnt"] += 1
                if st["take_cnt"] >= TAKE_MIN_FRAMES and not st["paid"]:
                    st["took"] = True
                    st["took_time"] = now
            elif st["took"]:
                # 집은 뒤 손이 가판대를 벗어남 = 물품을 들고 이동
                st["released"] = True
            if st["took"] and on_pos:
                st["paid"] = True
                st["theft"] = False

            # 도난 확정: 집고(released) 미결제이며 결제 제한시간 경과
            if (st["took"] and st["released"] and not st["paid"]
                    and (now - st["took_time"] > self.pay_timeout)):
                st["theft"] = True

        # 사라진 트랙: 집고 미결제면 도난 확정 후 잠시 유지
        for tid, st in list(self._state.items()):
            if tid not in present:
                if self._frame - st["last_seen"] > LOST_FRAMES:
                    if st["took"] and st["released"] and not st["paid"]:
                        st["theft"] = True
                    # 오래된 트랙 정리
                    if self._frame - st["last_seen"] > LOST_FRAMES * 20:
                        self._state.pop(tid, None)

        return {tid: True for tid, st in self._state.items()
                if st["theft"] and tid in present}

    @property
    def has_regions(self) -> bool:
        return bool(self._stand_boxes) and bool(self._pos_boxes)
