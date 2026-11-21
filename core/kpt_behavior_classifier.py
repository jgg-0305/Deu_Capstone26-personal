"""키포인트 기반 이상행동 분류 — KeypointLSTM + 트랙별 슬라이딩 윈도우 버퍼.

YOLO pose 가 추출한 17관절 좌표 시퀀스를 입력으로 받아
인물별로 이상행동을 분류한다.

파이프라인:
    YOLO pose → (17, 3) keypoints per person
    → _encode : bbox 상대 정규화 → (34,) 벡터
    → 30프레임 deque (track_id별)
    → KeypointLSTM → (class_id, confidence)

학습된 가중치 포맷: plain state_dict() (module. 접두사 없음)
    torch.save(model.state_dict(), 'kpt_behavior.pth')
새 가중치는 state_dict와 input_config를 함께 저장할 수 있다.
sample_fps가 있으면 웹의 영상 타임스탬프로 학습 시점의 관절 간격을 유지한다.
"""

from collections import deque, OrderedDict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_KPT_CONF_THR = 0.3   # 이 값 미만 관절은 (0, 0) 처리


class KeypointLSTM(nn.Module):
    """경량 키포인트 시퀀스 → 행동 분류기 (v1 — 구버전 체크포인트 호환용).

    입력 : (B, T, 34)  — T 프레임, 17관절 × (x, y) 정규화 좌표
    출력 : (B, num_classes)
    """

    def __init__(self, input_size: int = 34, hidden: int = 128,
                 num_layers: int = 2, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])  # 마지막 레이어의 hidden state


class KeypointLSTMv2(nn.Module):
    """v2 — 속도 특징 + BiLSTM + attention pooling.

    train/1_behavior/model_kpt.py 의 KeypointLSTMv2 와 동일 구조.
    입력은 v1과 같은 (B, T, 34) — 속도는 forward 내부에서 계산.
    """

    def __init__(self, input_size: int = 34, hidden: int = 128,
                 num_layers: int = 2, num_classes: int = 8, dropout: float = 0.5):
        super().__init__()
        feat = input_size * 2
        self.norm = nn.LayerNorm(feat)
        self.lstm = nn.LSTM(
            input_size=feat,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nz = (x != 0).float()
        vel = (x[:, 1:] - x[:, :-1]) * nz[:, 1:] * nz[:, :-1]
        vel = F.pad(vel, (0, 0, 1, 0))
        f = self.norm(torch.cat([x, vel], dim=2))
        out, _ = self.lstm(f)
        w = torch.softmax(self.attn(out), dim=1)
        return self.head((w * out).sum(dim=1))


def _encode(kpts, bbox, conf_thr: float = _KPT_CONF_THR) -> np.ndarray:
    """(17, 3) keypoints → (34,) bbox-상대 정규화 벡터.

    x' = (kx - x1) / bbox_w,  y' = (ky - y1) / bbox_h
    신뢰도 미달 관절은 (0, 0).
    """
    x1, y1, x2, y2 = bbox
    bw, bh = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
    vec = np.zeros(34, dtype=np.float32)
    if kpts is not None:
        for i, (kx, ky, kc) in enumerate(kpts):
            if float(kc) >= conf_thr:
                vec[i * 2]     = (float(kx) - x1) / bw
                vec[i * 2 + 1] = (float(ky) - y1) / bh
    return vec


def _observed_low_reach(sequence, frames):
    """Check whether either visible hand recently reached below its hip."""
    for vector in list(sequence)[-frames:]:
        pose = vector[:34].reshape(17, 2)
        wrists, hips = pose[9:11], pose[11:13]
        visible = wrists.any(axis=1) & hips.any(axis=1)
        if np.any(visible & (wrists[:, 1] - hips[:, 1] >= .12)):
            return True
    return False


def _near_another_person(track, tracks):
    x1, y1, x2, y2 = track["bbox"]
    height = max(float(y2 - y1), 1.0)
    for other in tracks:
        if other["track_id"] == track["track_id"] or not other.get("observed", True):
            continue
        ox1, oy1, ox2, oy2 = other["bbox"]
        distance = math.hypot((x1 + x2 - ox1 - ox2) / 2, (y1 + y2 - oy1 - oy2) / 2)
        if distance <= .75 * height:
            return True
    return False


class TrackBehaviorBuffer:
    """트랙 ID별 키포인트 시퀀스를 관리하고 행동 추론을 실행한다.

    매 프레임 update(tracks, kpt_map) 를 호출하면
    {track_id: (class_id, confidence)} 딕셔너리를 반환한다.
    버퍼가 아직 seq_len 을 채우지 못한 트랙은 결과 없음.
    """

    def __init__(self, model_path: str, seq_len: int = 30, stride: int = 5,
                 device=None, ema_alpha: float = 0.6, abn_conf_thr: float = 0.5,
                 max_missing: int = 30, max_gap: int = 3, *,
                 theft_confirmation_threshold: float | None = None,
                 verify_live_vandalism: bool = False):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        # 예측 확률 시간 스무딩 계수 (이전 EMA 가중치) — 깜빡임·순간 오탐 억제
        self.ema_alpha = ema_alpha
        # 스무딩된 이상행동 확률이 이 값 미만이면 정상으로 간주 (오경보 억제)
        self.abn_conf_thr = abn_conf_thr
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = self._build_and_load(model_path)
        self.model.to(self.device).eval()
        # New checkpoints declare their temporal input contract. Legacy plain
        # state_dict files retain the original per-call behavior.
        self.input_config = getattr(self.model, "behavior_input_config", {})
        self.sample_fps = self.input_config.get("sample_fps")
        self.confirmation_threshold = (
            theft_confirmation_threshold if self.sample_fps is not None else None
        )
        self.confirmation_labels = {5, 6} if self.input_config.get("confirm_abandonment") is True else {6}
        self.verify_live_vandalism = verify_live_vandalism and self.confirmation_threshold is not None
        if self.verify_live_vandalism:
            self.confirmation_labels.add(2)
        if self.input_config.get("confirm_assault") is True:
            self.confirmation_labels.add(7)
        self.arson_specialist = getattr(self.model, "arson_specialist", None)
        self.feature_size = 34
        self.context_encoder = None
        if self.input_config.get("architecture") == "hand_fusion_v1":
            from core.hand_behavior import INPUT_SIZE, encode_context
            self.feature_size = INPUT_SIZE
            self.context_encoder = encode_context
        self._sample_index = None

        # 트랙별 34차원 관절 특징 버퍼
        self._bufs: dict[int, deque] = {}
        # track_id → frame 카운터
        self._cnts: dict[int, int] = {}
        # track_id → EMA 확률 벡터 (num_classes,)
        self._probs: dict[int, np.ndarray] = {}
        self._arson_probs: dict[int, np.ndarray] = {}
        # 트랙별 행동 클래스와 신뢰도
        self._results: dict[int, tuple] = {}
        # Per-person inference provenance. Reusing a label between windows
        # must not make it look like new evidence to event arbitration.
        self.inference_origins: dict[int, tuple[int, int, float]] = {}
        self.inference_count = 0
        self._previous_raw_action: dict[int, tuple[int, float]] = {}
        self._confirmed_action: dict[int, tuple[int, float]] = {}
        self._pending_vandalism = set()
        self.max_missing = max_missing
        self.max_gap = max_gap
        self._missing: dict[int, int] = {}
        self.reset_count = 0

    @staticmethod
    def _build_and_load(path: str) -> nn.Module:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # plain state_dict 또는 wrapper dict 둘 다 처리
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # DataParallel 키 접두사 제거
        state = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v)
            for k, v in state.items()
        )
        # 체크포인트 구조로 모델 버전 자동 감지 (v2는 attn 레이어 보유)
        config = ckpt.get("input_config", {}) if "state_dict" in ckpt else {}
        if config:
            expected_labels = ["normal", "fall", "vandalism", "arson", "smoking", "abandon", "theft", "assault"]
            if config.get("labels") != expected_labels or config.get("sequence_length") != 30:
                raise ValueError("행동 모델의 클래스 순서 또는 시퀀스 길이가 앱과 다릅니다.")
            fps = config.get("sample_fps")
            if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("행동 모델의 학습 FPS가 올바르지 않습니다.")
        architecture = config.get("architecture")
        if architecture == "hand_fusion_v1":
            from core.hand_behavior import HandFusion
            model = HandFusion()
        elif architecture is None:
            model = KeypointLSTMv2() if "attn.weight" in state else KeypointLSTM()
        else:
            raise ValueError("지원하지 않는 행동 모델 구조입니다.")
        model.load_state_dict(state)
        if "arson_specialist_state_dict" in ckpt:
            if architecture != "hand_fusion_v1":
                raise ValueError("방화 보조 가중치는 HandFusion 모델에서만 지원합니다.")
            model.arson_specialist = HandFusion()
            model.arson_specialist.load_state_dict(ckpt["arson_specialist_state_dict"])
        model.behavior_input_config = config
        return model

    def update(self, tracks: list, kpt_map: dict, source_seconds=None, *, frame=None, features=None) -> dict:
        """tracks, kpt_map 을 받아 결과 딕셔너리를 반환·갱신한다."""
        if self.sample_fps is not None and source_seconds is not None:
            if not math.isfinite(source_seconds) or source_seconds < 0:
                raise ValueError("영상 시간은 유효한 0 이상의 초 단위 값이어야 합니다.")
            index = math.floor(source_seconds * self.sample_fps + 1e-6)
            if self._sample_index is not None:
                if index < self._sample_index:
                    # Seeking backwards starts a fresh temporal sequence.
                    for tid in list(self._bufs):
                        self._forget(tid)
                elif index == self._sample_index:
                    visible = {t["track_id"] for t in tracks if t.get("observed", True)
                               and (_encode(kpt_map.get(t["track_id"]), t["bbox"]) if features is None
                                    else features[t["track_id"]][:34]).any()}
                    return {tid: self._confirmed_action.get(tid, value)
                            for tid, value in self._results.items() if tid in visible
                            and sum(bool(item.any()) for item in self._bufs[tid]) >= self.seq_len * .5}
                else:
                    # Missing source-time slots must not compress a long gap
                    # into apparently adjacent motions.
                    gap = min(index - self._sample_index - 1, self.max_missing + 1)
                    for _ in range(gap):
                        self._advance([], {})
            self._sample_index = index
        return self._advance(tracks, kpt_map, frame=frame, features=features)

    def _advance(self, tracks: list, kpt_map: dict, *, frame=None, features=None) -> dict:
        observed_ids = set()
        for t in tracks:
            tid = t["track_id"]
            points = kpt_map.get(tid)
            vec = _encode(points, t["bbox"]) if features is None else features[tid]

            if not t.get("observed", True) or not vec[:34].any():
                continue
            observed_ids.add(tid)
            if self._missing.get(tid, 0) > self.max_gap:
                # 긴 가림 전후의 동작을 하나의 연속 시퀀스로 오인하지 않는다.
                self._forget(tid)

            if tid not in self._bufs:
                self._bufs[tid] = deque(maxlen=self.seq_len)
                self._cnts[tid] = 0
            self._missing[tid] = 0

            if self.context_encoder and features is None:
                vec = self.context_encoder(frame, points, t["bbox"], vec)
            self._bufs[tid].append(vec)
            self._cnts[tid] += 1

            # 버퍼가 충분히 쌓이고 stride 조건을 만족하면 추론
            if (len(self._bufs[tid]) == self.seq_len
                    and self._cnts[tid] % self.stride == 0
                    and sum(bool(item.any()) for item in self._bufs[tid]) >= self.seq_len * 0.5):
                probs = self._infer(self._bufs[tid])
                self.inference_count += 1
                raw_label = int(probs.argmax())
                raw_score = float(probs[raw_label])
                self.inference_origins[tid] = (self.inference_count, raw_label, raw_score)
                # 시간 EMA 스무딩 — 단발성 오탐·라벨 깜빡임 억제
                prev = self._probs.get(tid)
                if prev is not None:
                    probs = self.ema_alpha * prev + (1.0 - self.ema_alpha) * probs
                self._probs[tid] = probs

                cls = int(probs.argmax())
                conf = float(probs[cls])
                # 이상행동인데 스무딩 확률이 임계값 미만 → 정상으로 보고 (오경보 억제)
                if cls != 0 and conf < self.abn_conf_thr:
                    cls, conf = 0, float(probs[0])
                # Confirm across two fresh windows without modifying the EMA history
                # or reusing a prediction repeated on intervening video frames.
                self._confirmed_action.pop(tid, None)
                self._pending_vandalism.discard(tid)
                if self.confirmation_threshold is not None:
                    previous_label, previous_score = self._previous_raw_action.get(tid, (None, 0.))
                    self._previous_raw_action[tid] = (raw_label, raw_score)
                    confirmed_score = min(previous_score, raw_score)
                    if (self.verify_live_vandalism and cls == 2
                            and (raw_label != 2 or previous_label != 2 or confirmed_score < self.confirmation_threshold)):
                        self._pending_vandalism.add(tid)
                        cls, conf = 0, float(probs[0])
                    if (cls == 0 and raw_label == previous_label and raw_label in self.confirmation_labels
                            and confirmed_score >= self.confirmation_threshold
                            and (raw_label != 5 or _observed_low_reach(self._bufs[tid], 2 * self.stride - 1))
                            and (raw_label != 7 or _near_another_person(t, tracks))):
                        self._confirmed_action[tid] = (raw_label, confirmed_score)
                if self.arson_specialist is not None:
                    specialist = self._infer_model(self._bufs[tid], self.arson_specialist)
                    previous = self._arson_probs.get(tid)
                    if previous is not None:
                        specialist = self.ema_alpha * previous + (1.0 - self.ema_alpha) * specialist
                    self._arson_probs[tid] = specialist
                    # Use the former model's arson score only when the retrained
                    # model independently favors arson and has no confirmed action.
                    if (cls == 0 and tid not in self._confirmed_action and int(probs.argmax()) == 3 and probs[3] >= .5
                            and specialist[3] >= max(.90, self.abn_conf_thr, self.confirmation_threshold or 0)):
                        self._confirmed_action[tid] = (3, float(specialist[3]))
                self._results[tid] = (cls, conf)

        # 짧은 누락은 학습 추출기와 같은 0 벡터로 표시하고 수집 기록을 보존한다.
        # 실제로 관찰하지 못한 시점에는 새 행동 판정이나 사건을 생성하지 않는다.
        for tid in list(self._bufs):
            if tid not in observed_ids:
                self._previous_raw_action.pop(tid, None)
                self._confirmed_action.pop(tid, None)
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self.max_missing:
                    self._forget(tid)
                else:
                    self._bufs[tid].append(np.zeros(self.feature_size, dtype=np.float32))
                    self._cnts[tid] += 1

        return {tid: self._confirmed_action.get(tid, value) for tid, value in self._results.items()
                if tid in observed_ids and sum(bool(item.any()) for item in self._bufs[tid]) >= self.seq_len * 0.5}

    def _forget(self, tid):
        self._bufs.pop(tid, None)
        self._cnts.pop(tid, None)
        self._probs.pop(tid, None)
        self._arson_probs.pop(tid, None)
        self._results.pop(tid, None)
        self._previous_raw_action.pop(tid, None)
        self._confirmed_action.pop(tid, None)
        self._pending_vandalism.discard(tid)
        self.inference_origins.pop(tid, None)
        self._missing.pop(tid, None)
        self.reset_count += 1

    def reset(self):
        """Discard temporal state when a camera opens a new connection."""
        for tid in list(self._bufs):
            self._forget(tid)
        self._sample_index = None

    def progress(self, track_id):
        """추론 조건을 바꾸지 않고 해당 인물의 관절 수집 진행도를 반환한다."""
        buf = self._bufs.get(track_id, ())
        missing = self._missing.get(track_id, 0)
        valid = sum(bool(item.any()) for item in buf)
        probs = self._probs.get(track_id)
        candidate = None if probs is None else {"label": int(probs.argmax()), "score": float(probs.max())}
        state = "tracking_gap" if missing else "collecting" if len(buf) < self.seq_len else (
            "insufficient_keypoints" if valid < self.seq_len * 0.5 else "classified" if track_id in self._results else "collecting")
        theft_score = None if probs is None else float(probs[6])
        confirmed = self._confirmed_action.get(track_id)
        if state == "classified" and confirmed is not None:
            confirmed_label, confirmed_score = confirmed
            candidate = {"label": confirmed_label, "score": confirmed_score}
            if confirmed_label == 6:
                theft_score = confirmed_score
        # A weak normal argmax is also uncertain, not evidence that nothing happened.
        if state == "classified" and (candidate["score"] < self.abn_conf_thr or track_id in self._pending_vandalism):
            state = "uncertain"
        return {"collected": len(buf), "required": self.seq_len,
                "valid": valid, "missing_frames": missing, "state": state, "candidate": candidate,
                "theft_score": theft_score}

    def rule_candidates(self, visible_predictions, minimum):
        """Use the independently calibrated rule arbitration cutoff."""
        candidates = {}
        for tid in visible_predictions:
            prob = self._probs.get(tid)
            if prob is not None:
                label = int(prob.argmax())
                score = float(prob[label])
                candidates[tid] = (label if score >= minimum else 0, score)
        return candidates

    @torch.no_grad()
    def _infer(self, buf: deque) -> np.ndarray:
        return self._infer_model(buf, self.model)

    @torch.no_grad()
    def _infer_model(self, buf: deque, model: nn.Module) -> np.ndarray:
        seq = torch.from_numpy(np.stack(list(buf))).unsqueeze(0).to(self.device, dtype=torch.float32)
        prob = torch.softmax(model(seq), dim=1)[0]
        return prob.cpu().numpy()
