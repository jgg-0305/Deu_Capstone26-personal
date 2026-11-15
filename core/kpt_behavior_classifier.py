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


class TrackBehaviorBuffer:
    """트랙 ID별 키포인트 시퀀스를 관리하고 행동 추론을 실행한다.

    매 프레임 update(tracks, kpt_map) 를 호출하면
    {track_id: (class_id, confidence)} 딕셔너리를 반환한다.
    버퍼가 아직 seq_len 을 채우지 못한 트랙은 결과 없음.
    """

    def __init__(self, model_path: str, seq_len: int = 30, stride: int = 5,
                 device=None, ema_alpha: float = 0.6, abn_conf_thr: float = 0.5,
                 max_missing: int = 30, max_gap: int = 3):
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
        self._sample_index = None

        # 트랙별 34차원 관절 특징 버퍼
        self._bufs: dict[int, deque] = {}
        # track_id → frame 카운터
        self._cnts: dict[int, int] = {}
        # track_id → EMA 확률 벡터 (num_classes,)
        self._probs: dict[int, np.ndarray] = {}
        # 트랙별 행동 클래스와 신뢰도
        self._results: dict[int, tuple] = {}
        self.inference_count = 0
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
        model = KeypointLSTMv2() if "attn.weight" in state else KeypointLSTM()
        model.load_state_dict(state)
        config = ckpt.get("input_config", {}) if "state_dict" in ckpt else {}
        if config:
            expected_labels = ["normal", "fall", "vandalism", "arson", "smoking", "abandon", "theft", "assault"]
            if config.get("labels") != expected_labels or config.get("sequence_length") != 30:
                raise ValueError("행동 모델의 클래스 순서 또는 시퀀스 길이가 앱과 다릅니다.")
            fps = config.get("sample_fps")
            if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
                raise ValueError("행동 모델의 학습 FPS가 올바르지 않습니다.")
        model.behavior_input_config = config
        return model

    def update(self, tracks: list, kpt_map: dict, source_seconds=None) -> dict:
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
                               and _encode(kpt_map.get(t["track_id"]), t["bbox"]).any()}
                    return {tid: value for tid, value in self._results.items() if tid in visible
                            and sum(bool(item.any()) for item in self._bufs[tid]) >= self.seq_len * .5}
                else:
                    # Missing source-time slots must not compress a long gap
                    # into apparently adjacent motions.
                    gap = min(index - self._sample_index - 1, self.max_missing + 1)
                    for _ in range(gap):
                        self._advance([], {})
            self._sample_index = index
        return self._advance(tracks, kpt_map)

    def _advance(self, tracks: list, kpt_map: dict) -> dict:
        observed_ids = set()
        for t in tracks:
            tid = t["track_id"]
            vec = _encode(kpt_map.get(tid), t["bbox"])

            if not t.get("observed", True) or not vec.any():
                continue
            observed_ids.add(tid)
            if self._missing.get(tid, 0) > self.max_gap:
                # 긴 가림 전후의 동작을 하나의 연속 시퀀스로 오인하지 않는다.
                self._forget(tid)

            if tid not in self._bufs:
                self._bufs[tid] = deque(maxlen=self.seq_len)
                self._cnts[tid] = 0
            self._missing[tid] = 0

            self._bufs[tid].append(vec)
            self._cnts[tid] += 1

            # 버퍼가 충분히 쌓이고 stride 조건을 만족하면 추론
            if (len(self._bufs[tid]) == self.seq_len
                    and self._cnts[tid] % self.stride == 0
                    and sum(bool(item.any()) for item in self._bufs[tid]) >= self.seq_len * 0.5):
                probs = self._infer(self._bufs[tid])
                self.inference_count += 1
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
                self._results[tid] = (cls, conf)

        # 짧은 누락은 학습 추출기와 같은 0 벡터로 표시하고 수집 기록을 보존한다.
        # 실제로 관찰하지 못한 시점에는 새 행동 판정이나 사건을 생성하지 않는다.
        for tid in list(self._bufs):
            if tid not in observed_ids:
                self._missing[tid] = self._missing.get(tid, 0) + 1
                if self._missing[tid] > self.max_missing:
                    self._forget(tid)
                else:
                    self._bufs[tid].append(np.zeros(34, dtype=np.float32))
                    self._cnts[tid] += 1

        return {tid: value for tid, value in self._results.items()
                if tid in observed_ids and sum(bool(item.any()) for item in self._bufs[tid]) >= self.seq_len * 0.5}

    def _forget(self, tid):
        self._bufs.pop(tid, None)
        self._cnts.pop(tid, None)
        self._probs.pop(tid, None)
        self._results.pop(tid, None)
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
        # A weak normal argmax is also uncertain, not evidence that nothing happened.
        if state == "classified" and candidate["score"] < self.abn_conf_thr:
            state = "uncertain"
        return {"collected": len(buf), "required": self.seq_len,
                "valid": valid, "missing_frames": missing, "state": state, "candidate": candidate}

    @torch.no_grad()
    def _infer(self, buf: deque) -> np.ndarray:
        seq = torch.from_numpy(np.stack(list(buf))).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(seq), dim=1)[0]
        return prob.cpu().numpy()
