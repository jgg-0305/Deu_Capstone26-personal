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
"""

from collections import deque, OrderedDict

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
                 device=None, ema_alpha: float = 0.6, abn_conf_thr: float = 0.5):
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

        # track_id → deque[(34,)]
        self._bufs: dict[int, deque] = {}
        # track_id → frame 카운터
        self._cnts: dict[int, int] = {}
        # track_id → EMA 확률 벡터 (num_classes,)
        self._probs: dict[int, np.ndarray] = {}
        # track_id → (class_id, conf)
        self._results: dict[int, tuple] = {}

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
        return model

    def update(self, tracks: list, kpt_map: dict) -> dict:
        """tracks, kpt_map 을 받아 결과 딕셔너리를 반환·갱신한다."""
        active_ids = set()
        for t in tracks:
            tid = t["track_id"]
            active_ids.add(tid)
            vec = _encode(kpt_map.get(tid), t["bbox"])

            if tid not in self._bufs:
                self._bufs[tid] = deque(maxlen=self.seq_len)
                self._cnts[tid] = 0

            self._bufs[tid].append(vec)
            self._cnts[tid] += 1

            # 버퍼가 충분히 쌓이고 stride 조건을 만족하면 추론
            if (len(self._bufs[tid]) == self.seq_len
                    and self._cnts[tid] % self.stride == 0):
                probs = self._infer(self._bufs[tid])
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

        # 사라진 트랙 정리
        for tid in list(self._bufs):
            if tid not in active_ids:
                self._bufs.pop(tid)
                self._cnts.pop(tid)
                self._probs.pop(tid, None)
                self._results.pop(tid, None)

        return dict(self._results)

    @torch.no_grad()
    def _infer(self, buf: deque) -> np.ndarray:
        seq = torch.from_numpy(np.stack(list(buf))).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(seq), dim=1)[0]
        return prob.cpu().numpy()
