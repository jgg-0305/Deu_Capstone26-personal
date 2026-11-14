"""학습 데이터와 같은 BGR 3프레임 입력을 사용하는 장면 단위 분류기."""

from collections import deque

import cv2
import torch
from torch import nn
from torchvision.models import resnet50


class RgbBehaviorClassifier(nn.Module):
    def __init__(self, path):
        super().__init__()
        backbone = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.lstm = nn.Module()
        self.lstm.lstm = nn.LSTM(512, 256, batch_first=True)
        self.fc = nn.Linear(256, 8)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        weights = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict({key.removeprefix("module."): value for key, value in weights.items()})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device).eval()
        self.features = deque(maxlen=3)
        self.inference_count = 0
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def update(self, frame):
        resized = cv2.resize(frame, (224, 224))
        image = torch.from_numpy(resized).permute(2, 0, 1)[None].to(self.device, torch.float32) / 255
        self.features.append(self.backbone((image - self.mean) / self.std).reshape(1, 4, 512))
        if len(self.features) < 3:
            return None
        _, (hidden, _) = self.lstm.lstm(torch.cat(list(self.features), dim=1))
        scores = self.fc(hidden[-1]).softmax(dim=1)
        self.inference_count += 1
        score, label = scores[0].max(dim=0)
        return int(label), float(score)
