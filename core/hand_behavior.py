"""Aligned wrist imagery and reach/return motion for person-level classification."""
import cv2
import numpy as np
import torch
from torch import nn

PATCH = 16
MOTION = 16
INPUT_SIZE = 34 + MOTION + 2 * 3 * PATCH * PATCH
ARCHITECTURE = "hand_fusion_v1"


def encode_context(frame, points, box, pose):
    vector = np.zeros(INPUT_SIZE, np.float32)
    vector[:34] = pose
    if frame is None or points is None:
        return vector
    height, width = frame.shape[:2]
    scale = max(float(box[3] - box[1]), 1)
    radius = int(np.clip(scale * .16, 8, 112))
    for side, (wrist, elbow, shoulder, hip) in enumerate(((9, 7, 5, 11), (10, 8, 6, 12))):
        hand = points[wrist]
        if hand[2] < .3 or not (0 <= hand[0] < width and 0 <= hand[1] < height):
            continue
        motion = vector[34 + 8 * side:42 + 8 * side]
        motion[0] = 1
        for i, joint in enumerate((0, hip, shoulder, elbow)):
            if points[joint, 2] >= .3:
                motion[i + 1] = min(float(np.linalg.norm(hand[:2] - points[joint, :2])) / scale, 2)
        center = hand[:2].copy()
        if points[elbow, 2] >= .3:
            direction = hand[:2] - points[elbow, :2]
            center += direction * min(.35, radius / max(1., float(np.linalg.norm(direction))))
        x, y = map(int, center)
        crop = frame[max(0, y-radius):min(height, y+radius), max(0, x-radius):min(width, x+radius)]
        if crop.size and x + radius > 0 and y + radius > 0:
            patch = cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
            start = 34 + MOTION + side * 3 * PATCH * PATCH
            vector[start:start + 3 * PATCH * PATCH] = patch[:, :, ::-1].transpose(2, 0, 1).reshape(-1)
        motion[5:7] = (hand[:2] - np.asarray(box[:2])) / scale
        motion[7] = hand[2]
    return vector


class HandFusion(nn.Module):
    def __init__(self):
        super().__init__()
        from core.kpt_behavior_classifier import KeypointLSTMv2
        self.pose = KeypointLSTMv2()
        self.image = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(64, 16), nn.ReLU(),
        )
        self.temporal = nn.GRU(2 * 16 + 2 * MOTION, 48, batch_first=True)
        self.residual = nn.Sequential(nn.Dropout(.25), nn.Linear(48, 8))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        batch = x.size(0)
        timesteps = x.size(1)
        motion = x[..., 34:34 + MOTION]
        valid = motion[..., (0, 8)]
        patch = x[..., 34 + MOTION:].reshape(-1, 3, PATCH, PATCH).float() / 255
        appearance = self.image(patch).reshape(batch, timesteps, 2, 16) * valid.unsqueeze(-1)
        speed = torch.diff(motion, dim=1, prepend=motion[:, :1])
        pair_valid = valid * torch.cat((valid[:, :1], valid[:, :-1]), 1)
        speed = (speed.reshape(batch, timesteps, 2, 8) * pair_valid.unsqueeze(-1)).flatten(-2)
        _, state = self.temporal(torch.cat((appearance.flatten(-2), motion, speed), -1))
        return self.pose(x[..., :34]) + self.residual(state[-1])
