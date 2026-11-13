"""UniPose Decoder — source/UniPose/lib/models/modules/decoder.py 정리판."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    low_level_feat : ResNet layer1 출력 (256ch)
    x              : WASP 출력 (256ch)
    출력           : (B, num_classes, H', W') 히트맵
    """

    def __init__(self, num_classes, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        # low-level feature 압축 (256 → 48)
        self.conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 48 + 256 = 304 → 히트맵
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x, low):
        low = self.relu(self.bn1(self.conv1(low)))
        low = self.maxpool(low)
        x = F.interpolate(x, size=low.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, low], dim=1)
        return self.last_conv(x)


def build_decoder(num_classes, BatchNorm=nn.BatchNorm2d):
    return Decoder(num_classes, BatchNorm)
