"""WASP (Waterfall Atrous Spatial Pooling) — source/UniPose/lib/models/modules/wasp.py 정리판."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Atrous(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class WASP(nn.Module):
    def __init__(self, output_stride, BatchNorm):
        super().__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [24, 18, 12, 6]
        elif output_stride == 8:
            dilations = [48, 36, 24, 12]
        else:
            raise NotImplementedError

        self.aspp1 = _Atrous(inplanes, 256, 1, 0, dilations[0], BatchNorm)
        self.aspp2 = _Atrous(256, 256, 3, dilations[1], dilations[1], BatchNorm)
        self.aspp3 = _Atrous(256, 256, 3, dilations[2], dilations[2], BatchNorm)
        self.aspp4 = _Atrous(256, 256, 3, dilations[3], dilations[3], BatchNorm)

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.conv2 = nn.Conv2d(256, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        # waterfall 연결 후 conv2 통과
        x1 = self.conv2(self.conv2(x1))
        x2 = self.conv2(self.conv2(x2))
        x3 = self.conv2(self.conv2(x3))
        x4 = self.conv2(self.conv2(x4))

        x5 = F.interpolate(self.global_pool(x), size=x4.shape[2:],
                           mode="bilinear", align_corners=True)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)  # (B, 1280, H, W)
        out = self.relu(self.bn1(self.conv1(out)))
        return self.dropout(out)


def build_wasp(output_stride=8, BatchNorm=nn.BatchNorm2d):
    return WASP(output_stride, BatchNorm)
