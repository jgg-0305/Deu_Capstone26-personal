"""ResNet backbone — source/2. 키포인트 객체 인식/UniPose/lib/models/modules/resnet.py 정리판.

변경점:
- 절대 경로 하드코딩 제거 → pretrained_path 인자 받음
- 불필요한 logger 의존 제거 (print 사용)
"""
import math
import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            res = self.downsample(x)
        return self.relu(out + res)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            res = self.downsample(x)
        return self.relu(out + res)


class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm,
                 pretrained_path=None, name="resnet50"):
        self.inplanes = 64
        self.name = name
        super().__init__()

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError(f"output_stride {output_stride} not supported")

        blocks = [1, 2, 4]  # MG unit

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], strides[0], dilations[0], BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], strides[1], dilations[1], BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], strides[2], dilations[2], BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks, strides[3], dilations[3], BatchNorm)

        self._init_weight()
        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _make_layer(self, block, planes, num_blocks, stride, dilation, BatchNorm):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, dilation, downsample, BatchNorm)]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride, dilation, BatchNorm):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride,
                        dilation=blocks[0] * dilation, downsample=downsample, BatchNorm=BatchNorm)]
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, 1,
                                dilation=blocks[i] * dilation, BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained(self, path):
        pretrain = torch.load(path, map_location="cpu", weights_only=True)
        state = self.state_dict()
        loaded = {k: v for k, v in pretrain.items() if k in state}
        state.update(loaded)
        self.load_state_dict(state)
        print(f"[ResNet] pretrained from {path} ({len(loaded)}/{len(state)} keys)")

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        low = x              # layer1 출력 → decoder skip connection
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low


_ARCH = {
    "resnet18":  (BasicBlock,  [2, 2, 2, 2]),
    "resnet34":  (BasicBlock,  [3, 4, 6, 3]),
    "resnet50":  (Bottleneck,  [3, 4, 6, 3]),
    "resnet101": (Bottleneck,  [3, 4, 23, 3]),
    "resnet152": (Bottleneck,  [3, 8, 36, 3]),
}


def build_resnet(name="resnet50", output_stride=8, BatchNorm=nn.BatchNorm2d,
                 pretrained_path=None):
    name = name.lower()
    if name not in _ARCH:
        raise NotImplementedError(f"지원하지 않는 backbone: {name}")
    block, layers = _ARCH[name]
    return ResNet(block, layers, output_stride, BatchNorm,
                  pretrained_path=pretrained_path, name=name)
