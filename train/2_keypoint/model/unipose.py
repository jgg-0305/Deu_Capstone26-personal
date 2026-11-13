"""UniPose 키포인트 모델 — source/UniPose/lib/models/unipose.py 정리판.

변경점:
- sys.path 조작 / pprint 출력 제거
- build_* 함수 직접 import
- pretrained_path 인자로 backbone 가중치 로드
- final_state.pth 호환: DataParallel 없이 model.state_dict() 저장
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import build_resnet
from .wasp import build_wasp
from .decoder import build_decoder


class UniPose(nn.Module):
    """
    입력  : (B, 3, H, W)  — RGB, ImageNet 정규화
    출력  : (B, num_joints, H//8, W//8) 히트맵  (output_stride=8 기준)
    """

    def __init__(self, num_joints=17, backbone="resnet50", output_stride=8,
                 pretrained_path=None):
        super().__init__()
        BatchNorm = nn.BatchNorm2d
        self.output_stride = output_stride

        self.backbone = build_resnet(
            name=backbone,
            output_stride=output_stride,
            BatchNorm=BatchNorm,
            pretrained_path=pretrained_path,
        )
        self.wasp = build_wasp(output_stride=output_stride, BatchNorm=BatchNorm)
        self.decoder = build_decoder(num_classes=num_joints, BatchNorm=BatchNorm)

    def forward(self, x):
        feat, low = self.backbone(x)
        feat = self.wasp(feat)
        heatmap = self.decoder(feat, low)
        if self.output_stride != 8:
            heatmap = F.interpolate(heatmap, size=x.shape[2:],
                                    mode="bilinear", align_corners=True)
        return heatmap


if __name__ == "__main__":
    model = UniPose(num_joints=17, backbone="resnet50", output_stride=8)
    model.eval()
    x = torch.rand(1, 3, 256, 192)
    with torch.no_grad():
        y = model(x)
    print("input:", x.shape, "→ heatmap:", y.shape)
