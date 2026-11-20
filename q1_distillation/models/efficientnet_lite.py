"""EfficientNet-Lite0 (TensorFlow Lite / Google, 2020) adapted to 32x32 CIFAR inputs.

The "Lite" variants are EfficientNet-B0 optimised for edge deployment: no
squeeze-and-excitation, ReLU6 instead of swish, and the stem / head are not scaled with
the compound coefficient.  Lite0 uses width = depth = 1.0:

    stem 32 | MBConv1 k3 16 x1 | MBConv6 k3 24 x2 | MBConv6 k5 40 x2 | MBConv6 k3 80 x3
            | MBConv6 k5 112 x3 | MBConv6 k5 192 x4 | MBConv6 k3 320 x1 | head 1280

For CIFAR the stem and the 24-channel stage use stride 1 (total stride 16).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_act(in_ch, out_ch, k=3, stride=1, groups=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch, momentum=0.1, eps=1e-3)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class MBConv(nn.Module):
    """Mobile inverted bottleneck without squeeze-excite; stochastic depth on the residual."""

    def __init__(self, in_ch, out_ch, k, stride, expand, drop_path=0.0):
        super().__init__()
        hidden = in_ch * expand
        layers = []
        if expand != 1:
            layers.append(conv_bn_act(in_ch, hidden, 1))
        layers.append(conv_bn_act(hidden, hidden, k, stride, groups=hidden))
        layers.append(conv_bn_act(hidden, out_ch, 1, act=False))
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and in_ch == out_ch
        self.drop_path = drop_path

    def forward(self, x):
        out = self.block(x)
        if not self.use_res:
            return out
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep).to(out.dtype) / keep
            out = out * mask
        return x + out


class EfficientNetLite0(nn.Module):
    arch = "efficientnet_lite0"
    feature_channels = (112, 1280)     # (mid-feature @8x8, final feature @4x4)

    # (expand, kernel, stride, out_channels, repeats)  - CIFAR strides
    SETTINGS = [(1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
                (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]

    def __init__(self, num_classes: int = 100, dropout: float = 0.2, drop_path_rate: float = 0.1):
        super().__init__()
        total_blocks = sum(s[4] for s in self.SETTINGS)
        in_ch = 32
        stages = [conv_bn_act(3, in_ch, 3, 1)]
        self.mid_index = None
        b = 0
        for i, (t, k, s, c, n) in enumerate(self.SETTINGS):
            for j in range(n):
                stages.append(MBConv(in_ch, c, k, s if j == 0 else 1, t,
                                     drop_path=drop_path_rate * b / total_blocks))
                in_ch = c
                b += 1
            if i == 4:                                   # end of the 112-channel stage (8x8)
                self.mid_index = len(stages)
        stages.append(conv_bn_act(in_ch, 1280, 1))
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1 / 1280 ** 0.5, 1 / 1280 ** 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features: bool = False):
        mid = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i + 1 == self.mid_index:
                mid = x
        logits = self.classifier(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))
        return (logits, [mid, x]) if return_features else logits
