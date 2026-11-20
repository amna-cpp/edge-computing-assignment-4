"""MobileNetV2 (Sandler et al., 2018) adapted to 32x32 CIFAR inputs.

Inverted-residual blocks with linear bottlenecks and ReLU6.  For CIFAR the stem and
the second stage use stride 1 (total stride 16: 32 -> 32 -> 16 -> 8 -> 4), the usual
adaptation; everything else follows the paper (width multiplier 1.0).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_act(in_ch, out_ch, k=3, stride=1, groups=1, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        layers = []
        if expand != 1:
            layers.append(conv_bn_act(in_ch, hidden, 1))                       # expand 1x1
        layers.append(conv_bn_act(hidden, hidden, 3, stride, groups=hidden))   # depthwise 3x3
        layers.append(conv_bn_act(hidden, out_ch, 1, act=False))               # linear project
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class MobileNetV2(nn.Module):
    arch = "mobilenetv2"
    feature_channels = (96, 1280)      # (mid-feature @8x8, final feature @4x4)

    # t (expansion), c (channels), n (repeats), s (stride)  - CIFAR strides
    SETTINGS = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
                (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes: int = 100, dropout: float = 0.2):
        super().__init__()
        in_ch = 32
        stages = [conv_bn_act(3, in_ch, 3, 1)]
        self.mid_index = None
        for i, (t, c, n, s) in enumerate(self.SETTINGS):
            for j in range(n):
                stages.append(InvertedResidual(in_ch, c, s if j == 0 else 1, t))
                in_ch = c
            if i == 4:                                   # end of the 96-channel stage (8x8)
                self.mid_index = len(stages)
        stages.append(conv_bn_act(in_ch, 1280, 1))
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features: bool = False):
        mid = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i + 1 == self.mid_index:
                mid = x
        logits = self.classifier(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))
        return (logits, [mid, x]) if return_features else logits
