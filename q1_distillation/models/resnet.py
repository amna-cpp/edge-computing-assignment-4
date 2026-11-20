"""ResNet-50 (bottleneck) for 32x32 CIFAR images - the teacher network.

Differences w.r.t. the ImageNet ResNet-50: the 7x7/stride-2 stem and the max-pool are
replaced by a single 3x3 convolution (``stem_stride`` selects the stride), which is
the standard adaptation for CIFAR-sized inputs.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch: int, planes: int, stride: int = 1):
        super().__init__()
        out_ch = planes * self.expansion
        self.conv1 = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet50(nn.Module):
    """ResNet-50: stages of [3, 4, 6, 3] bottleneck blocks, widths 64-128-256-512 (x4 expansion)."""

    arch = "resnet50"
    feature_channels = (1024, 2048)    # channels of the two exposed feature maps

    def __init__(self, num_classes: int = 100, stem_stride: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stem_stride, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        cfg = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]
        layers, in_ch = [], 64
        for planes, blocks, stride in cfg:
            strides = [stride] + [1] * (blocks - 1)
            seq = []
            for s in strides:
                seq.append(Bottleneck(in_ch, planes, s))
                in_ch = planes * Bottleneck.expansion
            layers.append(nn.Sequential(*seq))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(2048, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # zero-init the last BN gamma of each residual branch -> identity at start
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.bn3.weight)

    def forward(self, x, return_features: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        logits = self.fc(torch.flatten(F.adaptive_avg_pool2d(f4, 1), 1))
        if return_features:
            return logits, [f3, f4]
        return logits
