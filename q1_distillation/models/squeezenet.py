"""SqueezeNet v1.1 (Iandola et al., 2016) adapted to 32x32 CIFAR inputs.

Adaptations: 3x3/stride-1 stem instead of 3x3/stride-2 + max-pool, BatchNorm after every
convolution (SqueezeNet without BN is very hard to train from scratch on CIFAR-100),
and a fully-convolutional classifier (1x1 conv -> global average pooling) as in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch, out_ch, k):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class Fire(nn.Module):
    """squeeze (1x1) -> expand (1x1 || 3x3), concatenated."""

    def __init__(self, in_ch, squeeze, expand1, expand3):
        super().__init__()
        self.squeeze = _conv_bn_relu(in_ch, squeeze, 1)
        self.expand1 = _conv_bn_relu(squeeze, expand1, 1)
        self.expand3 = _conv_bn_relu(squeeze, expand3, 3)

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)


class SqueezeNet(nn.Module):
    arch = "squeezenet"
    feature_channels = (256, 512)      # (mid-feature @16x16, final feature @8x8)

    def __init__(self, num_classes: int = 100, dropout: float = 0.2):
        super().__init__()
        self.stem = _conv_bn_relu(3, 64, 3)
        self.stage1 = nn.Sequential(Fire(64, 16, 64, 64), Fire(128, 16, 64, 64))     # 32x32
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = nn.Sequential(Fire(128, 32, 128, 128), Fire(256, 32, 128, 128))  # 16x16
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = nn.Sequential(Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),   # 8x8
                                    Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(512, num_classes, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_features: bool = False):
        x = self.stage1(self.stem(x))
        mid = self.stage2(self.pool1(x))                      # 16x16, 256 ch
        last = self.stage3(self.pool2(mid))                   # 8x8, 512 ch
        logits = torch.flatten(
            nn.functional.adaptive_avg_pool2d(self.classifier(self.drop(last)), 1), 1)
        return (logits, [mid, last]) if return_features else logits
