"""Automated architecture generator: turns a sampled configuration into an ``nn.Module``."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from q2_nas.search_space import Config, validate

INPUT_SIZE = 32          # every deployed model accepts 3x32x32 images; resizing happens in-graph


def make_activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU(inplace=True), "relu6": nn.ReLU6(inplace=True),
            "hardswish": nn.Hardswish(inplace=True), "silu": nn.SiLU(inplace=True),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True)}[name]


class ConvBlock(nn.Module):
    """Regular k x k convolution -> BN -> activation (residual when shapes allow)."""

    def __init__(self, in_ch, out_ch, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_ch, out_ch, k, stride, k // 2, bias=False),
                                  nn.BatchNorm2d(out_ch), make_activation(act))
        self.residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.body(x)
        return out + x if self.residual else out


class DWSepBlock(nn.Module):
    """Depthwise-separable convolution: depthwise k x k -> BN -> act -> pointwise 1x1 -> BN -> act."""

    def __init__(self, in_ch, out_ch, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, stride, k // 2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), make_activation(act),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), make_activation(act))
        self.residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.body(x)
        return out + x if self.residual else out


_BLOCKS = {"conv": ConvBlock, "dwsep": DWSepBlock}


class SearchableNet(nn.Module):
    """Backbone generated from a configuration dict (see ``search_space``)."""

    def __init__(self, cfg: Config, num_classes: int = 10):
        super().__init__()
        validate(cfg)
        self.cfg = cfg
        self.resolution = cfg["resolution"]
        first_act = cfg["stages"][0]["activation"]
        in_ch = cfg["stem_channels"]
        layers = [nn.Conv2d(3, in_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(in_ch),
                  make_activation(first_act)]
        for i, st in enumerate(cfg["stages"]):
            for j in range(st["depth"]):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(_BLOCKS[st["block"]](in_ch, st["channels"], st["kernel"], stride,
                                                   st["activation"]))
                in_ch = st["channels"]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        if self.resolution != x.shape[-1]:              # variable input resolution (in-graph resize)
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear",
                              align_corners=False)
        x = self.features(x)
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def build_from_config(cfg: Config, num_classes: int = 10) -> SearchableNet:
    return SearchableNet(cfg, num_classes)
