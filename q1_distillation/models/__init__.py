"""Model registry for question 1 (teacher + three student architectures)."""
from __future__ import annotations

import torch.nn as nn

from q1_distillation.models.efficientnet_lite import EfficientNetLite0
from q1_distillation.models.mobilenetv2 import MobileNetV2
from q1_distillation.models.resnet import ResNet50
from q1_distillation.models.squeezenet import SqueezeNet

TEACHER = "resnet50"
STUDENTS = ("mobilenetv2", "squeezenet", "efficientnet_lite0")

_REGISTRY = {
    "resnet50": ResNet50,
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}


def build_model(arch: str, num_classes: int = 100, **kwargs) -> nn.Module:
    if arch not in _REGISTRY:
        raise ValueError(f"unknown architecture {arch!r}; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[arch](num_classes=num_classes, **kwargs)
