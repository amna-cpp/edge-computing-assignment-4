"""Knowledge-distillation framework.

Implements the four ingredients required by the assignment:

1. **Soft-target distillation** - the student matches the teacher's output distribution
   (KL divergence).
2. **Temperature-based softmax** - both distributions are softened with temperature T;
   the soft term is multiplied by T^2 so its gradient magnitude stays comparable to the
   hard-label term (Hinton et al., 2015).
3. **Combined hard + soft loss** - ``L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL``.
4. **Feature-based distillation** (FitNets-style hints) - intermediate student feature maps
   are projected with 1x1-conv adapters to the teacher's channel width and regressed onto
   the (frozen) teacher feature maps.  The adapters only exist during training; they are
   not part of the deployed student, its checkpoint or its ONNX graph.

    L_total = alpha * CE + (1 - alpha) * T^2 * KL + beta * mean_i MSE(adapter_i(f_s_i), f_t_i)
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- logit terms
def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """softmax(z / T): T > 1 flattens the distribution and exposes 'dark knowledge'."""
    return F.softmax(logits / temperature, dim=-1)


def soft_target_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     temperature: float) -> torch.Tensor:
    """T^2 * KL( softmax(t / T) || softmax(s / T) ), averaged over the batch."""
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    p_teacher = softmax_with_temperature(teacher_logits, temperature)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)


def combined_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                     labels: torch.Tensor, temperature: float, alpha: float
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """alpha * hard CE + (1 - alpha) * soft KD; returns (total, hard, soft)."""
    hard = F.cross_entropy(student_logits, labels)
    soft = soft_target_loss(student_logits, teacher_logits, temperature)
    return alpha * hard + (1.0 - alpha) * soft, hard, soft


# --------------------------------------------------------------------------- feature terms
def _standardise(f: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-sample zero-mean / unit-variance over (C, H, W) -> the loss ignores feature scale."""
    mean = f.mean(dim=(1, 2, 3), keepdim=True)
    var = f.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    return (f - mean) / torch.sqrt(var + eps)


class FeatureAdapters(nn.Module):
    """1x1 conv + BN regressors mapping student feature maps to the teacher's channel width.

    Both maps are standardised per sample before the MSE (so the term is bounded and
    comparable in size to the logit losses, whatever the activation scale of the teacher).
    """

    def __init__(self, student_channels: Sequence[int], teacher_channels: Sequence[int]):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(s, t, 1, bias=False), nn.BatchNorm2d(t))
            for s, t in zip(student_channels, teacher_channels)])

    def forward(self, student_feats: List[torch.Tensor], teacher_feats: List[torch.Tensor]
                ) -> torch.Tensor:
        loss = 0.0
        for adapter, fs, ft in zip(self.adapters, student_feats, teacher_feats):
            fs = adapter(fs)
            if fs.shape[-2:] != ft.shape[-2:]:
                # teacher maps are pooled down to the student's spatial size
                ft = F.adaptive_avg_pool2d(ft, fs.shape[-2:])
            loss = loss + F.mse_loss(_standardise(fs), _standardise(ft))
        return loss / len(self.adapters)


# --------------------------------------------------------------------------- loss factory
class DistillationLoss:
    """Callable ``loss_fn(student, x, y)`` compatible with ``edge_common.engine.fit``."""

    def __init__(self, teacher: nn.Module, temperature: float = 4.0, alpha: float = 0.3,
                 feature_weight: float = 0.0, adapters: FeatureAdapters = None):
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.temperature, self.alpha = temperature, alpha
        self.feature_weight, self.adapters = feature_weight, adapters
        if feature_weight > 0 and adapters is None:
            raise ValueError("feature_weight > 0 requires FeatureAdapters")

    def __call__(self, student: nn.Module, x: torch.Tensor, y: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        self.teacher.eval()                         # teacher BN must stay in eval mode
        with torch.no_grad():
            if self.feature_weight > 0:
                t_logits, t_feats = self.teacher(x, return_features=True)
            else:
                t_logits, t_feats = self.teacher(x), None
        if self.feature_weight > 0:
            s_logits, s_feats = student(x, return_features=True)
        else:
            s_logits = student(x)
        loss, hard, soft = combined_kd_loss(s_logits, t_logits, y, self.temperature, self.alpha)
        stats = {"hard": hard.detach(), "soft": soft.detach()}
        if self.feature_weight > 0:
            feat = self.adapters(s_feats, t_feats)
            loss = loss + self.feature_weight * feat
            stats["feat"] = feat.detach()
        return loss, s_logits, stats
