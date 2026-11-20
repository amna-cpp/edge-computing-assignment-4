"""Checkpoint helpers: every checkpoint stores enough metadata to rebuild its model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from edge_common import CKPT_DIR
from q1_distillation.models import build_model

TEACHER_CKPT = CKPT_DIR / "teacher" / "resnet50_cifar100.pt"


def student_ckpt_path(arch: str, variant: str) -> Path:
    """variant: 'baseline' (no KD) or 'kd' (distilled) or an ablation name such as 'kd_logit'."""
    return CKPT_DIR / "students" / f"{arch}_{variant}.pt"


def save_model(model: nn.Module, path: Path, arch: str, model_kwargs: Dict[str, Any] = None,
               **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "model_kwargs": model_kwargs or {}, "num_classes": meta.pop("num_classes", 100),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": meta}, path)


def load_model(path: Path, device: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a model from a checkpoint written by :func:`save_model` (eval mode)."""
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], ckpt["num_classes"], **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
