"""Generic classification training engine shared by every training script.

* SGD (Nesterov) + linear warm-up + per-iteration cosine decay
* pluggable ``loss_fn`` so plain CE, knowledge distillation and NAS proxy
  training all reuse exactly the same loop
* resumable: a ``<ckpt>.resume`` file is written after every epoch
* per-epoch history written to CSV
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edge_common.data import CifarData

# loss_fn(model, x, y) -> (loss, logits, stats)
LossFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def ce_loss_fn(label_smoothing: float = 0.0) -> LossFn:
    def fn(model, x, y):
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        return loss, logits, {}
    return fn


@torch.no_grad()
def evaluate(model: nn.Module, data: CifarData, split: str = "test",
             batch_size: int = 500) -> Tuple[float, float]:
    """Returns (cross-entropy loss, top-1 accuracy in %)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in data.eval_batches(split, batch_size):
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return total_loss / n, 100.0 * correct / n


def fit(model: nn.Module, data: CifarData, *, epochs: int, batch_size: int, lr: float,
        weight_decay: float, device: torch.device, loss_fn: Optional[LossFn] = None,
        extra_modules: Iterable[nn.Module] = (), warmup_epochs: float = 1.0,
        ckpt_path: Optional[Path] = None, history_csv: Optional[Path] = None,
        eval_split: str = "test", resume: bool = True, momentum: float = 0.9,
        amp: str = "none", verbose: bool = True, tag: str = "",
        max_seconds: Optional[float] = None) -> List[Dict[str, float]]:
    """Train ``model`` and return the per-epoch history.

    ``extra_modules`` hold trainable parameters that are not part of the deployed
    model (e.g. feature-distillation adapters); they are optimised jointly.
    ``eval_split`` is the split reported every epoch ("val" for NAS, "test" otherwise).
    The final-epoch weights are kept - no checkpoint selection on the test set.
    """
    loss_fn = loss_fn or ce_loss_fn()
    extra_modules = list(extra_modules)
    params = list(model.parameters()) + [p for m in extra_modules for p in m.parameters()]
    # no weight decay on biases / norm parameters
    decay = [p for p in params if p.ndim > 1]
    no_decay = [p for p in params if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum, nesterov=True)

    steps_per_epoch = data.steps_per_epoch(batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return lr * (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * lr * (1.0 + math.cos(math.pi * progress))

    history: List[Dict[str, float]] = []
    start_epoch = 0
    resume_file = Path(str(ckpt_path) + ".resume") if ckpt_path else None
    if resume and resume_file is not None and resume_file.exists():
        state = torch.load(resume_file, map_location="cpu")
        model.load_state_dict(state["model"])
        for m, sd in zip(extra_modules, state["extra"]):
            m.load_state_dict(sd)
        opt.load_state_dict(state["optimizer"])
        history, start_epoch = state["history"], state["epoch"]
        if verbose:
            print(f"[{tag}] resumed from epoch {start_epoch}/{epochs}", flush=True)

    amp_ctx = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device_type=device.type,
                               dtype=torch.bfloat16 if amp == "bf16" else torch.float16))

    t_start = time.time()
    step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        for m in extra_modules:
            m.train()
        t0 = time.time()
        run_loss, run_correct, run_n = 0.0, 0, 0
        stat_sums: Dict[str, float] = {}
        for x, y in data.train_batches(batch_size):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with amp_ctx():
                loss, logits, stats = loss_fn(model, x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            run_loss += loss.detach() * y.numel()
            run_correct += (logits.detach().argmax(1) == y).sum()
            run_n += y.numel()
            for k, v in stats.items():
                v = v.detach() if torch.is_tensor(v) else v
                stat_sums[k] = stat_sums.get(k, 0.0) + v * y.numel()
        sync(device)
        train_loss = float(run_loss) / run_n
        if not math.isfinite(train_loss):
            raise RuntimeError(f"[{tag}] training diverged (loss={train_loss}) at epoch {epoch + 1}")
        train_acc = 100.0 * float(run_correct) / run_n
        eval_loss, eval_acc = evaluate(model, data, eval_split)
        rec = {"epoch": epoch + 1, "lr": lr_at(step - 1), "train_loss": train_loss,
               "train_acc": train_acc, f"{eval_split}_loss": eval_loss,
               f"{eval_split}_acc": eval_acc, "epoch_time_s": time.time() - t0}
        rec.update({k: float(v) / run_n for k, v in stat_sums.items()})
        history.append(rec)
        if verbose:
            extras = " ".join(f"{k}={float(v) / run_n:.3f}" for k, v in stat_sums.items())
            print(f"[{tag}] epoch {epoch + 1:3d}/{epochs} loss {train_loss:.3f} "
                  f"train_acc {train_acc:5.2f} {eval_split}_acc {eval_acc:5.2f} "
                  f"{extras} ({rec['epoch_time_s']:.1f}s)", flush=True)

        if history_csv is not None:
            _write_history(history_csv, history)
        if resume_file is not None:
            resume_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(resume_file) + ".tmp")
            torch.save({"model": model.state_dict(),
                        "extra": [m.state_dict() for m in extra_modules],
                        "optimizer": opt.state_dict(), "history": history,
                        "epoch": epoch + 1}, tmp)
            os.replace(tmp, resume_file)
        if max_seconds is not None and time.time() - t_start > max_seconds and epoch + 1 < epochs:
            print(f"[{tag}] time budget reached after epoch {epoch + 1}; stopping early", flush=True)
            break
    return history


def _write_history(path: Path, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for rec in history:
        for k in rec:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
