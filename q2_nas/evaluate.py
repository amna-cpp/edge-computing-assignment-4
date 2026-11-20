"""Automated training / validation pipeline for NAS candidates (CIFAR-10 proxy training)."""
from __future__ import annotations

import time
from typing import Dict

import torch

from edge_common.data import CifarData
from edge_common.engine import ce_loss_fn, evaluate, fit, set_seed
from q2_nas.generator import build_from_config
from q2_nas.search_space import Config


def train_candidate(cfg: Config, data: CifarData, device: torch.device, epochs: int,
                    batch_size: int = 128, lr: float = 0.08, weight_decay: float = 5e-4,
                    seed: int = 0) -> Dict[str, float]:
    """Train a candidate from scratch for a few epochs and report accuracy on the validation split.

    Returns {"val_acc": %, "train_acc": %, "train_time_s": s}.  The validation split is held
    out from the CIFAR-10 *training* set, the official test set is never used during search.
    """
    set_seed(seed)
    model = build_from_config(cfg, data.num_classes).to(device)
    t0 = time.time()
    hist = fit(model, data, epochs=epochs, batch_size=batch_size, lr=lr,
               weight_decay=weight_decay, device=device, loss_fn=ce_loss_fn(),
               warmup_epochs=0.5, eval_split="val", resume=False, verbose=False, tag="nas")
    _, val_acc = evaluate(model, data, "val")
    return {"val_acc": val_acc, "train_acc": hist[-1]["train_acc"],
            "train_time_s": time.time() - t0}
