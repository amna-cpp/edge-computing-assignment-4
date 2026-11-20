"""Train the best architecture found by the NAS from scratch (full CIFAR-10 training set).

    python -m q2_nas.train_best --epochs 40

Reads results/q2/best_architecture.json (written by ``q2_nas.search``), trains the network
from random initialisation on all 50 000 training images and evaluates it on the official
10 000-image test set.  The final-epoch weights are saved (no selection on the test set).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from edge_common import CKPT_DIR, RESULTS_DIR
from edge_common.data import CifarData, get_device
from edge_common.engine import ce_loss_fn, evaluate, fit, set_seed
from q2_nas.generator import build_from_config
from q2_nas.search_space import describe

NAS_CKPT = CKPT_DIR / "nas" / "nas_best.pt"
BEST_JSON = RESULTS_DIR / "q2" / "best_architecture.json"


def load_nas_model(path: Path = NAS_CKPT, device: str = "cpu"):
    """Rebuild the NAS network from a checkpoint (config + weights)."""
    ckpt = torch.load(path, map_location="cpu")
    model = build_from_config(ckpt["config"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    best = json.load(open(BEST_JSON))
    cfg = best["config"]
    print(f"best architecture: {describe(cfg)}", flush=True)
    set_seed(a.seed)
    device = get_device(a.device)
    data = CifarData("cifar10", device)                       # full 50k training set
    model = build_from_config(cfg, 10).to(device)

    t0 = time.time()
    fit(model, data, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
        device=device, loss_fn=ce_loss_fn(), tag="nas-best", ckpt_path=NAS_CKPT,
        resume=not a.no_resume, history_csv=RESULTS_DIR / "q2" / "nas_best_history.csv")
    _, acc = evaluate(model, data, "test")
    print(f"[nas-best] FINAL CIFAR-10 test top-1 = {acc:.2f}% (search proxy val acc was "
          f"{best['search_val_acc']:.2f}%)", flush=True)

    NAS_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "num_classes": 10,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": {"test_acc": acc, "epochs": a.epochs, "args": vars(a), "dataset": "cifar10",
                         "search": {k: best[k] for k in ("search_val_acc", "latency_ms", "size_mb",
                                                         "peak_mem_mb", "params", "macs")}}}, NAS_CKPT)
    Path(str(NAS_CKPT) + ".resume").unlink(missing_ok=True)
    with open(RESULTS_DIR / "q2" / "nas_best_result.json", "w") as f:
        json.dump({"test_acc": acc, "epochs": a.epochs, "train_seconds": time.time() - t0,
                   "description": describe(cfg), "config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
