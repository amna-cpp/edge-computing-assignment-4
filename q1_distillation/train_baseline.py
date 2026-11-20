"""Baseline pipeline: train the three student architectures with plain cross-entropy (NO distillation).

    python -m q1_distillation.train_baseline --arch all --epochs 15

Hyper-parameters are intentionally identical to ``train_kd`` so that any accuracy
difference between the two pipelines is attributable to distillation alone.
"""
from __future__ import annotations

import argparse
import json
import time

from edge_common import RESULTS_DIR
from edge_common.data import CifarData, get_device
from edge_common.deploy import count_macs, count_params
from edge_common.engine import ce_loss_fn, evaluate, fit, set_seed
from q1_distillation.ckpt import save_model, student_ckpt_path
from q1_distillation.models import STUDENTS, build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="all", choices=["all", *STUDENTS])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    device = get_device(a.device)
    data = CifarData("cifar100", device)
    archs = STUDENTS if a.arch == "all" else (a.arch,)
    for arch in archs:
        path = student_ckpt_path(arch, "baseline")
        if path.exists() and not a.no_resume:
            print(f"[{arch}/baseline] checkpoint exists ({path.name}) - skipping", flush=True)
            continue
        set_seed(a.seed)
        model = build_model(arch, 100).to(device)
        print(f"[{arch}/baseline] params={count_params(model) / 1e6:.2f}M "
              f"MACs={count_macs(model) / 1e6:.0f}M device={device}", flush=True)
        t0 = time.time()
        fit(model, data, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
            weight_decay=a.weight_decay, device=device, loss_fn=ce_loss_fn(),
            tag=f"{arch}/baseline", ckpt_path=path, resume=not a.no_resume,
            history_csv=RESULTS_DIR / "q1" / f"{arch}_baseline_history.csv")
        _, acc = evaluate(model, data, "test")
        print(f"[{arch}/baseline] FINAL test top-1 = {acc:.2f}%", flush=True)
        save_model(model, path, arch, test_acc=acc, epochs=a.epochs, variant="baseline",
                   args=vars(a), dataset="cifar100")
        with open(RESULTS_DIR / "q1" / f"{arch}_baseline_result.json", "w") as f:
            json.dump({"test_acc": acc, "epochs": a.epochs, "params": count_params(model),
                       "macs": count_macs(model), "train_seconds": time.time() - t0}, f, indent=2)
        (path.parent / (path.name + ".resume")).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
