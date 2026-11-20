"""Knowledge-distillation training of the three students from the ResNet-50 teacher.

    # full framework: hard + soft (temperature) + feature distillation  -> variant "kd"
    python -m q1_distillation.train_kd --arch all --epochs 15

    # ablation: logit-only distillation (no feature term)                -> variant "kd_logit"
    python -m q1_distillation.train_kd --arch all --epochs 15 --feature-weight 0 --variant kd_logit
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from edge_common import RESULTS_DIR
from edge_common.data import CifarData, get_device
from edge_common.deploy import count_macs, count_params
from edge_common.engine import evaluate, fit, set_seed
from q1_distillation.ckpt import TEACHER_CKPT, load_model, save_model, student_ckpt_path
from q1_distillation.distill import DistillationLoss, FeatureAdapters
from q1_distillation.models import STUDENTS, build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="all", choices=["all", *STUDENTS])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=4.0, help="softmax temperature T")
    ap.add_argument("--alpha", type=float, default=0.3, help="weight of the hard-label CE term")
    ap.add_argument("--feature-weight", type=float, default=1.0,
                    help="beta, weight of the feature-distillation term (0 disables it)")
    ap.add_argument("--variant", default="kd", help="name used in the checkpoint file name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    device = get_device(a.device)
    data = CifarData("cifar100", device)
    teacher, tmeta = load_model(TEACHER_CKPT, device)
    print(f"teacher resnet50 loaded (test acc {tmeta['meta'].get('test_acc', float('nan')):.2f}%)", flush=True)

    archs = STUDENTS if a.arch == "all" else (a.arch,)
    for arch in archs:
        path = student_ckpt_path(arch, a.variant)
        if path.exists() and not a.no_resume:
            print(f"[{arch}/{a.variant}] checkpoint exists ({path.name}) - skipping", flush=True)
            continue
        set_seed(a.seed)
        student = build_model(arch, 100).to(device)
        adapters = None
        if a.feature_weight > 0:
            adapters = FeatureAdapters(student.feature_channels, teacher.feature_channels).to(device)
        loss_fn = DistillationLoss(teacher, a.temperature, a.alpha, a.feature_weight, adapters)
        print(f"[{arch}/{a.variant}] params={count_params(student) / 1e6:.2f}M "
              f"MACs={count_macs(student) / 1e6:.0f}M T={a.temperature} alpha={a.alpha} "
              f"beta={a.feature_weight}", flush=True)
        t0 = time.time()
        fit(student, data, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
            weight_decay=a.weight_decay, device=device, loss_fn=loss_fn,
            extra_modules=[adapters] if adapters is not None else [],
            tag=f"{arch}/{a.variant}", ckpt_path=path, resume=not a.no_resume,
            history_csv=RESULTS_DIR / "q1" / f"{arch}_{a.variant}_history.csv")
        _, acc = evaluate(student, data, "test")
        print(f"[{arch}/{a.variant}] FINAL test top-1 = {acc:.2f}%", flush=True)
        save_model(student, path, arch, test_acc=acc, epochs=a.epochs, variant=a.variant,
                   args=vars(a), dataset="cifar100")
        with open(RESULTS_DIR / "q1" / f"{arch}_{a.variant}_result.json", "w") as f:
            json.dump({"test_acc": acc, "epochs": a.epochs, "params": count_params(student),
                       "macs": count_macs(student), "train_seconds": time.time() - t0,
                       "temperature": a.temperature, "alpha": a.alpha,
                       "feature_weight": a.feature_weight}, f, indent=2)
        (path.parent / (path.name + ".resume")).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
