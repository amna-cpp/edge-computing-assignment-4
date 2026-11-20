"""Train the ResNet-50 teacher on CIFAR-100.

    python -m q1_distillation.train_teacher --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time

from edge_common import RESULTS_DIR
from edge_common.data import CifarData, get_device
from edge_common.deploy import count_macs, count_params
from edge_common.engine import evaluate, fit, set_seed
from q1_distillation.ckpt import TEACHER_CKPT, save_model
from q1_distillation.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--stem-stride", type=int, default=2, choices=[1, 2],
                    help="stride of the 3x3 stem conv (2 = 4x cheaper, see report)")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    device = get_device(a.device)
    data = CifarData("cifar100", device)
    model = build_model("resnet50", 100, stem_stride=a.stem_stride).to(device)
    print(f"device={device} params={count_params(model) / 1e6:.2f}M "
          f"MACs={count_macs(model) / 1e6:.0f}M args={vars(a)}", flush=True)

    from edge_common.engine import ce_loss_fn
    t0 = time.time()
    history = fit(model, data, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                  weight_decay=a.weight_decay, device=device,
                  loss_fn=ce_loss_fn(a.label_smoothing), amp=a.amp, tag="teacher",
                  ckpt_path=TEACHER_CKPT, resume=not a.no_resume,
                  history_csv=RESULTS_DIR / "q1" / "teacher_history.csv")
    _, test_acc = evaluate(model, data, "test")
    print(f"[teacher] FINAL CIFAR-100 test top-1 = {test_acc:.2f}%  "
          f"(train time this run {time.time() - t0:.0f}s)", flush=True)
    save_model(model, TEACHER_CKPT, "resnet50", {"stem_stride": a.stem_stride},
               test_acc=test_acc, epochs=a.epochs, args=vars(a), dataset="cifar100")
    with open(RESULTS_DIR / "q1" / "teacher_result.json", "w") as f:
        json.dump({"test_acc": test_acc, "epochs": a.epochs, "params": count_params(model),
                   "macs": count_macs(model), "args": vars(a)}, f, indent=2)


if __name__ == "__main__":
    main()
