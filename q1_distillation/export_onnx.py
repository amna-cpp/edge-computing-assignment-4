"""Export the trained student models (and optionally the teacher) to ONNX.

    python -m q1_distillation.export_onnx                 # all students, baseline + kd
    python -m q1_distillation.export_onnx --teacher       # also export the ResNet-50 teacher

Each export is verified against the PyTorch model with ONNX Runtime (max logit difference).
Models use a dynamic batch dimension; input ``[N, 3, 32, 32]`` float32, output ``[N, 100]`` logits.
"""
from __future__ import annotations

import argparse
import csv

from edge_common import ONNX_DIR, RESULTS_DIR
from edge_common.deploy import export_onnx, file_size_mb, verify_onnx
from q1_distillation.ckpt import TEACHER_CKPT, load_model, student_ckpt_path
from q1_distillation.inference import onnx_path
from q1_distillation.models import STUDENTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=["baseline", "kd"])
    ap.add_argument("--teacher", action="store_true")
    a = ap.parse_args()

    jobs = [(arch, v, student_ckpt_path(arch, v), onnx_path(arch, v))
            for arch in STUDENTS for v in a.variants]
    if a.teacher:
        jobs.append(("resnet50", "teacher", TEACHER_CKPT, ONNX_DIR / "resnet50_teacher.onnx"))

    rows = []
    for arch, variant, ckpt, out in jobs:
        if not ckpt.exists():
            print(f"skip {arch}/{variant}: {ckpt.name} missing")
            continue
        model, meta = load_model(ckpt)
        export_onnx(model, out)
        diff = verify_onnx(model, out)
        size = file_size_mb(out)
        print(f"{arch:20s} {variant:9s} -> {out.name:34s} {size:6.2f} MB   max|torch-ort| = {diff:.1e}")
        rows.append({"model": arch, "variant": variant, "onnx_file": out.name,
                     "size_mb": round(size, 3), "max_abs_diff_vs_torch": f"{diff:.2e}",
                     "torch_test_acc": meta["meta"].get("test_acc")})
    if rows:
        path = RESULTS_DIR / "q1" / "onnx_export_summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"summary -> {path}")


if __name__ == "__main__":
    main()
