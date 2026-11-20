"""Automated benchmark of every deployable model with ONNX Runtime on the CPU.

    python -m benchmarks.benchmark_all                 # -> results/benchmark_results.csv (+ .json, kd_gains.csv)
    python -m benchmarks.benchmark_all --iters 200

For each exported model it records
    * top-1 test accuracy measured *through ONNX Runtime* (and the PyTorch accuracy for comparison)
    * parameters, MACs, ONNX file size
    * batch-1 latency (median / mean / p95) with 1 and with 4 CPU threads
    * batch-32 throughput (images / s, 4 threads)
    * peak inference memory (RSS growth, batch 1, fresh process per model)
    * PyTorch-CPU batch-1 latency, to show the benefit of the ONNX Runtime deployment

Run it on an otherwise idle machine - latency is measured, not estimated.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from edge_common import CKPT_DIR, DATA_DIR, ONNX_DIR, RESULTS_DIR
from edge_common.data import STATS, _load_cifar10_arrays
from edge_common.deploy import count_macs, count_params, file_size_mb, make_session, profile_onnx
from q1_distillation.ckpt import TEACHER_CKPT, load_model, student_ckpt_path
from q1_distillation.models import STUDENTS
from q2_nas.train_best import NAS_CKPT, load_nas_model


def _test_arrays(dataset: str):
    if dataset == "cifar100":
        d = np.load(DATA_DIR / "cifar100.npz")
        x, y = d["x_test"], d["y_test"]
    else:
        _, _, x, y = _load_cifar10_arrays()
    mean, std = (np.array(v, dtype=np.float32).reshape(1, 3, 1, 1) * 255 for v in STATS[dataset])
    return ((x.transpose(0, 3, 1, 2).astype(np.float32) - mean) / std).astype(np.float32), y


def onnx_accuracy(path: Path, x: np.ndarray, y: np.ndarray, batch: int = 500) -> float:
    sess = make_session(path, threads=4)
    name = sess.get_inputs()[0].name
    correct = sum(int((sess.run(None, {name: x[i:i + batch]})[0].argmax(1) == y[i:i + batch]).sum())
                  for i in range(0, len(x), batch))
    return 100.0 * correct / len(x)


@torch.no_grad()
def torch_latency_ms(model: torch.nn.Module, iters: int = 100, threads: int = 1) -> float:
    torch.set_num_threads(threads)
    model = model.eval().cpu()
    x = torch.randn(1, 3, 32, 32)
    for _ in range(20):
        model(x)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def collect_models() -> List[Dict]:
    """Every (model, ONNX file, checkpoint) that exists on disk."""
    specs = []
    if (ONNX_DIR / "resnet50_teacher.onnx").exists():
        specs.append(dict(name="resnet50", family="teacher", variant="teacher", dataset="cifar100",
                          onnx=ONNX_DIR / "resnet50_teacher.onnx", ckpt=TEACHER_CKPT))
    for arch in STUDENTS:
        for variant in ("baseline", "kd", "kd_logit"):
            p = ONNX_DIR / f"{arch}_{variant}.onnx"
            if p.exists():
                specs.append(dict(name=arch, family="student", variant=variant, dataset="cifar100",
                                  onnx=p, ckpt=student_ckpt_path(arch, variant)))
    if (ONNX_DIR / "nas_best.onnx").exists():
        specs.append(dict(name="nas_best", family="nas", variant="nas", dataset="cifar10",
                          onnx=ONNX_DIR / "nas_best.onnx", ckpt=NAS_CKPT))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200, help="timed runs per latency measurement")
    ap.add_argument("--warmup", type=int, default=30)
    a = ap.parse_args()

    specs = collect_models()
    if not specs:
        raise SystemExit("no ONNX models found in onnx_models/ - run the export stages first")
    arrays = {}
    rows: List[Dict] = []
    print(f"benchmarking {len(specs)} models on {platform.processor() or platform.machine()} "
          f"({platform.platform()}), ONNX Runtime CPU", flush=True)
    for s in specs:
        if s["dataset"] not in arrays:
            arrays[s["dataset"]] = _test_arrays(s["dataset"])
        x, y = arrays[s["dataset"]]
        if s["family"] == "nas":
            model, ck = load_nas_model(s["ckpt"])
        else:
            model, ck = load_model(s["ckpt"])
        lat1 = profile_onnx(s["onnx"], threads=1, warmup=a.warmup, iters=a.iters)
        lat4 = profile_onnx(s["onnx"], threads=4, warmup=a.warmup, iters=a.iters)
        thr = profile_onnx(s["onnx"], threads=4, warmup=10, iters=max(20, a.iters // 4),
                           input_shape=(32, 3, 32, 32))
        row = {
            "model": s["name"], "family": s["family"], "variant": s["variant"], "dataset": s["dataset"],
            "onnx_top1_acc": round(onnx_accuracy(s["onnx"], x, y), 2),
            "torch_top1_acc": round(float(ck["meta"].get("test_acc", float("nan"))), 2),
            "params_M": round(count_params(model) / 1e6, 3),
            "macs_M": round(count_macs(model) / 1e6, 1),
            "onnx_size_mb": round(file_size_mb(s["onnx"]), 3),
            "latency_ms_median_1thread": round(lat1["latency_ms_median"], 3),
            "latency_ms_mean_1thread": round(lat1["latency_ms_mean"], 3),
            "latency_ms_p95_1thread": round(lat1["latency_ms_p95"], 3),
            "latency_ms_median_4threads": round(lat4["latency_ms_median"], 3),
            "throughput_img_s_batch32_4threads": round(thr["throughput_ips"], 1),
            "peak_mem_mb_batch1": round(lat1["peak_mem_mb"], 1),
            "torch_cpu_latency_ms_1thread": round(torch_latency_ms(model, a.iters), 3),
        }
        rows.append(row)
        print(f"  {s['name']:20s} {s['variant']:9s} acc {row['onnx_top1_acc']:6.2f}%  "
              f"{row['params_M']:6.2f}M params  {row['onnx_size_mb']:7.2f} MB  "
              f"{row['latency_ms_median_1thread']:7.3f} ms (1t)  {row['latency_ms_median_4threads']:7.3f} ms (4t)  "
              f"{row['throughput_img_s_batch32_4threads']:8.0f} img/s  mem +{row['peak_mem_mb_batch1']:.1f} MB", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "benchmark_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump({"machine": {"platform": platform.platform(), "processor": platform.processor(),
                               "machine": platform.machine(), "python": platform.python_version(),
                               "torch": torch.__version__},
                   "rows": rows}, f, indent=2)

    # ---- knowledge-distillation gains per student
    gains = []
    by = {(r["model"], r["variant"]): r for r in rows}
    teacher = by.get(("resnet50", "teacher"))
    for arch in STUDENTS:
        base = by.get((arch, "baseline"))
        for variant in ("kd", "kd_logit"):
            kd = by.get((arch, variant))
            if base and kd:
                g = {"student": arch, "kd_variant": variant, "baseline_acc": base["onnx_top1_acc"],
                     "distilled_acc": kd["onnx_top1_acc"],
                     "kd_gain_pp": round(kd["onnx_top1_acc"] - base["onnx_top1_acc"], 2)}
                if teacher:
                    g.update(teacher_acc=teacher["onnx_top1_acc"],
                             size_reduction_x=round(teacher["onnx_size_mb"] / kd["onnx_size_mb"], 1),
                             speedup_x_1thread=round(teacher["latency_ms_median_1thread"]
                                                     / kd["latency_ms_median_1thread"], 1))
                gains.append(g)
    if gains:
        with open(RESULTS_DIR / "kd_gains.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gains[0]))
            w.writeheader()
            w.writerows(gains)
        print("\nknowledge-distillation gains (top-1, percentage points):")
        for g in gains:
            print(f"  {g['student']:20s} {g['kd_variant']:9s} baseline {g['baseline_acc']:6.2f} -> "
                  f"distilled {g['distilled_acc']:6.2f}  ({g['kd_gain_pp']:+.2f} pp)")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
