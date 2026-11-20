"""Model-complexity counting, ONNX export / verification and ONNX Runtime CPU profiling."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn

from edge_common import ROOT


# --------------------------------------------------------------------------- complexity
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: Sequence[int] = (1, 3, 32, 32)) -> int:
    """Multiply-accumulates of Conv2d and Linear layers for one forward pass.

    Runs on a CPU *copy* of the model so the caller's model (device, train/eval mode)
    is left untouched.
    """
    probe = copy.deepcopy(model).eval().cpu()
    total = [0]

    def conv_hook(m: nn.Conv2d, _, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        total[0] += out.numel() * k // out.shape[0]

    def linear_hook(m: nn.Linear, _, out):
        total[0] += m.in_features * m.out_features

    for m in probe.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(conv_hook)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(linear_hook)
    probe(torch.zeros(*input_shape))
    return int(total[0])


# --------------------------------------------------------------------------- ONNX
def export_onnx(model: nn.Module, path: Path, input_shape: Sequence[int] = (1, 3, 32, 32),
                opset: int = 17, dynamic_batch: bool = True) -> Path:
    """Export ``model`` (eval mode, CPU) to ONNX with a dynamic batch axis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = copy.deepcopy(model).eval().cpu()
    dummy = torch.randn(*input_shape)
    kwargs = {}
    if dynamic_batch:
        kwargs["dynamic_axes"] = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["logits"],
                      opset_version=opset, do_constant_folding=True, dynamo=False, **kwargs)
    return path


def make_session(path: Path, threads: int = 1):
    """CPU-only ONNX Runtime session with a fixed intra-op thread count."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def verify_onnx(model: nn.Module, path: Path, input_shape: Sequence[int] = (4, 3, 32, 32),
                atol: float = 1e-4) -> float:
    """Max abs difference between PyTorch (CPU) and ONNX Runtime logits on random input."""
    model = copy.deepcopy(model).eval().cpu()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        ref = model(x).numpy()
    out = make_session(path).run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    if diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff:.2e} (> {atol})")
    return diff


# --------------------------------------------------------------------------- profiling
def file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def profile_onnx(path: Path, threads: int = 1, warmup: int = 20, iters: int = 100,
                 input_shape: Sequence[int] = (1, 3, 32, 32), timeout: int = 300) -> Dict[str, float]:
    """Measure CPU latency and peak inference memory of an ONNX model in a fresh process.

    A new interpreter is used for every model so that ``ru_maxrss`` (a process-wide
    high-water mark) reflects only that model.  Returns latency statistics in ms
    and ``peak_mem_mb`` (RSS increase caused by session creation + inference).
    """
    cmd = [sys.executable, "-m", "edge_common.profile_worker", str(path),
           "--threads", str(threads), "--warmup", str(warmup), "--iters", str(iters),
           "--shape", *map(str, input_shape)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"profiling failed for {path}:\n{res.stderr[-2000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])
