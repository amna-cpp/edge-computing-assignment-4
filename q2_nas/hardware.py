"""Hardware-aware evaluation of a candidate architecture.

Every candidate is exported to ONNX (the deployment format) and then, in a fresh
process, run with ONNX Runtime on the CPU (1 intra-op thread, batch 1 - a typical edge
setting).  Nothing is estimated from FLOPs or parameter counts: the numbers below are
measured on this machine.

    latency_ms     median wall-clock time of one inference (after warm-up)
    size_mb        size of the ONNX file on disk (the deployable model storage)
    peak_mem_mb    RSS growth of the inference process while creating the session and
                   running the model (resource.getrusage; where supported)
"""
from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch.nn as nn

from edge_common.deploy import count_macs, count_params, export_onnx, file_size_mb, profile_onnx


@dataclass
class Constraints:
    """Predefined deployment budget; a candidate that violates any limit is rejected."""
    max_latency_ms: float = 1.5       # batch-1 CPU latency (ONNX Runtime, 1 thread)
    max_size_mb: float = 1.0          # ONNX file size
    max_peak_mem_mb: float = 16.0     # inference peak RSS (the bare ORT process already costs ~9.5 MB)

    def violations(self, m: Dict[str, float]) -> List[str]:
        out = []
        if m["latency_ms"] > self.max_latency_ms:
            out.append(f"latency {m['latency_ms']:.2f}ms > {self.max_latency_ms}ms")
        if m["size_mb"] > self.max_size_mb:
            out.append(f"size {m['size_mb']:.2f}MB > {self.max_size_mb}MB")
        if m["peak_mem_mb"] > self.max_peak_mem_mb:
            out.append(f"memory {m['peak_mem_mb']:.1f}MB > {self.max_peak_mem_mb}MB")
        return out

    def asdict(self) -> Dict[str, float]:
        return asdict(self)


def measure_hardware(model: nn.Module, workdir: Optional[Path] = None, threads: int = 1,
                     warmup: int = 15, iters: int = 60, keep_onnx: Optional[Path] = None
                     ) -> Dict[str, float]:
    """Export ``model`` to ONNX and measure latency, size and peak memory on the CPU."""
    model = model.eval().cpu()
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        onnx_path = Path(tmp) / "candidate.onnx"
        export_onnx(model, onnx_path, dynamic_batch=False)
        prof = profile_onnx(onnx_path, threads=threads, warmup=warmup, iters=iters)
        size = file_size_mb(onnx_path)
        if keep_onnx is not None:
            keep_onnx.parent.mkdir(parents=True, exist_ok=True)
            keep_onnx.write_bytes(onnx_path.read_bytes())
    return {"latency_ms": prof["latency_ms_median"], "latency_ms_p95": prof["latency_ms_p95"],
            "size_mb": size, "peak_mem_mb": prof["peak_mem_mb"],
            "params": count_params(model), "macs": count_macs(model)}
