"""Subprocess worker: measure ONNX Runtime CPU latency and peak memory for one model.

Run by ``edge_common.deploy.profile_onnx`` (one fresh process per model so the
RSS high-water mark is attributable to that model).  Prints one JSON line.
"""
import argparse
import json
import platform
import resource
import time

import numpy as np
import onnxruntime as ort


def max_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return r / (1024 * 1024) if platform.system() == "Darwin" else r / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--shape", type=int, nargs=4, default=[1, 3, 32, 32])
    a = ap.parse_args()

    base_rss = max_rss_mb()                       # after importing numpy + onnxruntime
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(a.model, opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(*a.shape).astype(np.float32)

    for _ in range(a.warmup):
        sess.run(None, {name: x})
    times = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    peak = max_rss_mb()
    print(json.dumps({
        "latency_ms_mean": float(t.mean()), "latency_ms_median": float(np.median(t)),
        "latency_ms_p95": float(np.percentile(t, 95)), "latency_ms_std": float(t.std()),
        "throughput_ips": float(a.shape[0] * 1000.0 / np.median(t)),
        "peak_mem_mb": float(peak - base_rss), "peak_rss_mb": float(peak),
        "threads": a.threads, "batch": a.shape[0],
    }))


if __name__ == "__main__":
    main()
