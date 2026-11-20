"""Stand-alone CPU inference with ONNX Runtime (no PyTorch needed at run time).

    python -m q1_distillation.onnx_infer --model onnx_models/mobilenetv2_kd.onnx --image cat.png
    python -m q1_distillation.onnx_infer --model onnx_models/squeezenet_kd.onnx --evaluate
    python -m q1_distillation.onnx_infer --all --evaluate          # every exported student

Only numpy, Pillow and onnxruntime are used - this is what would run on the edge device.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MEAN = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32).reshape(1, 3, 1, 1) * 255
STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32).reshape(1, 3, 1, 1) * 255


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads                 # CPU only, fixed thread count
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def classify(sess: ort.InferenceSession, image: Image.Image, top_k: int = 5):
    x = np.asarray(image.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    x = ((x.transpose(2, 0, 1)[None] - MEAN) / STD).astype(np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    names = json.loads((ROOT / "data" / "cifar100_classes.json").read_text())
    return [(names[i], float(p[i])) for i in np.argsort(-p)[:top_k]]


def evaluate(sess: ort.InferenceSession, batch: int = 500):
    d = np.load(ROOT / "data" / "cifar100.npz")
    x = ((d["x_test"].transpose(0, 3, 1, 2).astype(np.float32) - MEAN) / STD).astype(np.float32)
    y = d["y_test"]
    name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    correct = sum(int((sess.run(None, {name: x[i:i + batch]})[0].argmax(1) == y[i:i + batch]).sum())
                  for i in range(0, len(x), batch))
    return 100.0 * correct / len(x), time.perf_counter() - t0


def single_image_latency(sess: ort.InferenceSession, runs: int = 200) -> float:
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(20):
        sess.run(None, {name: x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--all", action="store_true", help="run on every *_baseline / *_kd model in onnx_models/")
    ap.add_argument("--image")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()

    if a.all:
        paths = sorted(p for p in (ROOT / "onnx_models").glob("*.onnx")
                       if p.stem.endswith(("_baseline", "_kd", "_kd_logit")))
    elif a.model:
        paths = [Path(a.model)]
    else:
        ap.error("give --model or --all")
    for path in paths:
        sess = make_session(path, a.threads)
        print(f"== {path.name}  (providers: {sess.get_providers()})")
        if a.image:
            print("   " + ", ".join(f"{n} {100 * p:.1f}%" for n, p in classify(sess, Image.open(a.image))))
        if a.evaluate:
            acc, secs = evaluate(sess)
            print(f"   CIFAR-100 test top-1 {acc:.2f}%  ({secs:.1f}s for 10000 images)  "
                  f"batch-1 latency {single_image_latency(sess):.2f} ms")


if __name__ == "__main__":
    main()
