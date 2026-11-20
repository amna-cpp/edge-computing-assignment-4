"""Inference pipeline for the NAS model using ONNX Runtime (CPU).

    python -m q2_nas.inference --image path/to/picture.png        # classify one image
    python -m q2_nas.inference --evaluate                         # CIFAR-10 test accuracy + latency
    python -m q2_nas.inference --samples 8                        # show predictions for test images

The exported graph takes a normalised float32 ``1x3x32x32`` tensor; any input-resolution change
chosen by the search is performed inside the graph.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
from PIL import Image

from edge_common import ONNX_DIR
from edge_common.data import STATS
from edge_common.deploy import make_session

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
NAS_ONNX = ONNX_DIR / "nas_best.onnx"


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> normalised float32 [1, 3, 32, 32]."""
    arr = np.asarray(img.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0
    mean, std = (np.array(v, dtype=np.float32).reshape(1, 1, 3) for v in STATS["cifar10"])
    return ((arr - mean) / std).transpose(2, 0, 1)[None]


class NASPredictor:
    def __init__(self, onnx_path: Path = NAS_ONNX, threads: int = 1):
        self.session = make_session(onnx_path, threads)
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]

    def predict(self, img: Image.Image, top_k: int = 3):
        logits = self.logits(preprocess(img))[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = np.argsort(-probs)[:top_k]
        return [(CIFAR10_CLASSES[i], float(probs[i])) for i in idx]


def evaluate_test_set(pred: NASPredictor, batch: int = 500):
    from edge_common.data import _load_cifar10_arrays
    _, _, x, y = _load_cifar10_arrays()
    mean, std = (np.array(v, dtype=np.float32).reshape(1, 3, 1, 1) * 255 for v in STATS["cifar10"])
    x = (x.transpose(0, 3, 1, 2).astype(np.float32) - mean) / std
    t0 = time.perf_counter()
    correct = sum(int((pred.logits(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
                  for i in range(0, len(x), batch))
    return 100.0 * correct / len(x), time.perf_counter() - t0


def main(argv: Sequence[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default=str(NAS_ONNX))
    ap.add_argument("--image")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    pred = NASPredictor(Path(a.onnx), a.threads)

    if a.image:
        print(f"{a.image}:")
        for name, p in pred.predict(Image.open(a.image)):
            print(f"  {name:12s} {100 * p:5.1f}%")
    if a.samples:
        from edge_common.data import _load_cifar10_arrays
        _, _, x, y = _load_cifar10_arrays()
        for i in range(a.samples):
            top = pred.predict(Image.fromarray(x[i]), 1)[0]
            ok = "OK " if top[0] == CIFAR10_CLASSES[y[i]] else "BAD"
            print(f"  test[{i}] true={CIFAR10_CLASSES[y[i]]:10s} pred={top[0]:10s} ({100 * top[1]:5.1f}%) {ok}")
    if a.evaluate:
        acc, secs = evaluate_test_set(pred)
        print(f"CIFAR-10 test accuracy (ONNX Runtime, CPU, {a.threads} thread): {acc:.2f}%  "
              f"[10000 images in {secs:.1f}s]")
    if not (a.image or a.evaluate or a.samples):
        ap.print_help()


if __name__ == "__main__":
    main()
