"""Unified inference pipeline for every trained student (baseline and distilled).

One interface, three student architectures x two training variants, two back-ends:

    python -m q1_distillation.inference --image cat.png                       # all 6 models, PyTorch
    python -m q1_distillation.inference --image cat.png --backend onnx        # all 6 models, ONNX Runtime
    python -m q1_distillation.inference --arch mobilenetv2 --variant kd --image cat.png
    python -m q1_distillation.inference --evaluate                            # CIFAR-100 test accuracy of every model
    python -m q1_distillation.inference --samples 5                           # predictions on CIFAR-100 test images

Programmatic use::

    from q1_distillation.inference import StudentPredictor
    p = StudentPredictor("squeezenet", "kd", backend="onnx")
    p.predict(PIL.Image.open("x.png"), top_k=3)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from edge_common import DATA_DIR, ONNX_DIR
from edge_common.data import STATS
from q1_distillation.ckpt import load_model, student_ckpt_path
from q1_distillation.models import STUDENTS

VARIANTS = ("baseline", "kd")
_MEAN = np.array(STATS["cifar100"][0], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array(STATS["cifar100"][1], dtype=np.float32).reshape(1, 3, 1, 1)


def class_names() -> List[str]:
    return json.loads((DATA_DIR / "cifar100_classes.json").read_text())


def preprocess(images: Sequence[Image.Image]) -> np.ndarray:
    """PIL images -> normalised float32 NCHW batch (resize to 32x32, CIFAR-100 statistics)."""
    arr = np.stack([np.asarray(im.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.uint8)
                    for im in images]).transpose(0, 3, 1, 2).astype(np.float32)
    return (arr - _MEAN * 255.0) / (_STD * 255.0)


def onnx_path(arch: str, variant: str) -> Path:
    return ONNX_DIR / f"{arch}_{variant}.onnx"


class StudentPredictor:
    """Loads one student (``arch`` x ``variant``) with the PyTorch or ONNX Runtime back-end."""

    def __init__(self, arch: str, variant: str = "kd", backend: str = "torch", threads: int = 1):
        if arch not in STUDENTS:
            raise ValueError(f"arch must be one of {STUDENTS}")
        self.arch, self.variant, self.backend = arch, variant, backend
        if backend == "torch":
            import torch
            self._torch = torch
            self.model, _ = load_model(student_ckpt_path(arch, variant))
            torch.set_num_threads(threads)
        elif backend == "onnx":
            from edge_common.deploy import make_session
            self.session = make_session(onnx_path(arch, variant), threads)
            self.input_name = self.session.get_inputs()[0].name
        else:
            raise ValueError("backend must be 'torch' or 'onnx'")

    @property
    def name(self) -> str:
        return f"{self.arch}/{self.variant}/{self.backend}"

    def logits(self, batch: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            with self._torch.no_grad():
                return self.model(self._torch.from_numpy(batch)).numpy()
        return self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]

    def predict(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        logits = self.logits(preprocess([image]))[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        names = class_names()
        return [(names[i], float(probs[i])) for i in np.argsort(-probs)[:top_k]]

    def evaluate(self, batch_size: int = 500) -> Tuple[float, float]:
        """CIFAR-100 test top-1 (%) and wall-clock seconds for the 10 000 test images."""
        d = np.load(DATA_DIR / "cifar100.npz")
        x = (d["x_test"].transpose(0, 3, 1, 2).astype(np.float32) - _MEAN * 255.0) / (_STD * 255.0)
        y = d["y_test"]
        t0 = time.perf_counter()
        correct = sum(int((self.logits(x[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum())
                      for i in range(0, len(x), batch_size))
        return 100.0 * correct / len(x), time.perf_counter() - t0


def available_models(backend: str, archs: Sequence[str], variants: Sequence[str]) -> List[Tuple[str, str]]:
    found = []
    for arch in archs:
        for variant in variants:
            path = student_ckpt_path(arch, variant) if backend == "torch" else onnx_path(arch, variant)
            if path.exists():
                found.append((arch, variant))
            else:
                print(f"  (skipping {arch}/{variant}: {path.name} not found)")
    return found


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="all", choices=["all", *STUDENTS])
    ap.add_argument("--variant", default="all", help="baseline | kd | kd_logit | all (= baseline + kd)")
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--image", help="path of an image to classify")
    ap.add_argument("--samples", type=int, default=0, help="classify the first N CIFAR-100 test images")
    ap.add_argument("--evaluate", action="store_true", help="CIFAR-100 test accuracy of every selected model")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    if not (a.image or a.samples or a.evaluate):
        ap.print_help()
        return

    archs = STUDENTS if a.arch == "all" else (a.arch,)
    variants = VARIANTS if a.variant == "all" else (a.variant,)
    models = [StudentPredictor(ar, va, a.backend, a.threads)
              for ar, va in available_models(a.backend, archs, variants)]

    if a.image:
        img = Image.open(a.image)
        print(f"image: {a.image}")
        for m in models:
            top = ", ".join(f"{n} {100 * p:.1f}%" for n, p in m.predict(img, a.top_k))
            print(f"  {m.name:38s} {top}")
    if a.samples:
        d = np.load(DATA_DIR / "cifar100.npz")
        names = class_names()
        for i in range(a.samples):
            print(f"test image {i}: true = {names[d['y_test'][i]]}")
            for m in models:
                n, p = m.predict(Image.fromarray(d["x_test"][i]), 1)[0]
                print(f"  {m.name:38s} {n:16s} {100 * p:5.1f}%  {'OK' if n == names[d['y_test'][i]] else 'wrong'}")
    if a.evaluate:
        print(f"CIFAR-100 test accuracy ({a.backend} back-end, {a.threads} CPU thread)")
        for m in models:
            acc, secs = m.evaluate()
            print(f"  {m.name:38s} top-1 {acc:5.2f}%   (10000 images in {secs:5.1f}s)")


if __name__ == "__main__":
    main()
