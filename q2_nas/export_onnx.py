"""Export the trained NAS model to ONNX and check it against PyTorch.

    python -m q2_nas.export_onnx
"""
from __future__ import annotations

import argparse

from edge_common import ONNX_DIR
from edge_common.deploy import export_onnx, file_size_mb, verify_onnx
from q2_nas.train_best import NAS_CKPT, load_nas_model

NAS_ONNX = ONNX_DIR / "nas_best.onnx"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(NAS_CKPT))
    ap.add_argument("--out", default=str(NAS_ONNX))
    a = ap.parse_args()
    model, ckpt = load_nas_model(a.ckpt)
    path = export_onnx(model, a.out)
    diff = verify_onnx(model, path)
    print(f"exported {path} ({file_size_mb(path):.3f} MB); max |torch - onnxruntime| logit diff = {diff:.2e}")


if __name__ == "__main__":
    main()
