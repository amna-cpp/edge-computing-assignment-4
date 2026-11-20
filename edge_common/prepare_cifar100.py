"""One-time conversion of the CIFAR-100 parquet files into a compact ``cifar100.npz``.

The parquet files (train.parquet / test.parquet) were obtained from the
``uoft-cs/cifar100`` dataset on Hugging Face.  ``pyarrow`` is only required for
this conversion step, never for training or inference.

Usage:
    python -m edge_common.prepare_cifar100
"""
import io
import sys

import numpy as np
import pandas as pd
from PIL import Image

from edge_common import DATA_DIR


def _decode(df: pd.DataFrame):
    """Decode the PNG bytes of every row into a (N, 32, 32, 3) uint8 array."""
    images = np.empty((len(df), 32, 32, 3), dtype=np.uint8)
    for i, cell in enumerate(df["img"]):
        images[i] = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    labels = df["fine_label"].to_numpy(dtype=np.int64)
    return images, labels


def main() -> int:
    src = DATA_DIR / "hf_cifar100"
    out = DATA_DIR / "cifar100.npz"
    arrays = {}
    for split in ("train", "test"):
        path = src / f"{split}.parquet"
        if not path.exists():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1
        x, y = _decode(pd.read_parquet(path))
        arrays[f"x_{split}"], arrays[f"y_{split}"] = x, y
        print(f"{split}: images {x.shape} labels {y.shape} classes {len(np.unique(y))}")
    np.savez_compressed(out, **arrays)
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # class names (stored in the parquet schema metadata) for readable inference output
    import json
    import pyarrow.parquet as pq
    meta = pq.read_schema(src / "test.parquet").metadata
    names = json.loads(meta[b"huggingface"])["info"]["features"]["fine_label"]["names"]
    (DATA_DIR / "cifar100_classes.json").write_text(json.dumps(names))
    print(f"saved {len(names)} class names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
