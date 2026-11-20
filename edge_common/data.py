"""CIFAR-10 / CIFAR-100 loading with fully on-device batching and augmentation.

The whole dataset is stored as a uint8 tensor on the compute device (150 MB for
CIFAR), so there is no DataLoader / worker overhead.  Augmentation (random crop
with 4px padding + horizontal flip) is applied per-sample on the GPU.
"""
from __future__ import annotations

import pickle
from typing import Iterator, Optional, Tuple

import numpy as np
import torch

from edge_common import DATA_DIR

STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _load_cifar10_arrays():
    folder = DATA_DIR / "cifar-10-batches-py"
    xs, ys = [], []
    for name in [f"data_batch_{i}" for i in range(1, 6)]:
        with open(folder / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32))
        ys.extend(d[b"labels"])
    with open(folder / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x_test = d[b"data"].reshape(-1, 3, 32, 32)
    y_test = np.array(d[b"labels"], dtype=np.int64)
    x_train = np.concatenate(xs).transpose(0, 2, 3, 1)
    return x_train, np.array(ys, dtype=np.int64), x_test.transpose(0, 2, 3, 1), y_test


def _load_cifar100_arrays():
    npz = DATA_DIR / "cifar100.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"{npz} not found - run `python -m edge_common.prepare_cifar100` first."
        )
    d = np.load(npz)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def get_device(preference: str = "auto") -> torch.device:
    """cuda > mps > cpu unless a specific device is requested."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CifarData:
    """In-memory CIFAR dataset living on ``device``.

    Parameters
    ----------
    name:        "cifar10" or "cifar100"
    val_size:    number of training images held out as a validation split (0 = none).
                 The split is stratification-free but deterministic (seeded).
    train_limit: optionally keep only the first N (shuffled) training images, used
                 by the NAS proxy training to make candidate evaluation cheap.
    """

    def __init__(self, name: str, device: torch.device, val_size: int = 0,
                 train_limit: Optional[int] = None, seed: int = 0):
        if name not in STATS:
            raise ValueError(f"unknown dataset {name!r}")
        self.name, self.device = name, device
        self.num_classes = NUM_CLASSES[name]
        loader = _load_cifar10_arrays if name == "cifar10" else _load_cifar100_arrays
        x_tr, y_tr, x_te, y_te = loader()

        rng = np.random.RandomState(seed)
        order = rng.permutation(len(x_tr))
        x_tr, y_tr = x_tr[order], y_tr[order]
        if val_size:
            x_val, y_val = x_tr[:val_size], y_tr[:val_size]
            x_tr, y_tr = x_tr[val_size:], y_tr[val_size:]
        else:
            x_val = y_val = None
        if train_limit:
            x_tr, y_tr = x_tr[:train_limit], y_tr[:train_limit]

        to_dev = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        # stored NHWC-uint8 -> NCHW-uint8 on device
        self.x_train = to_dev(x_tr).permute(0, 3, 1, 2).contiguous()
        self.y_train = to_dev(y_tr)
        self.x_test = to_dev(x_te).permute(0, 3, 1, 2).contiguous()
        self.y_test = to_dev(y_te)
        self.x_val = to_dev(x_val).permute(0, 3, 1, 2).contiguous() if x_val is not None else None
        self.y_val = to_dev(y_val) if y_val is not None else None

        mean, std = STATS[name]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1) * 255.0
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1) * 255.0

    # ------------------------------------------------------------------ helpers
    def normalise(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.float() - self.mean) / self.std

    @staticmethod
    def _augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
        """Per-sample random crop (reflect-free zero pad) + horizontal flip, on device."""
        b, c, h, w = x.shape
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        ar_h = torch.arange(h, device=x.device)
        ar_w = torch.arange(w, device=x.device)
        iy = (oy[:, None] + ar_h[None, :])[:, None, :, None]          # B,1,H,1
        ix = (ox[:, None] + ar_w[None, :])[:, None, None, :]          # B,1,1,W
        bi = torch.arange(b, device=x.device)[:, None, None, None]
        ci = torch.arange(c, device=x.device)[None, :, None, None]
        out = xp[bi, ci, iy, ix]
        flip = torch.rand(b, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    # ------------------------------------------------------------------ loaders
    def train_batches(self, batch_size: int, augment: bool = True,
                      drop_last: bool = True) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.x_train.shape[0]
        perm = torch.randperm(n, device=self.device)
        stop = (n // batch_size) * batch_size if drop_last else n
        for i in range(0, stop, batch_size):
            idx = perm[i:i + batch_size]
            x = self.normalise(self.x_train[idx])
            if augment:
                x = self._augment(x)
            yield x, self.y_train[idx]

    def steps_per_epoch(self, batch_size: int, drop_last: bool = True) -> int:
        n = self.x_train.shape[0]
        return n // batch_size if drop_last else -(-n // batch_size)

    def eval_batches(self, split: str = "test", batch_size: int = 500):
        x, y = {"test": (self.x_test, self.y_test), "val": (self.x_val, self.y_val),
                "train": (self.x_train, self.y_train)}[split]
        if x is None:
            raise ValueError(f"split {split!r} is not available")
        for i in range(0, x.shape[0], batch_size):
            yield self.normalise(x[i:i + batch_size]), y[i:i + batch_size]

    def numpy_split(self, split: str = "test"):
        """Normalised float32 NCHW numpy arrays + labels (for ONNX Runtime evaluation)."""
        x, y = {"test": (self.x_test, self.y_test), "val": (self.x_val, self.y_val)}[split]
        return self.normalise(x).cpu().numpy().astype(np.float32), y.cpu().numpy()
