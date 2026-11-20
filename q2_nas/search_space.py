"""Searchable CNN architecture space for edge deployment (question 2).

An architecture is a plain JSON-serialisable ``dict``::

    {"resolution": 24,              # input resolution fed to the backbone (in-graph resize from 32x32)
     "stem_channels": 24,
     "stages": [                    # 2 - 4 stages; the first block of every stage after the first has stride 2
        {"depth": 2, "channels": 32, "kernel": 3, "block": "dwsep", "activation": "relu6"},
        ...]}

Searchable dimensions (all required by the assignment):

* variable convolutional depth        -> number of stages (2-4) and blocks per stage (1-3)
* variable numbers of filters         -> stem width and per-stage channels
* different kernel sizes              -> 3 / 5 / 7 per stage
* depthwise-separable conv blocks     -> per-stage block type: "conv" (regular) or "dwsep"
* different activation functions      -> per-stage ReLU / ReLU6 / Hardswish / SiLU / LeakyReLU
* variable input resolutions          -> 16 / 24 / 32
"""
from __future__ import annotations

import copy
import json
import random
from typing import Dict, List

RESOLUTIONS = [16, 24, 32]
STEM_CHANNELS = [16, 24, 32]
NUM_STAGES = [2, 3, 4]
DEPTHS = [1, 2, 3]
CHANNELS = [16, 24, 32, 48, 64, 96, 128, 160]
KERNELS = [3, 5, 7]
BLOCKS = ["conv", "dwsep"]
ACTIVATIONS = ["relu", "relu6", "hardswish", "silu", "leaky_relu"]

Config = Dict


def _random_stage(rng: random.Random) -> Dict:
    return {"depth": rng.choice(DEPTHS), "channels": rng.choice(CHANNELS),
            "kernel": rng.choice(KERNELS), "block": rng.choice(BLOCKS),
            "activation": rng.choice(ACTIVATIONS)}


def random_config(rng: random.Random) -> Config:
    return {"resolution": rng.choice(RESOLUTIONS), "stem_channels": rng.choice(STEM_CHANNELS),
            "stages": [_random_stage(rng) for _ in range(rng.choice(NUM_STAGES))]}


def validate(cfg: Config) -> None:
    assert cfg["resolution"] in RESOLUTIONS and cfg["stem_channels"] in STEM_CHANNELS
    assert len(cfg["stages"]) in NUM_STAGES
    for s in cfg["stages"]:
        assert s["depth"] in DEPTHS and s["channels"] in CHANNELS and s["kernel"] in KERNELS
        assert s["block"] in BLOCKS and s["activation"] in ACTIVATIONS


def config_key(cfg: Config) -> str:
    """Canonical string used to avoid evaluating the same architecture twice."""
    return json.dumps(cfg, sort_keys=True)


def mutate(cfg: Config, rng: random.Random, strength: float = 0.25) -> Config:
    """Return a mutated copy: each gene changes with probability ``strength`` (>= 1 change)."""
    child = copy.deepcopy(cfg)
    changed = False

    def maybe(p=strength):
        nonlocal changed
        hit = rng.random() < p
        changed = changed or hit
        return hit

    if maybe():
        child["resolution"] = rng.choice([r for r in RESOLUTIONS if r != child["resolution"]])
    if maybe():
        child["stem_channels"] = rng.choice([c for c in STEM_CHANNELS if c != child["stem_channels"]])
    if maybe(strength / 2):                                            # structural: depth of the network
        n = len(child["stages"])
        options = [m for m in NUM_STAGES if m != n and abs(m - n) == 1]
        new_n = rng.choice(options)
        if new_n > n:
            child["stages"].append(_random_stage(rng))
        else:
            child["stages"].pop()
    for stage in child["stages"]:
        for gene, space in (("depth", DEPTHS), ("channels", CHANNELS), ("kernel", KERNELS),
                            ("block", BLOCKS), ("activation", ACTIVATIONS)):
            if maybe():
                stage[gene] = rng.choice([v for v in space if v != stage[gene]])
    if not changed:                                                    # guarantee a real mutation
        stage = rng.choice(child["stages"])
        gene, space = rng.choice([("depth", DEPTHS), ("channels", CHANNELS), ("kernel", KERNELS),
                                  ("block", BLOCKS), ("activation", ACTIVATIONS)])
        stage[gene] = rng.choice([v for v in space if v != stage[gene]])
    return child


def crossover(a: Config, b: Config, rng: random.Random) -> Config:
    """Uniform crossover of global genes and of the stages the parents have in common."""
    child = {"resolution": rng.choice([a["resolution"], b["resolution"]]),
             "stem_channels": rng.choice([a["stem_channels"], b["stem_channels"]])}
    n = rng.choice([len(a["stages"]), len(b["stages"])])
    stages: List[Dict] = []
    for i in range(n):
        donors = [p["stages"][i] for p in (a, b) if i < len(p["stages"])]
        stages.append(copy.deepcopy(rng.choice(donors)))
    child["stages"] = stages
    return child


def describe(cfg: Config) -> str:
    stages = " | ".join(f"{s['block']}{s['kernel']}x{s['depth']}-{s['channels']}-{s['activation']}"
                        for s in cfg["stages"])
    return f"res{cfg['resolution']} stem{cfg['stem_channels']} :: {stages}"
