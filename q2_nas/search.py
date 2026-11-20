"""Hardware-aware evolutionary neural architecture search (regularised / aging evolution).

    python -m q2_nas.search --budget 60

Pipeline for every candidate
    1. sample (random) or derive (tournament selection + mutation / crossover) a configuration
    2. generate the network and *measure* it on the CPU: ONNX Runtime latency, ONNX file size,
       peak inference memory                                  (q2_nas/hardware.py)
    3. reject it immediately - without spending any training time - if a deployment constraint
       is violated
    4. otherwise train it on CIFAR-10 (proxy schedule) and score it on a held-out validation split
    5. fitness = val_acc - w_lat * latency/max_latency - w_size * size/max_size
       (accuracy is rewarded, measured latency and storage size are penalised)

Search strategy: aging evolution (Real et al., 2019).  A fixed-size population is kept as a FIFO
queue; each step samples ``--sample`` members, the fittest becomes the parent, a child is produced
(mutation, or crossover with a second tournament winner) and replaces the oldest member.

Everything is logged to results/q2/nas_history.{csv,json}; the best feasible architecture is
written to results/q2/best_architecture.json.  The search is resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import deque
from typing import Dict, List

from edge_common import RESULTS_DIR
from edge_common.data import CifarData, get_device
from edge_common.engine import set_seed
from q2_nas.evaluate import train_candidate
from q2_nas.generator import build_from_config
from q2_nas.hardware import Constraints, measure_hardware
from q2_nas.search_space import (config_key, crossover, describe, mutate, random_config)

OUT_DIR = RESULTS_DIR / "q2"
HISTORY_JSON = OUT_DIR / "nas_history.json"
HISTORY_CSV = OUT_DIR / "nas_history.csv"
BEST_JSON = OUT_DIR / "best_architecture.json"


def fitness(val_acc: float, hw: Dict[str, float], c: Constraints, w_lat: float, w_size: float) -> float:
    return val_acc / 100.0 - w_lat * hw["latency_ms"] / c.max_latency_ms \
        - w_size * hw["size_mb"] / c.max_size_mb


def save_history(history: List[Dict], constraints: Constraints, args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_JSON, "w") as f:
        json.dump({"constraints": constraints.asdict(), "args": vars(args), "history": history}, f, indent=1)
    cols = ["id", "step", "origin", "status", "reject_reason", "val_acc", "fitness", "latency_ms",
            "latency_ms_p95", "size_mb", "peak_mem_mb", "params", "macs", "train_time_s",
            "resolution", "num_stages", "description", "config"]
    with open(HISTORY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in history:
            row = dict(r)
            row["resolution"] = r["config"]["resolution"]
            row["num_stages"] = len(r["config"]["stages"])
            row["description"] = describe(r["config"])
            row["config"] = json.dumps(r["config"])
            w.writerow(row)


def pareto_front(trained: List[Dict]) -> List[Dict]:
    """Non-dominated set w.r.t. (maximise accuracy, minimise latency, minimise size)."""
    front = []
    for r in trained:
        dominated = any(o["val_acc"] >= r["val_acc"] and o["latency_ms"] <= r["latency_ms"]
                        and o["size_mb"] <= r["size_mb"] and
                        (o["val_acc"] > r["val_acc"] or o["latency_ms"] < r["latency_ms"]
                         or o["size_mb"] < r["size_mb"]) for o in trained)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: -r["val_acc"])


def write_best(history: List[Dict], constraints: Constraints) -> Dict:
    trained = [r for r in history if r["status"] == "trained"]
    best = max(trained, key=lambda r: r["fitness"])
    with open(BEST_JSON, "w") as f:
        json.dump({"config": best["config"], "description": describe(best["config"]),
                   "search_val_acc": best["val_acc"], "fitness": best["fitness"],
                   "latency_ms": best["latency_ms"], "size_mb": best["size_mb"],
                   "peak_mem_mb": best["peak_mem_mb"], "params": best["params"],
                   "macs": best["macs"], "found_at_candidate_id": best["id"],
                   "constraints": constraints.asdict()}, f, indent=2)
    with open(OUT_DIR / "pareto_front.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "val_acc", "latency_ms", "size_mb", "peak_mem_mb", "description"])
        for r in pareto_front(trained):
            w.writerow([r["id"], f"{r['val_acc']:.2f}", f"{r['latency_ms']:.3f}",
                        f"{r['size_mb']:.3f}", f"{r['peak_mem_mb']:.1f}", describe(r["config"])])
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=60, help="number of candidates to TRAIN")
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--sample", type=int, default=4, help="tournament size")
    ap.add_argument("--crossover-prob", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=6, help="proxy-training epochs per candidate")
    ap.add_argument("--train-limit", type=int, default=25000, help="training images used per candidate")
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--max-latency-ms", type=float, default=Constraints.max_latency_ms)
    ap.add_argument("--max-size-mb", type=float, default=Constraints.max_size_mb)
    ap.add_argument("--max-peak-mem-mb", type=float, default=Constraints.max_peak_mem_mb)
    ap.add_argument("--w-latency", type=float, default=0.03)
    ap.add_argument("--w-size", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    constraints = Constraints(args.max_latency_ms, args.max_size_mb, args.max_peak_mem_mb)
    device = get_device(args.device)
    data = CifarData("cifar10", device, val_size=args.val_size, train_limit=args.train_limit, seed=args.seed)
    rng = random.Random(args.seed)
    set_seed(args.seed)

    history: List[Dict] = []
    if HISTORY_JSON.exists() and not args.no_resume:
        history = json.load(open(HISTORY_JSON))["history"]
        rng = random.Random(args.seed + len(history))
        print(f"resumed {len(history)} candidates from {HISTORY_JSON.name}", flush=True)
    seen = {config_key(r["config"]) for r in history}
    population: deque = deque([r for r in history if r["status"] == "trained"][-args.population:],
                              maxlen=args.population)
    n_trained = sum(r["status"] == "trained" for r in history)
    n_rejected = sum(r["status"] == "rejected" for r in history)
    print(f"device={device} constraints={constraints.asdict()}\n"
          f"budget={args.budget} population={args.population} epochs={args.epochs} "
          f"train_limit={args.train_limit}", flush=True)

    def evaluate_candidate(cfg, origin: str) -> Dict:
        nonlocal n_trained, n_rejected
        seen.add(config_key(cfg))
        hw = measure_hardware(build_from_config(cfg, data.num_classes))
        rec = {"id": len(history), "step": n_trained, "origin": origin, "config": cfg, **hw}
        reasons = constraints.violations(hw)
        if reasons:
            rec.update(status="rejected", reject_reason="; ".join(reasons), val_acc=None, fitness=None)
            n_rejected += 1
            print(f"  [reject] #{rec['id']:3d} {rec['reject_reason']}", flush=True)
        else:
            res = train_candidate(cfg, data, device, args.epochs, seed=args.seed)
            rec.update(status="trained", reject_reason="", **res,
                       fitness=fitness(res["val_acc"], hw, constraints, args.w_latency, args.w_size))
            n_trained += 1
            print(f"  [{n_trained:3d}/{args.budget}] #{rec['id']:3d} {origin:9s} val_acc {rec['val_acc']:5.2f} "
                  f"fit {rec['fitness']:.4f} lat {hw['latency_ms']:.2f}ms size {hw['size_mb']:.2f}MB "
                  f"mem {hw['peak_mem_mb']:.1f}MB ({res['train_time_s']:.0f}s) {describe(cfg)}", flush=True)
        history.append(rec)
        save_history(history, constraints, args)
        return rec

    def fresh(make) -> Dict:
        """Generate a not-yet-seen configuration with ``make()``."""
        for _ in range(200):
            cfg = make()
            if config_key(cfg) not in seen:
                return cfg
        return random_config(rng)

    t0 = time.time()
    # ---- phase 1: random initial population (infeasible samples are rejected, not trained)
    while len(population) < args.population and n_trained < args.budget:
        rec = evaluate_candidate(fresh(lambda: random_config(rng)), "random")
        if rec["status"] == "trained":
            population.append(rec)

    # ---- phase 2: aging evolution
    def tournament() -> Dict:
        return max(rng.sample(list(population), min(args.sample, len(population))),
                   key=lambda r: r["fitness"])

    while n_trained < args.budget:
        parent = tournament()
        if rng.random() < args.crossover_prob:
            other = tournament()
            child = fresh(lambda: mutate(crossover(parent["config"], other["config"], rng), rng, 0.15))
            origin = "crossover"
        else:
            child = fresh(lambda: mutate(parent["config"], rng))
            origin = "mutation"
        rec = evaluate_candidate(child, origin)
        if rec["status"] == "trained":
            population.append(rec)          # deque(maxlen) drops the oldest member

    best = write_best(history, constraints)
    print(f"\nsearch finished in {(time.time() - t0) / 60:.1f} min: {n_trained} trained, "
          f"{n_rejected} rejected by hardware constraints", flush=True)
    print(f"BEST: val_acc {best['val_acc']:.2f}% latency {best['latency_ms']:.2f}ms "
          f"size {best['size_mb']:.2f}MB mem {best['peak_mem_mb']:.1f}MB\n      "
          f"{describe(best['config'])}", flush=True)
    print(f"saved {BEST_JSON}", flush=True)


if __name__ == "__main__":
    main()
