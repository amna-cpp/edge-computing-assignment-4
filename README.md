# Edge Computing & Intelligent Systems — Assignment 4

Knowledge distillation (Q1) and hardware-aware neural architecture search (Q2), both ending in
ONNX models that run on the CPU with ONNX Runtime.

| Question | What it does | Code |
|---|---|---|
| **Q1** | ResNet-50 teacher on CIFAR-100 → distil into MobileNetV2, SqueezeNet, EfficientNet-Lite0 (hard + soft/temperature + feature loss); matching no-KD baselines; unified inference; ONNX export + ONNX Runtime CPU inference | `q1_distillation/` |
| **Q2** | Searchable CNN space + architecture generator + evolutionary NAS on CIFAR-10 that *measures* CPU latency / model size / peak memory and rejects violators; best model re-trained from scratch, exported to ONNX, ONNX Runtime inference | `q2_nas/` |
| Shared | on-device data pipeline, training engine, ONNX/profiling helpers | `edge_common/` |
| Benchmarks | automated benchmark of all models → CSV | `benchmarks/` |

## 1. Installation

Python 3.9+ (tested 3.9.6, macOS 15, Apple M4). Training uses the Apple GPU (`mps`) when present,
CUDA if available, otherwise the CPU (slow). All latency / memory measurements are CPU-only.

```bash
pip install -r requirements.txt
```

### Data
* **CIFAR-10** – place the standard python version in `data/cifar-10-batches-py/`
  (<https://www.cs.toronto.edu/~kriz/cifar.html>).
* **CIFAR-100** – the files used here are the parquet files of the Hugging Face dataset
  `uoft-cs/cifar100` (`train-00000-of-00001.parquet`, `test-00000-of-00001.parquet`), saved as
  `data/hf_cifar100/train.parquet` and `test.parquet`. Convert them once (needs `pyarrow`):
  ```bash
  python -m edge_common.prepare_cifar100      # -> data/cifar100.npz, data/cifar100_classes.json
  ```

(The datasets are not shipped in the submission archive.)

## 2. Run everything

```bash
bash run_pipeline.sh all
```
or stage by stage (every stage is resumable; finished checkpoints are skipped):

```bash
# ---------------- Question 1 ----------------
python -m q1_distillation.train_teacher --epochs 30                 # ResNet-50 teacher  -> checkpoints/teacher/
python -m q1_distillation.train_baseline --arch all --epochs 20     # 3 students, no KD  -> checkpoints/students/*_baseline.pt
python -m q1_distillation.train_kd --arch all --epochs 20           # 3 students, KD     -> checkpoints/students/*_kd.pt
python -m q1_distillation.train_kd --arch all --epochs 20 \
       --feature-weight 0 --variant kd_logit                        # ablation: logit-only KD
python -m q1_distillation.export_onnx --teacher                     # -> onnx_models/*.onnx (verified vs PyTorch)

# unified inference (all students, PyTorch or ONNX back-end)
python -m q1_distillation.inference --samples 3
python -m q1_distillation.inference --image some_picture.png --backend onnx
python -m q1_distillation.inference --evaluate --backend onnx
# stand-alone ONNX Runtime script (no PyTorch needed)
python -m q1_distillation.onnx_infer --all --evaluate

# ---------------- Question 2 ----------------
python -m q2_nas.search --budget 60          # evolutionary NAS -> results/q2/{nas_history.csv,json,best_architecture.json}
python -m q2_nas.train_best --epochs 60      # train best architecture from scratch -> checkpoints/nas/nas_best.pt
python -m q2_nas.export_onnx                 # -> onnx_models/nas_best.onnx
python -m q2_nas.inference --evaluate        # ONNX Runtime CPU inference (+ --image / --samples)

# ---------------- Benchmark ----------------
python -m benchmarks.benchmark_all           # -> results/benchmark_results.csv, results/kd_gains.csv
```

> **Run `q2_nas.search` and `benchmarks.benchmark_all` on an otherwise idle machine.** Latency is
> measured, not estimated; a training job running in the background inflated measured latencies by up to 2x.

## 3. Repository layout

```
edge_common/       data.py (GPU-resident CIFAR + augmentation) · engine.py (training loop) ·
                   deploy.py (ONNX export/verify, ORT profiling) · profile_worker.py
q1_distillation/   models/{resnet,mobilenetv2,squeezenet,efficientnet_lite}.py · distill.py (KD losses)
                   train_{teacher,baseline,kd}.py · inference.py · export_onnx.py · onnx_infer.py
q2_nas/            search_space.py · generator.py · hardware.py · evaluate.py · search.py
                   train_best.py · export_onnx.py · inference.py
benchmarks/        benchmark_all.py
checkpoints/       teacher/ · students/ · nas/          (trained weights)
onnx_models/       exported ONNX models
results/           histories, search logs, benchmark CSVs, figures
report/            technical report (PDF)
```

## 4. Deliverable map

| # | Deliverable | Location |
|---|---|---|
| 1 | Teacher model | `q1_distillation/train_teacher.py`, `checkpoints/teacher/resnet50_cifar100.pt` |
| 2 | Student architectures | `q1_distillation/models/{mobilenetv2,squeezenet,efficientnet_lite}.py` |
| 3 | Baseline students | `train_baseline.py`, `checkpoints/students/*_baseline.pt` |
| 4 | Knowledge distillation | `distill.py`, `train_kd.py`, `checkpoints/students/*_kd.pt` |
| 5 | NAS search space | `q2_nas/search_space.py`, `q2_nas/generator.py` |
| 6 | NAS algorithm | `q2_nas/search.py` (evolutionary / aging evolution) |
| 7 | NAS results | `results/q2/nas_history.{csv,json}`, `best_architecture.json`, `pareto_front.csv` |
| 8 | Optimised architecture | `checkpoints/nas/nas_best.pt` |
| 9 | Inference pipelines | `q1_distillation/inference.py`, `q2_nas/inference.py` |
| 10 | ONNX deployment | `onnx_models/`, `q1_distillation/onnx_infer.py`, `q2_nas/inference.py` |
| 11 | Benchmarking | `benchmarks/benchmark_all.py`, `results/benchmark_results.csv` |
| 12 | Technical report | `report/technical_report.pdf` |
| 13 | README | this file |

## 5. Key design decisions (details in the report)

* **Models are implemented from scratch** (no torchvision/timm), CIFAR-adapted (3x3 stem, no max-pool).
* **Teacher stem stride 2** – a stride-1 ResNet-50 is ~3x slower to train on the available hardware.
* **Distillation loss**: `alpha·CE + (1-alpha)·T²·KL(softmax(t/T) ‖ softmax(s/T)) + beta·feature-MSE`
  (T = 4, alpha = 0.3, beta = 1); feature hints use 1x1-conv adapters that exist only during training.
* **Baselines vs KD** use identical data, schedule and hyper-parameters; they differ only in the loss.
* **Final-epoch weights** are reported everywhere – no checkpoint selection on the test set.
* **NAS** never touches the test set (held-out validation split of the training set); hardware is
  measured with ONNX Runtime on the CPU (1 thread, batch 1) in a fresh process per candidate, and
  violators are rejected *before* training.
