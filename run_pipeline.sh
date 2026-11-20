#!/usr/bin/env bash
# Reproduces the whole assignment, stage by stage.  Every stage is idempotent / resumable:
# finished checkpoints are skipped and interrupted runs continue from their last epoch.
#
#   bash run_pipeline.sh all                       # everything, in the order below
#   bash run_pipeline.sh nas_search kd baselines   # only some stages
#
# Stages
#   data        convert the CIFAR-100 parquet files to data/cifar100.npz   (one-off)
#   teacher     Q1  train the ResNet-50 teacher on CIFAR-100
#   nas_search  Q2  hardware-aware evolutionary NAS on CIFAR-10   (run on an otherwise idle
#                    machine: CPU latency is *measured*, background load would distort it)
#   kd          Q1  distil the three students (hard + soft + feature loss)
#   baselines   Q1  train the three students without distillation
#   nas_final   Q2  train the best architecture from scratch, export it to ONNX
#   export      Q1  export all students (+ teacher) to ONNX and verify them
#   kd_ablation Q1  logit-only distillation (no feature term) for the ablation study
#   bench       benchmark every model with ONNX Runtime on the CPU -> results/benchmark_results.csv
#
# Epoch counts can be changed through environment variables, e.g. STUDENT_EPOCHS=10 bash run_pipeline.sh kd
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-30}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-20}"
NAS_BUDGET="${NAS_BUDGET:-60}"
NAS_FINAL_EPOCHS="${NAS_FINAL_EPOCHS:-60}"
NAS_EPOCHS="${NAS_EPOCHS:-6}"
NAS_TRAIN_LIMIT="${NAS_TRAIN_LIMIT:-25000}"

run() { echo; echo ">>> $*"; "$@"; }

stage() {
  case "$1" in
    data)        run $PY -m edge_common.prepare_cifar100 ;;
    teacher)     run $PY -u -m q1_distillation.train_teacher --epochs "$TEACHER_EPOCHS" ;;
    nas_search)  run $PY -u -m q2_nas.search --budget "$NAS_BUDGET" --epochs "$NAS_EPOCHS" \
                         --train-limit "$NAS_TRAIN_LIMIT" ;;
    kd)          run $PY -u -m q1_distillation.train_kd --arch all --epochs "$STUDENT_EPOCHS" ;;
    baselines)   run $PY -u -m q1_distillation.train_baseline --arch all --epochs "$STUDENT_EPOCHS" ;;
    nas_final)   run $PY -u -m q2_nas.train_best --epochs "$NAS_FINAL_EPOCHS"
                 run $PY -m q2_nas.export_onnx ;;
    export)      run $PY -m q1_distillation.export_onnx --teacher ;;
    kd_ablation) run $PY -u -m q1_distillation.train_kd --arch all --epochs "$STUDENT_EPOCHS" \
                         --feature-weight 0 --variant kd_logit ;;
    bench)       run $PY -m benchmarks.benchmark_all ;;
    all)         for s in data teacher nas_search kd baselines nas_final export bench; do stage "$s"; done ;;
    *)           echo "unknown stage: $1" >&2; exit 1 ;;
  esac
}

[ $# -ge 1 ] || { sed -n '2,22p' "$0"; exit 1; }
for s in "$@"; do stage "$s"; done
echo; echo "pipeline finished: $*"
