#!/usr/bin/env bash
# Train + evaluate all three models on ONE GPU of the college server, end to end.
#
#   GPU=0 bash scripts/train_on_gpu.sh phase7_S        # run name -> results/phase7_S/
#
# Steps: check GPU + torch -> verify the processed data -> 1-epoch smoke test ->
#        full training (02) -> Grad-CAM (03) -> figures + significance tests (04).
# Everything is printed AND saved to logs/<run name>.log.
# Run it inside tmux so it keeps going if your SSH connection drops.
set -euo pipefail

RUN_NAME="${1:?usage: GPU=<id> bash scripts/train_on_gpu.sh <run-name>}"
GPU="${GPU:-0}"                      # which of the 4 H200s to use (check nvidia-smi first)
cd "$(dirname "$0")/.."              # repo root, wherever you start from
OUT="results/${RUN_NAME}"
mkdir -p "${OUT}" logs
LOG="logs/${RUN_NAME}.log"
export CUDA_VISIBLE_DEVICES="${GPU}" # the scripts then see only this GPU, as cuda:0
export PYTHONUNBUFFERED=1            # print lines immediately (live log)

exec > >(tee -a "${LOG}") 2>&1       # copy all output into the log file
echo "=== ${RUN_NAME} | $(date) | host $(hostname) | GPU ${GPU} | commit $(git rev-parse --short HEAD)"

if [ -e "${OUT}/model_comparison_metrics.csv" ]; then
  echo "!! ${OUT} already has results. Pick a new run name so nothing is overwritten."; exit 1
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU: activate the right environment / check CUDA_VISIBLE_DEVICES"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
PY

echo "--- 1/5 verify processed data (patch shape, splits, patient check)"
python 01_data_pipeline.py --verify-only

echo "--- 2/5 smoke test: 1 epoch on 64 nodules (catches errors in ~1 min, results thrown away)"
python pipelines/02_train.py --epochs 1 --limit 64 --results-dir "/tmp/smoke_${RUN_NAME}_$$"

echo "--- 3/5 full training: RF + 3D ResNet + Hybrid"
SECONDS=0
python pipelines/02_train.py --results-dir "${OUT}"
echo "training took $((SECONDS / 60)) min $((SECONDS % 60)) s"

echo "--- 4/5 Grad-CAM cases"
python pipelines/03_evaluate_xai.py --results-dir "${OUT}"

echo "--- 5/5 report figures + significance tests"
python pipelines/04_report_figures.py --results-dir "${OUT}"

echo "=== done: results in ${OUT}/ , log in ${LOG}"
cat "${OUT}/model_comparison_metrics.csv"
