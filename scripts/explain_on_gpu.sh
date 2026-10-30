#!/usr/bin/env bash
# Phase 7 step R: score the explanations of every trained seed in a run, then summarise.
#
#   GPU=1 bash scripts/explain_on_gpu.sh phase7_A
#   FORCE=1 GPU=1 bash scripts/explain_on_gpu.sh phase7_A     # re-score seeds that are already done
#
# Needs the .pth checkpoints, so run it on the GPU server where the run was trained.
# Safe to re-run: seeds that already have explain/summary.json are skipped.
set -euo pipefail
RUN_NAME="${1:?usage: GPU=<id> bash scripts/explain_on_gpu.sh <run-name>}"
GPU="${GPU:-0}"
ALLOW_CPU="${ALLOW_CPU:-0}"; CPU_FLAG=""; [ "${ALLOW_CPU}" = "1" ] && CPU_FLAG="--allow-cpu"
cd "$(dirname "$0")/.."
RUN_DIR="results/${RUN_NAME}"
mkdir -p logs
LOG="logs/${RUN_NAME}_explain.log"
export CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1
exec > >(tee -a "${LOG}") 2>&1
echo "=== explain ${RUN_NAME} | $(date) | GPU ${GPU} | commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"

SECONDS=0
found=0
for SEED_DIR in "${RUN_DIR}"/*/seed_*; do
  [ -e "${SEED_DIR}/best_hybrid_net.pth" ] || { echo "--- ${SEED_DIR}: no checkpoint here, skipping"; continue; }
  found=$((found + 1))
  if [ -e "${SEED_DIR}/explain/summary.json" ] && [ "${FORCE:-0}" != "1" ]; then echo "--- ${SEED_DIR}: already done"; continue; fi
  echo "--- ${SEED_DIR}: explain"
  python pipelines/05_explain_eval.py --results-dir "${SEED_DIR}" ${CPU_FLAG} > "${SEED_DIR}/explain_stdout.txt"
  tail -n 25 "${SEED_DIR}/explain_stdout.txt"
done
[ "${found}" -gt 0 ] || { echo "!! no checkpoints under ${RUN_DIR}/*/seed_*"; exit 1; }
echo "explaining took $((SECONDS / 60)) min $((SECONDS % 60)) s"
echo "--- summary over seeds"
python scripts/aggregate_explain.py "${RUN_DIR}"
echo "=== done: ${RUN_DIR}/EXPLAIN_SUMMARY.md, log ${LOG}"
