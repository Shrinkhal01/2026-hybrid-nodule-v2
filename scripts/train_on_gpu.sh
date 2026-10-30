#!/usr/bin/env bash
# Train + evaluate all three models on ONE GPU of the college server, for several feature
# sets and seeds, then summarise. Safe to re-run: finished seed folders are skipped.
#
#   GPU=1 bash scripts/train_on_gpu.sh phase7_A                       # all + image, seeds 42-46
#   GPU=1 FEATURES="all" SEEDS="42" bash scripts/train_on_gpu.sh quick  # one run, like Run 1
#
# Layout: results/<run>/<feature set>/seed_<n>/   (+ results/<run>/SUMMARY.md at the end)
# Steps:  check GPU + torch -> verify the processed data -> 1-epoch smoke test ->
#         for each feature set and seed: train (02) -> Grad-CAM (03) -> figures (04) ->
#         aggregate over seeds (scripts/aggregate_runs.py).
# Everything is printed AND saved to logs/<run>.log. Run it inside tmux.
set -euo pipefail

RUN_NAME="${1:?usage: GPU=<id> [FEATURES=\"all image\"] [SEEDS=\"42 43 44 45 46\"] bash scripts/train_on_gpu.sh <run-name>}"
GPU="${GPU:-0}"                        # which of the 4 H200s to use (check nvidia-smi first)
FEATURES="${FEATURES:-all image}"      # all = 15 features, image = 12 (no radiologist ratings)
SEEDS="${SEEDS:-42 43 44 45 46}"       # several seeds -> mean +- SD instead of one lucky number
EPOCHS="${EPOCHS:-40}"
ALLOW_CPU="${ALLOW_CPU:-0}"            # 1 = debug run without a GPU (Mac / laptop), very slow
CPU_FLAG=""; [ "${ALLOW_CPU}" = "1" ] && CPU_FLAG="--allow-cpu"
cd "$(dirname "$0")/.."                # repo root, wherever you start from
RUN_DIR="results/${RUN_NAME}"
mkdir -p "${RUN_DIR}" logs
LOG="logs/${RUN_NAME}.log"
export CUDA_VISIBLE_DEVICES="${GPU}"   # the scripts then see only this GPU, as cuda:0
export PYTHONUNBUFFERED=1              # print lines immediately (live log)

exec > >(tee -a "${LOG}") 2>&1         # copy all output into the log file
echo "=== ${RUN_NAME} | $(date) | host $(hostname) | GPU ${GPU} | commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
echo "=== feature sets: ${FEATURES} | seeds: ${SEEDS} | epochs: ${EPOCHS}"

export ALLOW_CPU
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
python - <<'PY'
import os, torch, pandas, sklearn
if not torch.cuda.is_available() and os.environ.get("ALLOW_CPU") != "1":
    raise SystemExit("torch cannot see a GPU: activate the right environment / check CUDA_VISIBLE_DEVICES")
print("torch", torch.__version__, "| pandas", pandas.__version__, "| sklearn", sklearn.__version__,
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU debug run)")
PY

echo "--- verify processed data (patch shape, splits, patient check)"
python 01_data_pipeline.py --verify-only

echo "--- smoke test: 1 epoch on 64 nodules per feature set (results thrown away)"
for FS in ${FEATURES}; do
  python pipelines/02_train.py --features "${FS}" --epochs 1 --limit 64 ${CPU_FLAG} --results-dir "/tmp/smoke_${RUN_NAME}_${FS}_$$"
done

SECONDS=0
for FS in ${FEATURES}; do
  for SEED in ${SEEDS}; do
    OUT="${RUN_DIR}/${FS}/seed_${SEED}"
    if [ -e "${OUT}/figures/significance_tests.csv" ]; then
      echo "--- ${FS} / seed ${SEED}: already done, skipping"; continue
    fi
    echo "--- ${FS} / seed ${SEED}: train"
    python pipelines/02_train.py --features "${FS}" --seed "${SEED}" --epochs "${EPOCHS}" ${CPU_FLAG} --results-dir "${OUT}"
    echo "--- ${FS} / seed ${SEED}: Grad-CAM"
    python pipelines/03_evaluate_xai.py --results-dir "${OUT}"
    echo "--- ${FS} / seed ${SEED}: figures + significance tests"
    python pipelines/04_report_figures.py --results-dir "${OUT}" > "${OUT}/report_stdout.txt"
  done
done
echo "all runs took $((SECONDS / 60)) min $((SECONDS % 60)) s"

echo "--- summary over seeds and feature sets"
python scripts/aggregate_runs.py "${RUN_DIR}"
echo "=== done: results in ${RUN_DIR}/ (SUMMARY.md), log in ${LOG}"
