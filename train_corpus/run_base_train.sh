#!/usr/bin/env bash
set -euo pipefail

# Launch nanochat base pretraining.
# Use prepare_pretrain_base_data.sh first if needed.
#
# Example:
#   train_corpus/run_base_train.sh

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
RUN_NAME="${RUN_NAME:-d24_harl_r12}"
MODEL_TAG="${MODEL_TAG:-$RUN_NAME}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
FP8="${FP8:-0}"
LOG_FILE="${LOG_FILE:-data/base_train.log}"

mkdir -p "$(dirname "$LOG_FILE")"

CMD=(
  uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE"
  -m scripts.base_train --
  --depth="$DEPTH"
  --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO"
  --device-batch-size="$DEVICE_BATCH_SIZE"
  --run="$RUN_NAME"
  --model-tag="$MODEL_TAG"
  --device-type="$DEVICE_TYPE"
)

if [[ "$FP8" == "1" ]]; then
  CMD+=(--fp8)
fi

echo "Launching base pretraining..."
echo "  nproc: $NPROC_PER_NODE"
echo "  depth: $DEPTH"
echo "  ratio: $TARGET_PARAM_DATA_RATIO"
echo "  device batch size: $DEVICE_BATCH_SIZE"
echo "  run: $RUN_NAME"
echo "  model tag: $MODEL_TAG"
echo "  fp8: $FP8"
echo "  log: $LOG_FILE"

"${CMD[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
