#!/usr/bin/env bash
set -euo pipefail

# Launch nanochat SFT stage.
#
# Notes:
# - chat_sft.py expects identity conversations at:
#   $NANOCHAT_BASE_DIR/identity_conversations.jsonl
# - This script auto-downloads the default identity file if missing.
#
# Example:
#   train_corpus/run_sft_train.sh

NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOCHAT_BASE_DIR

IDENTITY_URL="${IDENTITY_URL:-https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl}"
IDENTITY_PATH="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RUN_NAME="${RUN_NAME:-sft_harl}"
MODEL_TAG="${MODEL_TAG:-d24_harl_r12}"
MODEL_STEP="${MODEL_STEP:-}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-1}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
LOG_FILE="${LOG_FILE:-data/sft_train.log}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$NANOCHAT_BASE_DIR"

if [[ ! -f "$IDENTITY_PATH" ]]; then
  echo "Downloading default identity conversations to $IDENTITY_PATH"
  curl -L -o "$IDENTITY_PATH" "$IDENTITY_URL"
fi

CMD=(
  uv run torchrun --standalone --nproc_per_node="$NPROC_PER_NODE"
  -m scripts.chat_sft --
  --run="$RUN_NAME"
  --model-tag="$MODEL_TAG"
  --device-batch-size="$DEVICE_BATCH_SIZE"
  --load-optimizer="$LOAD_OPTIMIZER"
  --device-type="$DEVICE_TYPE"
)

if [[ -n "$MODEL_STEP" ]]; then
  CMD+=(--model-step="$MODEL_STEP")
fi

echo "Launching SFT..."
echo "  nproc: $NPROC_PER_NODE"
echo "  run: $RUN_NAME"
echo "  model tag: $MODEL_TAG"
echo "  model step: ${MODEL_STEP:-latest}"
echo "  device batch size: $DEVICE_BATCH_SIZE"
echo "  load optimizer: $LOAD_OPTIMIZER"
echo "  log: $LOG_FILE"

"${CMD[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
