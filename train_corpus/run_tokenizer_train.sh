#!/usr/bin/env bash
set -euo pipefail

# Train nanochat tokenizer on local base_data_climbmix parquet files.
# Use prepare_tokenizer_base_data.sh first if needed.
#
# Example:
#   train_corpus/run_tokenizer_train.sh

MAX_CHARS="${MAX_CHARS:-5000000000}"
DOC_CAP="${DOC_CAP:-10000}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
LOG_FILE="${LOG_FILE:-data/tok_train.log}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "Training tokenizer..."
echo "  max chars: $MAX_CHARS"
echo "  doc cap: $DOC_CAP"
echo "  vocab size: $VOCAB_SIZE"
echo "  log: $LOG_FILE"

uv run python -m scripts.tok_train \
  --max-chars "$MAX_CHARS" \
  --doc-cap "$DOC_CAP" \
  --vocab-size "$VOCAB_SIZE" \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
