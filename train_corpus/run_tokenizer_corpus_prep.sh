#!/usr/bin/env bash
set -euo pipefail

# Build tokenizer corpus shards from configured sources and upload to HF.
#
# Example:
#   train_corpus/run_tokenizer_corpus_prep.sh

CONFIG="${CONFIG:-train_corpus/configs/tokenizer_sources.yml}"
TARGET_TOKENS="${TARGET_TOKENS:-5000000000}"
UPLOAD_REPO="${UPLOAD_REPO:-himalaya-ai/gpt2-tokenizer-corpus}"
LOG_FILE="${LOG_FILE:-data/tokenizer_sampler.out}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "Running tokenizer corpus sampler..."
echo "  config: $CONFIG"
echo "  target tokens: $TARGET_TOKENS"
echo "  upload repo: $UPLOAD_REPO"
echo "  log: $LOG_FILE"

uv run python train_corpus/sample_tokenizer_corpus.py \
  --config "$CONFIG" \
  --target-tokens "$TARGET_TOKENS" \
  --upload-repo "$UPLOAD_REPO" \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
