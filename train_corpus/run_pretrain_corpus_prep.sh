#!/usr/bin/env bash
set -euo pipefail

# Build pretraining corpus shards (HARL-first mix by default) and upload to HF.
#
# Example:
#   train_corpus/run_pretrain_corpus_prep.sh

TARGET_TOKENS="${TARGET_TOKENS:-10000000000}"
UPLOAD_REPO="${UPLOAD_REPO:-himalaya-ai/gpt2-pretrain-corpus}"
SOURCES_CONFIG="${SOURCES_CONFIG:-train_corpus/configs/pretrain_sources.yml}"
GROUPS_CONFIG="${GROUPS_CONFIG:-train_corpus/configs/pretrain_parallel_groups.yml}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SHARD_ROWS="${SHARD_ROWS:-100000}"
LOG_FILE="${LOG_FILE:-data/pretrain_prepare.log}"

mkdir -p "$(dirname "$LOG_FILE")"

echo "Running pretraining corpus sampler..."
echo "  sources: $SOURCES_CONFIG"
echo "  groups: $GROUPS_CONFIG"
echo "  target tokens: $TARGET_TOKENS"
echo "  upload repo: $UPLOAD_REPO"
echo "  max parallel: $MAX_PARALLEL"
echo "  shard rows: $SHARD_ROWS"
echo "  log: $LOG_FILE"

uv run python train_corpus/run_pretrain_corpus.py \
  --sources-config "$SOURCES_CONFIG" \
  --groups-config "$GROUPS_CONFIG" \
  --target-tokens "$TARGET_TOKENS" \
  --upload-repo "$UPLOAD_REPO" \
  --max-parallel "$MAX_PARALLEL" \
  --shard-rows "$SHARD_ROWS" \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
