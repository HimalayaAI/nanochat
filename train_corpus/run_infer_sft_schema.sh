#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${LOG_FILE:-data/sft_schema_infer.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.infer_sft_source_schema \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
