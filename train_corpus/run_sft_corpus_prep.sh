#!/usr/bin/env bash
set -euo pipefail

# Build mixed SFT corpus from multiple HF datasets.
#
# Example (local only):
#   train_corpus/run_sft_corpus_prep.sh
#
# Example (upload):
#   train_corpus/run_sft_corpus_prep.sh --upload-repo himalaya-ai/nepali-sft-mix-v1

LOG_FILE="${LOG_FILE:-data/sft_corpus_prep.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.prepare_sft_mixture \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"

