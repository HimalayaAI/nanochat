#!/usr/bin/env bash
set -euo pipefail

# Build Devanagari-focused SFT corpus (excludes himalaya-ai/nepali-sft-dataset).
#
# Example (local only):
#   train_corpus/run_sft_deva_corpus_prep.sh
#
# Example (upload):
#   train_corpus/run_sft_deva_corpus_prep.sh --upload-repo himalaya-ai/deva-sft-compile-v1

LOG_FILE="${LOG_FILE:-data/sft_deva_corpus_prep.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.prepare_sft_mixture \
  --config train_corpus/configs/sft_sources_deva.yml \
  --sharegpt-out data/sft_deva_mix/sharegpt_deva_mixture.jsonl \
  --customjson-out data/sft_deva_mix/customjson_deva_mixture.jsonl \
  --stats-out data/sft_deva_mix/stats_deva_mixture.json \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
