#!/usr/bin/env bash
set -euo pipefail

# Build a full HimalayaAI SFT pool (Nepali + Devanagari + selected global SFT).
#
# Example:
#   train_corpus/run_sft_harl_full_corpus_prep.sh
#
# Example with upload:
#   train_corpus/run_sft_harl_full_corpus_prep.sh \
#     --upload-repo himalaya-ai/sft-harl-full-v1

LOG_FILE="${LOG_FILE:-data/sft_harl_full_corpus_prep.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.prepare_sft_mixture \
  --config train_corpus/configs/sft_sources_harl_full.yml \
  --sharegpt-out data/sft_harl_full/sharegpt_harl_full.jsonl \
  --customjson-out data/sft_harl_full/customjson_harl_full.jsonl \
  --stats-out data/sft_harl_full/stats_harl_full.json \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
