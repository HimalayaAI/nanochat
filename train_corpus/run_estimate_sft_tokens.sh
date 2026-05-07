#!/usr/bin/env bash
set -euo pipefail

# Estimate token budget from compiled SFT customjson JSONL.
#
# Example:
#   train_corpus/run_estimate_sft_tokens.sh \
#     --input data/sft_harl_full/customjson_harl_full.jsonl

uv run python -m train_corpus.estimate_sft_customjson_tokens "$@"
