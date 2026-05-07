#!/usr/bin/env bash
set -euo pipefail

# Run Nepali Indic evaluation (Aya Indic Eval npi split) against either:
# - nanochat checkpoints, or
# - HF-exported models.
#
# Examples:
#   train_corpus/run_indic_eval_ne.sh \
#     --backend nanochat --source base --model-tag d15_harl_fulltokens_sdpa_bs32
#
#   train_corpus/run_indic_eval_ne.sh \
#     --backend hf --hf-model himalaya-ai/himalayagpt-0.5b --trust-remote-code

LOG_FILE="${LOG_FILE:-data/indic_eval_ne.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.benchmarks.indic_eval_ne \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"

