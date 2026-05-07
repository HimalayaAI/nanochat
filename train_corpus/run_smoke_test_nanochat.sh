#!/usr/bin/env bash
set -euo pipefail

# Wrapper for checkpoint-level nanochat smoke tests.
#
# Example:
#   train_corpus/run_smoke_test_nanochat.sh \
#     --source base --model-tag d15_harl_fulltokens_sdpa_bs32

LOG_FILE="${LOG_FILE:-data/nanochat_smoke_test.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.benchmarks.smoke_test_nanochat \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"

