#!/usr/bin/env bash
set -euo pipefail

# Wrapper for generation smoke tests on HF-exported nanochat models.
#
# Examples:
#   train_corpus/run_smoke_test_hf_model.sh --repo-id himalaya-ai/himalayagpt-0.5b
#   train_corpus/run_smoke_test_hf_model.sh --local-dir /data/.../hf_export/himalayagpt-0.5b

LOG_FILE="${LOG_FILE:-data/hf_model_smoke_test.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.smoke_test_hf_model \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
