#!/usr/bin/env bash
set -euo pipefail

# Wrapper for HF export validation checks.
#
# Examples:
#   train_corpus/run_validate_hf_model_export.sh --repo-id himalaya-ai/himalayagpt-0.5b
#   train_corpus/run_validate_hf_model_export.sh --local-dir /data/.../hf_export/himalayagpt-0.5b

LOG_FILE="${LOG_FILE:-data/hf_model_validate.log}"
mkdir -p "$(dirname "$LOG_FILE")"

uv run python -m train_corpus.validate_hf_model_export \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
