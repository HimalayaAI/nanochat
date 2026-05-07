#!/usr/bin/env bash
set -euo pipefail

# Export nanochat checkpoint to HF-compatible custom-code model package,
# optionally upload to an HF model repo.
#
# Example:
#   train_corpus/run_export_hf_model.sh \
#     --source base \
#     --model-tag d24_harl_r12 \
#     --upload-repo himalaya-ai/nanochat-d24-harl

OUTPUT_DIR="${OUTPUT_DIR:-data/hf_model_export}"
LOG_FILE="${LOG_FILE:-data/hf_model_export.log}"

mkdir -p "$(dirname "$LOG_FILE")"

uv run python train_corpus/export_nanochat_to_hf.py \
  --output-dir "$OUTPUT_DIR" \
  "$@" \
  2>&1 | tee -a "$LOG_FILE"
