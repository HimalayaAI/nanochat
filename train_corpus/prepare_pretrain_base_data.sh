#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-himalaya-ai/gpt2-pretrain-corpus}"

"$SCRIPT_DIR/prepare_base_data_from_hf.sh" --repo "$REPO" "$@"
