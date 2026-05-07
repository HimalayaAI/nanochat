#!/usr/bin/env bash
set -euo pipefail

# Download parquet shards from an HF dataset repo and make them visible to nanochat
# under $NANOCHAT_BASE_DIR/base_data_climbmix (as a symlink).
#
# Example:
#   train_corpus/prepare_base_data_from_hf.sh \
#     --repo himalaya-ai/gpt2-pretrain-corpus

usage() {
  cat <<'EOF'
Usage: prepare_base_data_from_hf.sh --repo <hf_dataset_repo> [options]

Options:
  --repo <repo>           HF dataset repo id (required), e.g. himalaya-ai/gpt2-pretrain-corpus
  --include <pattern>     Include glob for hf download (default: data/*.parquet)
  --base-dir <path>       NANOCHAT_BASE_DIR (default: $NANOCHAT_BASE_DIR or $HOME/.cache/nanochat)
  --cache-name <name>     Download cache folder name (default: repo name with / replaced by -)
  --keep-old              Keep old base_data_climbmix if present (default: replace with symlink)
  -h, --help              Show this help
EOF
}

REPO=""
INCLUDE_PATTERN="data/*.parquet"
BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
CACHE_NAME=""
KEEP_OLD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --include) INCLUDE_PATTERN="$2"; shift 2 ;;
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --cache-name) CACHE_NAME="$2"; shift 2 ;;
    --keep-old) KEEP_OLD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$REPO" ]]; then
  echo "Error: --repo is required" >&2
  usage
  exit 1
fi

if [[ -z "$CACHE_NAME" ]]; then
  CACHE_NAME="${REPO//\//-}"
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' CLI not found. Install huggingface_hub CLI first." >&2
  exit 1
fi

mkdir -p "$BASE_DIR/downloads"

DOWNLOAD_DIR="$BASE_DIR/downloads/$CACHE_NAME"
TARGET_LINK="$BASE_DIR/base_data_climbmix"

echo "Downloading dataset shards from: $REPO"
echo "Local download dir: $DOWNLOAD_DIR"
hf download "$REPO" \
  --repo-type dataset \
  --include "$INCLUDE_PATTERN" \
  --local-dir "$DOWNLOAD_DIR"

if [[ -d "$DOWNLOAD_DIR/data" ]]; then
  SOURCE_DIR="$DOWNLOAD_DIR/data"
else
  SOURCE_DIR="$DOWNLOAD_DIR"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Error: expected source data directory not found: $SOURCE_DIR" >&2
  exit 1
fi

if [[ $KEEP_OLD -eq 0 ]]; then
  rm -rf "$TARGET_LINK"
  ln -s "$SOURCE_DIR" "$TARGET_LINK"
  echo "Linked $TARGET_LINK -> $SOURCE_DIR"
else
  echo "--keep-old set, not replacing $TARGET_LINK"
fi

PARQUET_COUNT=$(find "$SOURCE_DIR" -type f -name '*.parquet' | wc -l | tr -d ' ')
echo "Ready. Found $PARQUET_COUNT parquet files in $SOURCE_DIR"
