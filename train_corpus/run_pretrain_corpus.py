#!/usr/bin/env python3
"""
Prepare a pretraining corpus using the same mixture strategy as tokenizer sampling.

This is a thin wrapper around run_parallel_sampling.py with pretraining-friendly
defaults so launches are repeatable and easy.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TARGET_TOKENS = 10_000_000_000  # Karpathy-like GPT-2 speedrun-scale target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pretraining corpus sampling with tokenizer-like group mix"
    )
    parser.add_argument(
        "--sources-config",
        default="train_corpus/configs/tokenizer_sources.yml",
        help="Source definitions YAML (defaults to tokenizer_sources.yml)",
    )
    parser.add_argument(
        "--groups-config",
        default="train_corpus/configs/pretrain_parallel_groups.yml",
        help="Group mix YAML (defaults to pretrain_parallel_groups.yml)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help=(
            "Total target tokens across all groups. "
            f"Default: {DEFAULT_TARGET_TOKENS:,}"
        ),
    )
    parser.add_argument(
        "--upload-repo",
        required=True,
        help="HF dataset repo to upload prepared pretraining shards",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="How many groups to sample in parallel",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=100_000,
        help="Rows per uploaded parquet shard",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/pretrain_checkpoints",
        help="Directory for per-group checkpoint JSON files",
    )
    parser.add_argument(
        "--stats-dir",
        default="data/pretrain_stats",
        help="Directory for per-group stats JSON files",
    )
    parser.add_argument(
        "--scale-targets",
        action="store_true",
        help="Scale source-level explicit targets (if present) to match target tokens",
    )
    parser.add_argument(
        "--keep-local-jsonl",
        action="store_true",
        help="Keep local JSONL output from sampler (default is upload-only mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command without launching sampling jobs",
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)

    wrapper_path = Path(__file__).resolve().with_name("run_parallel_sampling.py")
    cmd = [
        sys.executable,
        str(wrapper_path),
        "--config",
        args.sources_config,
        "--groups",
        args.groups_config,
        "--target-tokens",
        str(args.target_tokens),
        "--upload-repo",
        args.upload_repo,
        "--max-parallel",
        str(args.max_parallel),
        "--shard-rows",
        str(args.shard_rows),
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--stats-dir",
        args.stats_dir,
    ]
    if args.scale_targets:
        cmd.append("--scale-targets")
    if not args.keep_local_jsonl:
        cmd.append("--skip-local")

    print("Launching pretraining corpus preparation:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
