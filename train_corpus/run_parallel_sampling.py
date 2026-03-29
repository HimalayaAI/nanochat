#!/usr/bin/env python3
"""
Launch parallel tokenizer sampling jobs based on group regexes.

Each group spawns sample_tokenizer_corpus.py with include-regex + shard-prefix,
and optional upload config. This avoids manual multiple commands.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_groups(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".json"):
        return json.loads(Path(path).read_text())
    return yaml.safe_load(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parallel tokenizer sampling groups")
    parser.add_argument("--config", required=True, help="Tokenizer sources config")
    parser.add_argument("--groups", required=True, help="Group definitions YAML/JSON")
    parser.add_argument("--target-tokens", type=int, help="Override total target tokens")
    parser.add_argument(
        "--scale-targets",
        action="store_true",
        help="Scale per-source target_tokens to match --target-tokens",
    )
    parser.add_argument("--upload-repo", help="HF repo to upload shards")
    parser.add_argument("--shard-rows", type=int, default=100000)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        default="data",
        help="Directory to store per-group checkpoint files",
    )
    parser.add_argument(
        "--stats-dir",
        default="data",
        help="Directory to store per-group stats files",
    )
    args = parser.parse_args()

    groups = load_groups(args.groups)
    if not isinstance(groups, list):
        raise SystemExit("Groups file must be a list")

    total_group_weight = sum(float(g.get("weight", 0.0)) for g in groups)
    if args.target_tokens and total_group_weight > 0:
        for group in groups:
            if group.get("target_tokens") is not None:
                continue
            weight = float(group.get("weight", 0.0))
            if weight <= 0:
                continue
            group["target_tokens"] = int(args.target_tokens * (weight / total_group_weight))

    procs: List[subprocess.Popen] = []

    def launch(group: Dict[str, Any]) -> subprocess.Popen:
        include = group.get("include_regex")
        prefix = group["shard_prefix"]
        log_path = group.get("log", f"data/tokenizer_{prefix}.log")
        sources = group.get("sources")
        group_target = group.get("target_tokens")
        equal_split = bool(group.get("equal_split", False))
        checkpoint_path = str(
            Path(args.checkpoint_dir) / f"tokenizer_checkpoint_{prefix}.json"
        )
        stats_path = str(Path(args.stats_dir) / f"tokenizer_stats_{prefix}.json")
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("sample_tokenizer_corpus.py")),
            "--config",
            args.config,
            "--shard-prefix",
            prefix,
            "--shard-rows",
            str(args.shard_rows),
            "--checkpoint",
            checkpoint_path,
            "--stats",
            stats_path,
        ]
        if sources:
            cmd += ["--include-sources", ",".join(sources)]
        elif include:
            cmd += ["--include-regex", include]
        if group_target is not None:
            cmd += ["--target-tokens", str(group_target)]
        elif args.target_tokens:
            cmd += ["--target-tokens", str(args.target_tokens)]
        if args.scale_targets:
            cmd += ["--scale-targets"]
        if equal_split:
            cmd += ["--equal-split"]
        if args.upload_repo:
            cmd += ["--upload-repo", args.upload_repo]
        if args.skip_local:
            cmd += ["--skip-local"]
        with open(log_path, "w", encoding="utf-8") as log_f:
            return subprocess.Popen(cmd, stdout=log_f, stderr=log_f)

    queue = list(groups)
    while queue or procs:
        while queue and len(procs) < args.max_parallel:
            group = queue.pop(0)
            procs.append(launch(group))
        # wait for any to finish
        for proc in list(procs):
            ret = proc.poll()
            if ret is not None:
                procs.remove(proc)

    print("All sampling groups completed.")


if __name__ == "__main__":
    main()
