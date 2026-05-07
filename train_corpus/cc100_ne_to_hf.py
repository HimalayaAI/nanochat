#!/usr/bin/env python3
"""
Download CC100 Nepali (.txt.xz), stream to Parquet shards, and optionally upload to HF.

Example:
  python train_corpus/cc100_ne_to_hf.py \
    --url http://data.statmt.org/cc-100/ne.txt.xz \
    --repo himalaya-ai/cc100-ne \
    --rows-per-shard 500000 \
    --out-dir data/cc100-ne-parquet

Output schema:
  id (int64), text (string), language (string), source (string)
"""

from __future__ import annotations

import argparse
import lzma
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token, login


def download_with_resume(url: str, dest: Path, chunk_size: int = 8 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        status = getattr(resp, "status", None) or resp.getcode()
        mode = "ab" if existing > 0 and status == 206 else "wb"
        if mode == "wb" and existing > 0:
            existing = 0
        with open(dest, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)


def iter_lines(
    path: Path, encoding: str, errors: str, max_lines: Optional[int]
) -> Iterable[tuple[int, str]]:
    with lzma.open(path, "rt", encoding=encoding, errors=errors) as f:
        for idx, line in enumerate(f, start=1):
            if max_lines and idx > max_lines:
                break
            text = line.strip()
            if text:
                yield idx, text


def find_start_index(out_dir: Path, prefix: str) -> int:
    if not out_dir.exists():
        return 0
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.parquet$")
    max_idx = -1
    for path in out_dir.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def write_shards(
    lines: Iterable[tuple[int, str]],
    out_dir: Path,
    shard_prefix: str,
    rows_per_shard: int,
    language: str,
    source: str,
    api: Optional[HfApi],
    repo_id: Optional[str],
    keep_local: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = find_start_index(out_dir, shard_prefix)
    buffer_text = []
    buffer_id = []

    def flush() -> None:
        nonlocal shard_idx, buffer_text, buffer_id
        if not buffer_text:
            return
        n = len(buffer_text)
        table = pa.Table.from_pydict(
            {
                "id": buffer_id,
                "text": buffer_text,
                "language": [language] * n,
                "source": [source] * n,
            }
        )
        shard_name = f"{shard_prefix}-{shard_idx:06d}.parquet"
        shard_path = out_dir / shard_name
        pq.write_table(table, shard_path, compression="zstd")
        if api and repo_id:
            api.upload_file(
                path_or_fileobj=str(shard_path),
                path_in_repo=f"data/{shard_name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            if not keep_local:
                shard_path.unlink(missing_ok=True)
        shard_idx += 1
        buffer_text = []
        buffer_id = []

    for row_id, text in lines:
        buffer_id.append(row_id)
        buffer_text.append(text)
        if len(buffer_text) >= rows_per_shard:
            flush()
    flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CC100 Nepali and upload Parquet shards to HF")
    parser.add_argument("--url", required=True, help="Source URL for ne.txt.xz")
    parser.add_argument("--repo", required=True, help="HF dataset repo (org/name)")
    parser.add_argument("--out-dir", default="data/cc100-ne-parquet", help="Local parquet shard directory")
    parser.add_argument("--download-path", default="data/cc100-ne.txt.xz", help="Local .xz path")
    parser.add_argument("--rows-per-shard", type=int, default=500_000, help="Rows per parquet shard")
    parser.add_argument("--shard-prefix", default="cc100-ne", help="Shard filename prefix")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding")
    parser.add_argument("--encoding-errors", default="ignore", help="Encoding error handling")
    parser.add_argument("--language", default="ne", help="Language label to store in column")
    parser.add_argument("--source", default="cc100", help="Source label to store in column")
    parser.add_argument("--max-lines", type=int, default=0, help="Optional line cap for testing")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if file already exists")
    parser.add_argument("--skip-upload", action="store_true", help="Only write local shards")
    parser.add_argument("--keep-local", action="store_true", help="Keep local shards after upload")
    args = parser.parse_args()

    download_path = Path(args.download_path)
    if not args.skip_download or not download_path.exists():
        print(f"Downloading {args.url} -> {download_path}")
        download_with_resume(args.url, download_path)

    api = None
    if not args.skip_upload:
        token = get_token() or os.getenv("HF_TOKEN")
        if not token:
            login()
            token = get_token()
        api = HfApi()
        api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True, token=token)

    lines = iter_lines(
        download_path,
        encoding=args.encoding,
        errors=args.encoding_errors,
        max_lines=args.max_lines or None,
    )
    write_shards(
        lines=lines,
        out_dir=Path(args.out_dir),
        shard_prefix=args.shard_prefix,
        rows_per_shard=args.rows_per_shard,
        language=args.language,
        source=args.source,
        api=api,
        repo_id=args.repo if not args.skip_upload else None,
        keep_local=args.keep_local,
    )

    print("Done.")


if __name__ == "__main__":
    main()
