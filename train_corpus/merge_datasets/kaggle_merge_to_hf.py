#!/usr/bin/env python3
"""
Merge Kaggle Nepali pretraining datasets into a HF repo.

This script:
  - reads a Kaggle inventory JSONL (from kaggle_inventory.py)
  - downloads selected datasets via Kaggle API
  - extracts text from files (txt/jsonl/json/csv/parquet)
  - applies quality filters + Nepali detection
  - text-hash dedupes and uploads parquet shards to HF
"""

from __future__ import annotations

import argparse
import csv
import sys
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("Kaggle API not available. Install with: pip install kaggle") from exc

import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token

# Ensure project root is on sys.path for scripts.* imports
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_corpus.merge_datasets.quality_filters import FilterSpec, normalize_text, passes_quality
from train_corpus.merge_datasets.merge_corpus_to_hf import (
    DedupeStore,
    get_max_shard_index,
    hash_text,
    prefill_dedupe_from_hf,
    upload_parquet_batch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


TEXT_CANDIDATES = [
    "text",
    "content",
    "article",
    "body",
    "document",
    "news",
    "sentence",
    "paragraph",
    "summary",
]


def make_doc_id(repo_id: str, file_path: str, row_idx: int) -> str:
    import hashlib

    raw = f"{repo_id}|{file_path}|{row_idx}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()


def iter_inventory(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_checkpoint(path: Optional[str]) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)
    return done


def append_checkpoint(path: Optional[str], repo_id: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(repo_id + "\n")


def select_text_column(columns: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in TEXT_CANDIDATES:
        if cand in lower:
            return lower[cand]
    for cand in TEXT_CANDIDATES:
        for col in columns:
            if cand in col.lower():
                return col
    if len(columns) == 1:
        return columns[0]
    return None


def iter_text_from_txt(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_text_from_jsonl(path: str) -> Iterator[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, dict):
                col = select_text_column(list(obj.keys()))
                if col and isinstance(obj.get(col), str):
                    yield obj[col]


def iter_text_from_json(path: str) -> Iterator[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
    except Exception:
        return

    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                yield item
            elif isinstance(item, dict):
                col = select_text_column(list(item.keys()))
                if col and isinstance(item.get(col), str):
                    yield item[col]
        return
    if isinstance(obj, dict):
        # try a common key
        if "data" in obj and isinstance(obj["data"], list):
            for item in obj["data"]:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict):
                    col = select_text_column(list(item.keys()))
                    if col and isinstance(item.get(col), str):
                        yield item[col]


def _markdown_table(headers: List[str], values: List[str]) -> str:
    safe_headers = [h.strip() or "col" for h in headers]
    safe_values = [v.strip() for v in values]
    header_row = "| " + " | ".join(safe_headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(safe_headers)) + " |"
    val_row = "| " + " | ".join(safe_values) + " |"
    return "\n".join([header_row, sep_row, val_row])


def iter_text_from_csv(path: str) -> Iterator[str]:
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except Exception:
        pass
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                sample = f.read(4096)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if not reader.fieldnames:
                return
            text_col = select_text_column(reader.fieldnames)
            for row in reader:
                if text_col:
                    val = row.get(text_col)
                    if isinstance(val, str) and val.strip():
                        yield val
                else:
                    values = [str(row.get(h, "") or "") for h in reader.fieldnames]
                    yield _markdown_table(reader.fieldnames, values)
    except csv.Error as exc:
        logger.warning("CSV parse failed for %s: %s", path, exc)
        return


def iter_text_from_parquet(path: str) -> Iterator[str]:
    pf = pq.ParquetFile(path)
    cols = pf.schema.names
    text_col = select_text_column(cols)
    if text_col:
        for batch in pf.iter_batches(columns=[text_col]):
            for val in batch.column(0).to_pylist():
                if isinstance(val, str) and val.strip():
                    yield val
    else:
        for batch in pf.iter_batches(columns=cols):
            rows = batch.to_pylist()
            for row in rows:
                values = [str(row.get(h, "") or "") for h in cols]
                yield _markdown_table(cols, values)


def iter_text_from_file(path: str) -> Iterator[str]:
    lower = path.lower()
    if lower.endswith(".txt"):
        yield from iter_text_from_txt(path)
    elif lower.endswith(".jsonl"):
        yield from iter_text_from_jsonl(path)
    elif lower.endswith(".json"):
        yield from iter_text_from_json(path)
    elif lower.endswith(".csv") or lower.endswith(".tsv"):
        yield from iter_text_from_csv(path)
    elif lower.endswith(".parquet"):
        yield from iter_text_from_parquet(path)


def collect_text_files(dataset_dir: str, candidates: List[str], max_files: Optional[int]) -> List[str]:
    files: List[str] = []
    if candidates:
        for rel in candidates:
            path = os.path.join(dataset_dir, rel)
            if os.path.isfile(path):
                files.append(path)
    if not files:
        for root, _, filenames in os.walk(dataset_dir):
            for name in filenames:
                lower = name.lower()
                if any(
                    lower.endswith(ext)
                    for ext in (".txt", ".jsonl", ".json", ".csv", ".tsv", ".parquet")
                ):
                    files.append(os.path.join(root, name))
    if max_files:
        files = files[: max_files]
    return files


def load_filter_spec(path: Optional[str]) -> Optional[FilterSpec]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return FilterSpec.from_dict(data)


def default_filter_spec() -> FilterSpec:
    return FilterSpec.from_dict(
        {
            "min_chars": 20,
            "min_words": 3,
            "min_devanagari_ratio": 0.5,
            "short_text": {"max_chars": 80, "min_words": 6, "require_sentence_punct": True},
            "max_digit_ratio": 0.4,
            "max_symbol_ratio": 0.4,
            "max_repeated_char_ratio": 0.3,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Kaggle text datasets to HF")
    parser.add_argument("--inventory", required=True, help="Kaggle inventory JSONL")
    parser.add_argument("--target-repo", required=True, help="Target HF repo (org/name)")
    parser.add_argument("--batch-size", type=int, default=100000, help="Rows per shard")
    parser.add_argument("--token", help="HF token (defaults to cache or HF_TOKEN)")
    parser.add_argument("--dedupe-store", default="data/dedupe_text_hashes.sqlite")
    parser.add_argument("--refresh-dedupe", action="store_true", default=False)
    parser.add_argument("--no-refresh-dedupe", action="store_false", dest="refresh_dedupe")
    parser.add_argument("--max-datasets", type=int, help="Limit number of datasets")
    parser.add_argument("--max-files", type=int, help="Limit files per dataset")
    parser.add_argument("--max-rows", type=int, help="Limit total rows")
    parser.add_argument("--download-dir", default="data/kaggle_downloads")
    parser.add_argument("--keep-files", action="store_true", help="Keep downloaded files")
    parser.add_argument("--include-review", action="store_true", help="Include review datasets too")
    parser.add_argument("--filter-json", help="Path to JSON filter spec override")
    parser.add_argument("--no-tabular-markdown", action="store_true", help="Skip tabular->markdown conversion")
    parser.add_argument("--checkpoint", default="data/kaggle_merge_done.txt", help="Checkpoint file to skip completed repos")
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    hf_api = HfApi()
    token = args.token or get_token()

    repo_exists = True
    try:
        hf_api.repo_info(args.target_repo, repo_type="dataset")
    except Exception:
        repo_exists = False
        hf_api.create_repo(args.target_repo, repo_type="dataset", private=True)

    store = DedupeStore(args.dedupe_store, reset=args.refresh_dedupe)
    try:
        if repo_exists and args.refresh_dedupe:
            prefill_dedupe_from_hf(store, args.target_repo, token=token)

        shard_index = get_max_shard_index(hf_api, args.target_repo) + 1 if repo_exists else 1
        logger.info("Starting shard index: %s", shard_index)

        filter_spec = load_filter_spec(args.filter_json) or default_filter_spec()

        out_rows: List[Dict[str, Any]] = []
        out_hashes: List[bytes] = []
        pending: List[Tuple[bytes, Dict[str, Any]]] = []
        uploaded = 0
        total_rows = 0

        done = load_checkpoint(args.checkpoint)

        for idx, row in enumerate(iter_inventory(args.inventory), start=1):
            if args.max_datasets and idx > args.max_datasets:
                break
            if row.get("quality_bucket") == "reject":
                continue
            if row.get("task_type") != "pretrain" and not args.include_review:
                continue
            if row.get("quality_bucket") != "accept" and not args.include_review:
                continue

            repo_id = row.get("repo_id")
            if not repo_id:
                continue
            if repo_id in done:
                logger.info("Skipping already completed repo: %s", repo_id)
                continue
            logger.info("Processing Kaggle dataset: %s", repo_id)

            dataset_dir = os.path.join(args.download_dir, repo_id.replace("/", "__"))
            os.makedirs(dataset_dir, exist_ok=True)
            api.dataset_download_files(repo_id, path=dataset_dir, unzip=True, quiet=True)

            files = collect_text_files(dataset_dir, row.get("text_file_candidates") or [], args.max_files)
            if not files:
                logger.info("No text files found for %s; skipping", repo_id)
                if not args.keep_files:
                    shutil.rmtree(dataset_dir, ignore_errors=True)
                continue

            for file_path in files:
                lower = file_path.lower()
                if args.no_tabular_markdown and (lower.endswith(".csv") or lower.endswith(".tsv") or lower.endswith(".parquet")):
                    continue
                try:
                    for row_idx, text in enumerate(iter_text_from_file(file_path)):
                        text_norm = normalize_text(text)
                        if not text_norm:
                            continue
                        if not passes_quality(text_norm, filter_spec):
                            continue
                        doc = {
                            "text": text_norm,
                            "source": f"kaggle:{repo_id}",
                            "url": f"https://www.kaggle.com/datasets/{repo_id}",
                            "language": "ne",
                            "doc_id": make_doc_id(repo_id, file_path, row_idx),
                        }
                        h = hash_text(text_norm)
                        pending.append((h, doc))

                        if len(pending) >= 1000:
                            new_items = store.filter_new(pending)
                            pending = []
                            for h_new, row_new in new_items:
                                out_rows.append(row_new)
                                out_hashes.append(h_new)
                                total_rows += 1
                                if args.max_rows and total_rows >= args.max_rows:
                                    break
                            if args.max_rows and total_rows >= args.max_rows:
                                break

                        if len(out_rows) >= args.batch_size:
                            upload_parquet_batch(
                                api=hf_api,
                                repo_id=args.target_repo,
                                token=token,
                                rows=out_rows,
                                shard_index=shard_index,
                            )
                            store.insert_hashes(out_hashes)
                            out_rows = []
                            out_hashes = []
                            shard_index += 1
                            uploaded += 1

                        if args.max_rows and total_rows >= args.max_rows:
                            break
                except Exception as exc:
                    logger.warning("Skipping file due to error %s: %s", file_path, exc)
                    continue
                if args.max_rows and total_rows >= args.max_rows:
                    break

            if not args.keep_files:
                shutil.rmtree(dataset_dir, ignore_errors=True)

            append_checkpoint(args.checkpoint, repo_id)
            done.add(repo_id)

            if args.max_rows and total_rows >= args.max_rows:
                break

        if pending:
            new_items = store.filter_new(pending)
            for h_new, row_new in new_items:
                out_rows.append(row_new)
                out_hashes.append(h_new)
                total_rows += 1
                if args.max_rows and total_rows >= args.max_rows:
                    break

        if out_rows:
            upload_parquet_batch(
                api=hf_api,
                repo_id=args.target_repo,
                token=token,
                rows=out_rows,
                shard_index=shard_index,
            )
            store.insert_hashes(out_hashes)
            uploaded += 1

        logger.info("Kaggle merge complete. Uploaded %s shard(s).", uploaded)
    finally:
        store.close()


if __name__ == "__main__":
    main()
