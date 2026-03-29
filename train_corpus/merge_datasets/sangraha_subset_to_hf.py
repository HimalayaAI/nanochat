#!/usr/bin/env python3
"""Build a Nepali subset of ai4bharat/sangraha and upload to HF.

Creates a slim dataset with columns: text, source, language, doc_id, type.
Intended for tokenizer sampling and corpus consolidation.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from datasets import load_dataset, Dataset, Features, Value
from huggingface_hub import HfApi, get_token, login

# Ensure project root is on sys.path for scripts.* imports
import sys

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.merge_datasets.merge_corpus_to_hf import get_max_shard_index
from scripts.merge_datasets.quality_filters import normalize_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_parquet_files(
    api: HfApi, repo_id: str, subset: str, split: str
) -> List[str]:
    prefix = f"{subset}/{split}/"
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning("Failed to list repo files for %s: %s", repo_id, exc)
        return []
    parquet_files = [
        f for f in files if f.startswith(prefix) and f.endswith(".parquet")
    ]
    parquet_files.sort()
    return [f"hf://datasets/{repo_id}/{path}" for path in parquet_files]


def iter_sangraha_rows(
    repo_id: str,
    subset: str,
    split: str,
    download_first: bool,
    api: HfApi,
) -> Iterator[Dict[str, Any]]:
    data_files = resolve_parquet_files(api, repo_id, subset, split)
    if data_files:
        logger.info("Using %s parquet files for %s/%s", len(data_files), subset, split)
        ds = load_dataset(
            "parquet",
            data_files=data_files,
            split="train",
            streaming=not download_first,
        )
    else:
        logger.warning(
            "No parquet files found for %s/%s; falling back to dataset builder",
            subset,
            split,
        )
        ds = load_dataset(
            repo_id, name=subset, split=split, streaming=not download_first
        )
    for row in ds:
        yield row


def make_source_key(repo_id: str, subset: str, split: str) -> str:
    return f"{repo_id}:{subset}/{split}"


def upload_parquet_batch(
    *,
    api: HfApi,
    repo_id: str,
    token: str,
    rows: List[Dict[str, Any]],
    shard_index: int,
    split_name: str,
) -> None:
    data_dict = {
        "text": [row.get("text") for row in rows],
        "source": [row.get("source") for row in rows],
        "language": [row.get("language") for row in rows],
        "doc_id": [row.get("doc_id") for row in rows],
        "type": [row.get("type") for row in rows],
    }

    features = Features(
        {
            "text": Value("string"),
            "source": Value("string"),
            "language": Value("string"),
            "doc_id": Value("string"),
            "type": Value("string"),
        }
    )
    hf_dataset = Dataset.from_dict(data_dict, features=features)
    os.makedirs("data/hf_merge_export", exist_ok=True)
    parquet_path = (
        f"data/hf_merge_export/{split_name}-{shard_index:06d}-of-000000.parquet"
    )
    repo_path = f"data/{split_name}-{shard_index:06d}-of-000000.parquet"
    hf_dataset.to_parquet(parquet_path)

    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    os.remove(parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sangraha subset to HF")
    parser.add_argument("--source-repo", default="ai4bharat/sangraha")
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--verified-splits", default="nep")
    parser.add_argument("--synthetic-splits", default="npi_Deva")
    parser.add_argument("--unverified-splits", default="nep")
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument(
        "--checkpoint",
        default="data/sangraha_subset_done.txt",
        help="Checkpoint file of completed subset/split keys",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Download full split to cache before iterating",
    )
    args = parser.parse_args()

    token = get_token() or os.getenv("HF_TOKEN")
    if not token:
        login()
        token = get_token()

    api = HfApi(token=token)
    try:
        api.repo_info(args.target_repo, repo_type="dataset", token=token)
    except Exception:
        api.create_repo(
            repo_id=args.target_repo,
            repo_type="dataset",
            exist_ok=True,
            token=token,
        )

    done: set[str] = set()
    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(line)

    shard_index = get_max_shard_index(args.target_repo, token) + 1
    logger.info("Starting shard index: %s", shard_index)

    work_plan: List[Tuple[str, str]] = []
    for split in parse_list(args.verified_splits):
        work_plan.append(("verified", split))
    for split in parse_list(args.synthetic_splits):
        work_plan.append(("synthetic", split))
    for split in parse_list(args.unverified_splits):
        work_plan.append(("unverified", split))

    for subset, split in work_plan:
        split_name = f"{subset}_{split}"
        source_key = make_source_key(args.source_repo, subset, split)
        if source_key in done:
            logger.info("Skipping completed: %s", source_key)
            continue

        logger.info("Processing %s", source_key)
        batch: List[Dict[str, Any]] = []
        batches_written = 0

        for row_idx, row in enumerate(
            iter_sangraha_rows(
                args.source_repo, subset, split, args.download_first, api
            )
        ):
            text_raw = row.get("text")
            if not isinstance(text_raw, str):
                continue
            text_norm = normalize_text(text_raw)
            if not text_norm:
                continue

            doc_id = row.get("doc_id")
            if doc_id is None:
                doc_id = f"{source_key}:{row_idx}"

            row_type = row.get("type")

            batch.append(
                {
                    "text": text_norm,
                    "source": source_key,
                    "language": split,
                    "doc_id": str(doc_id),
                    "type": str(row_type) if row_type is not None else None,
                }
            )

            if len(batch) >= args.batch_size:
                upload_parquet_batch(
                    api=api,
                    repo_id=args.target_repo,
                    token=token,
                    rows=batch,
                    shard_index=shard_index,
                    split_name=split_name,
                )
                shard_index += 1
                batches_written += 1
                batch = []
                if args.max_batches and batches_written >= args.max_batches:
                    break

        if batch:
            upload_parquet_batch(
                api=api,
                repo_id=args.target_repo,
                token=token,
                rows=batch,
                shard_index=shard_index,
                split_name=split_name,
            )
            shard_index += 1

        if args.checkpoint:
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            with open(args.checkpoint, "a", encoding="utf-8") as f:
                f.write(source_key + "\n")

    logger.info("Sangraha subset export complete.")


if __name__ == "__main__":
    main()
