#!/usr/bin/env python3
"""Compile Devanagari-heavy corpora (HPLT + C4) into one HF dataset.

Outputs a unified dataset with columns:
  text, source, language, doc_id, url

Designed for building a large Devanagari corpus without duplicating Wikipedia/Sangraha.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml

from datasets import Dataset, Features, Value, load_dataset
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

DEFAULT_HPLT_CONFIGS = [
    "npi_Deva",
    "hin_Deva",
    "mar_Deva",
    "mai_Deva",
    "san_Deva",
    "bho_Deva",
    "awa_Deva",
    "mag_Deva",
    "hne_Deva",
    "kas_Deva",
]

DEFAULT_C4_CONFIGS = ["hi", "ne", "mr"]
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "sources" / "devanagari_compile.yml"
)


def parse_list(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")
    return data


def iter_dataset_rows(
    repo_id: str,
    config: str,
    split: str,
    download_first: bool,
) -> Iterator[Dict[str, Any]]:
    ds = load_dataset(repo_id, name=config, split=split, streaming=not download_first)
    for row in ds:
        yield row


def make_source_key(repo_id: str, config: str, split: str) -> str:
    return f"{repo_id}:{config}:{split}"


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
        "url": [row.get("url") for row in rows],
    }

    features = Features(
        {
            "text": Value("string"),
            "source": Value("string"),
            "language": Value("string"),
            "doc_id": Value("string"),
            "url": Value("string"),
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


def _iter_work_plan_from_config(
    cfg: Dict[str, Any],
    *,
    only: Optional[set[str]],
    exclude: Optional[set[str]],
) -> List[Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any], Optional[str]]]:
    datasets = cfg.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("Config 'datasets' must be a list")

    work_plan: List[
        Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any], Optional[str]]
    ] = []
    for entry in datasets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("repo")
        if not name:
            continue
        if only and name not in only:
            continue
        if exclude and name in exclude:
            continue
        if entry.get("enabled", True) is False:
            continue

        repo_id = entry.get("repo")
        if not repo_id:
            continue

        configs = entry.get("configs") or []
        splits = entry.get("splits") or ["train"]
        kind = entry.get("kind") or "generic"
        filters = entry.get("filters") or {}
        fields = entry.get("fields") or {}
        language_override = entry.get("language_override")
        if not isinstance(filters, dict):
            filters = {}
        if not isinstance(fields, dict):
            fields = {}

        for config in configs:
            for split in splits:
                work_plan.append(
                    (repo_id, config, split, kind, filters, fields, language_override)
                )

    return work_plan


def _iter_work_plan_from_args(
    args: argparse.Namespace,
) -> List[
    Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any], Optional[str]]
]:
    work_plan: List[
        Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any], Optional[str]]
    ] = []
    for cfg in parse_list(args.hplt_configs):
        for split in parse_list(args.hplt_splits):
            work_plan.append((args.hplt_repo, cfg, split, "hplt", {}, {}, None))
    for cfg in parse_list(args.c4_configs):
        for split in parse_list(args.c4_splits):
            work_plan.append((args.c4_repo, cfg, split, "c4", {}, {}, None))
    return work_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Devanagari corpora to HF")
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--hplt-repo", default="HPLT/HPLT2.0_cleaned")
    parser.add_argument("--c4-repo", default="allenai/c4")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config listing datasets to compile",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated dataset names from config to run (overrides enabled)",
    )
    parser.add_argument(
        "--exclude",
        help="Comma-separated dataset names from config to skip",
    )
    parser.add_argument(
        "--hplt-configs",
        default=",".join(DEFAULT_HPLT_CONFIGS),
        help="Comma-separated HPLT configs (e.g. npi_Deva,hin_Deva,...)",
    )
    parser.add_argument(
        "--c4-configs",
        default=",".join(DEFAULT_C4_CONFIGS),
        help="Comma-separated C4 configs (e.g. hi,ne,mr)",
    )
    parser.add_argument("--hplt-splits", default="train")
    parser.add_argument("--c4-splits", default="train")
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument(
        "--checkpoint",
        default="data/devanagari_compile_done.txt",
        help="Checkpoint file of completed dataset/config/split keys",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Download full split to cache before iterating",
    )
    parser.add_argument(
        "--require-empty",
        action="store_true",
        help="Fail if target repo already has shards (safety guard).",
    )
    parser.add_argument(
        "--min-start-index",
        type=int,
        default=0,
        help="Fail if computed starting shard index is <= this value.",
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

    max_existing = get_max_shard_index(api, args.target_repo)
    shard_index = max_existing + 1
    logger.info("Existing max shard index: %s", max_existing)
    logger.info("Starting shard index: %s", shard_index)

    if args.require_empty and max_existing > 0:
        raise SystemExit(
            f"Refusing to run: target repo has shards (max index {max_existing})."
        )
    if shard_index <= args.min_start_index:
        raise SystemExit(
            f"Refusing to run: starting shard index {shard_index} <= {args.min_start_index}."
        )

    only = set(parse_list(args.only)) if args.only else None
    exclude = set(parse_list(args.exclude)) if args.exclude else None

    config_path = Path(args.config) if args.config else None
    work_plan: List[
        Tuple[str, str, str, str, Dict[str, Any], Dict[str, Any], Optional[str]]
    ]
    if config_path and config_path.exists():
        cfg = load_config(config_path)
        work_plan = _iter_work_plan_from_config(cfg, only=only, exclude=exclude)
        if not work_plan:
            logger.warning("No work items from config; falling back to CLI lists.")
            work_plan = _iter_work_plan_from_args(args)
    else:
        work_plan = _iter_work_plan_from_args(args)

    for repo_id, config, split, kind, filters, fields, language_override in work_plan:
        source_key = make_source_key(repo_id, config, split)
        if source_key in done:
            logger.info("Skipping completed: %s", source_key)
            continue

        logger.info("Processing %s", source_key)
        batch: List[Dict[str, Any]] = []
        batches_written = 0

        for row_idx, row in enumerate(
            iter_dataset_rows(repo_id, config, split, args.download_first)
        ):
            text_field = fields.get("text") or "text"
            url_field = fields.get("url") or "url"
            doc_id_field = fields.get("doc_id") or "doc_id"
            lang_field = fields.get("language") or "lang"

            text_raw = row.get(text_field)
            if not isinstance(text_raw, str):
                continue
            text_norm = normalize_text(text_raw)
            if not text_norm:
                continue

            if filters:
                row_source = row.get("source")
                source_include = filters.get("source_include")
                source_exclude = filters.get("source_exclude")
                if source_include:
                    if row_source is None or row_source not in source_include:
                        continue
                if source_exclude and row_source in source_exclude:
                    continue

                lang_equals = filters.get("lang_equals")
                if lang_equals and isinstance(lang_equals, dict):
                    field = lang_equals.get("field")
                    value = lang_equals.get("value")
                    if field and value is not None:
                        if row.get(field) != value:
                            continue

            if kind == "hplt" or repo_id == args.hplt_repo:
                lang = row.get(lang_field) or config
                url = row.get("u") or row.get(url_field)
                doc_id = (
                    row.get("id")
                    or row.get(doc_id_field)
                    or row.get("u")
                    or f"{source_key}:{row_idx}"
                )
            elif kind == "c4" or repo_id == args.c4_repo:
                lang = config
                url = row.get(url_field)
                doc_id = (
                    row.get("id")
                    or row.get(doc_id_field)
                    or row.get(url_field)
                    or f"{source_key}:{row_idx}"
                )
            elif kind == "sangraha":
                lang = split
                url = row.get(url_field)
                doc_id = (
                    row.get(doc_id_field)
                    or row.get("id")
                    or f"{source_key}:{row_idx}"
                )
            else:
                lang = row.get(lang_field) or config
                url = row.get(url_field) or row.get("u")
                doc_id = (
                    row.get("id")
                    or row.get(doc_id_field)
                    or row.get(url_field)
                    or row.get("u")
                    or f"{source_key}:{row_idx}"
                )

            if language_override:
                lang = language_override

            batch.append(
                {
                    "text": text_norm,
                    "source": source_key,
                    "language": str(lang),
                    "doc_id": str(doc_id),
                    "url": str(url) if url is not None else None,
                }
            )

            if len(batch) >= args.batch_size:
                upload_parquet_batch(
                    api=api,
                    repo_id=args.target_repo,
                    token=token,
                    rows=batch,
                    shard_index=shard_index,
                    split_name="train",
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
                split_name="train",
            )
            shard_index += 1

        if args.checkpoint:
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            with open(args.checkpoint, "a", encoding="utf-8") as f:
                f.write(source_key + "\n")

    logger.info("Devanagari compile complete.")


if __name__ == "__main__":
    main()
