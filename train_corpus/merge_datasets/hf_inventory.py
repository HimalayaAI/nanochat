#!/usr/bin/env python3
"""
Build an inventory of Hugging Face datasets matching "nepali" and emit JSONL with
schema + suggested column mappings for merge.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import requests
from datasets import load_dataset
from huggingface_hub import HfApi


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
    "Article",
    "sentence",
    "paragraph",
    "summary",
]
URL_CANDIDATES = ["url", "link", "source", "Source"]
LANG_CANDIDATES = ["language", "lang", "Language"]
DOC_ID_CANDIDATES = ["id", "doc_id", "docid", "index"]

SFT_CANDIDATES = {
    "instruction",
    "input",
    "output",
    "prompt",
    "response",
    "question",
    "answer",
    "messages",
    "conversations",
    "dialogue",
    "chat",
    "system",
    "assistant",
    "user",
}
LABEL_CANDIDATES = {
    "label",
    "labels",
    "category",
    "class",
    "sentiment",
    "tag",
    "tags",
}
PARALLEL_PAIRS = [
    ("incorrect_sentence", "correct_sentence"),
    ("source", "target"),
    ("src", "tgt"),
    ("sentence1", "sentence2"),
    ("text1", "text2"),
    ("premise", "hypothesis"),
]
MODALITY_AUDIO_HINTS = {"audio", "speech", "asr", "tts", "wav", "voice"}
MODALITY_IMAGE_HINTS = {"image", "vision", "video", "frame", "pixel"}


def _best_match(columns: List[str], candidates: List[str]) -> Optional[str]:
    if not columns:
        return None
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in lower_map:
            return lower_map[cand_lower]
    # Fuzzy: substring match
    for cand in candidates:
        cand_lower = cand.lower()
        for col in columns:
            if cand_lower in col.lower():
                return col
    return None


def suggest_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    return {
        "text": _best_match(columns, TEXT_CANDIDATES),
        "url": _best_match(columns, URL_CANDIDATES),
        "language": _best_match(columns, LANG_CANDIDATES),
        "doc_id": _best_match(columns, DOC_ID_CANDIDATES),
    }


def _extract_columns_from_features(features: Any) -> List[str]:
    if isinstance(features, dict):
        return list(features.keys())
    return []


def _extract_feature_types(features: Any) -> Dict[str, str]:
    types: Dict[str, str] = {}
    if not isinstance(features, dict):
        return types
    for col, spec in features.items():
        if isinstance(spec, dict):
            feature_type = spec.get("_type") or spec.get("type") or spec.get("dtype")
            if isinstance(feature_type, str):
                types[col] = feature_type
    return types


def triage_dataset(
    columns: List[str],
    feature_types: Dict[str, str],
    mapping: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    colset = {c.lower() for c in columns}

    audio_present = any(
        t.lower() == "audio" for t in feature_types.values() if isinstance(t, str)
    ) or any(any(hint in col for hint in MODALITY_AUDIO_HINTS) for col in colset)
    image_present = any(
        t.lower() == "image" for t in feature_types.values() if isinstance(t, str)
    ) or any(any(hint in col for hint in MODALITY_IMAGE_HINTS) for col in colset)

    if audio_present and image_present:
        modality = "mixed"
    elif audio_present:
        modality = "speech"
    elif image_present:
        modality = "image"
    else:
        modality = "text_only"

    is_sft = any(c in colset for c in SFT_CANDIDATES)
    is_parallel = any(a in colset and b in colset for a, b in PARALLEL_PAIRS)
    is_classification = any(c in colset for c in LABEL_CANDIDATES)

    if is_sft:
        task_type = "sft"
    elif is_parallel:
        task_type = "parallel_text"
    elif is_classification:
        task_type = "classification"
    else:
        task_type = "pretrain"

    has_text = bool(mapping.get("text"))

    if not has_text:
        quality_bucket = "reject"
    elif modality == "text_only" and task_type == "pretrain":
        quality_bucket = "accept"
    else:
        quality_bucket = "review"

    review_needed = quality_bucket != "accept"

    notes: List[str] = []
    if audio_present:
        notes.append("audio feature present")
    if image_present:
        notes.append("image feature present")
    if is_sft:
        notes.append("sft fields present")
    if is_parallel:
        notes.append("parallel text fields present")
    if is_classification:
        notes.append("label field present")

    return {
        "modality": modality,
        "task_type": task_type,
        "quality_bucket": quality_bucket,
        "review_needed": review_needed,
        "triage_notes": "; ".join(notes) if notes else None,
    }


def fetch_info(
    repo_id: str,
    *,
    timeout: int,
    retries: int,
) -> Optional[Dict[str, Any]]:
    url = "https://datasets-server.huggingface.co/info"
    params = {"dataset": repo_id}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"{resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                logger.warning("Info fetch failed for %s: %s", repo_id, exc)
                return None
            sleep_s = min(2 ** attempt, 10)
            time.sleep(sleep_s)
    return None


def fallback_columns(repo_id: str, split: str) -> List[str]:
    try:
        ds = load_dataset(repo_id, split=split, streaming=True)
        for row in ds:
            if isinstance(row, dict):
                return list(row.keys())
            break
    except Exception:
        return []
    return []


def iter_datasets(search: str, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    api = HfApi()
    count = 0
    for ds in api.list_datasets(search=search, full=True):
        yield ds
        count += 1
        if limit and count >= limit:
            break


def build_inventory_rows(
    repo_id: str,
    info: Optional[Dict[str, Any]],
    *,
    min_text_columns: int,
    timeout: int,
    retries: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    notes: List[str] = []

    if not info or "dataset_info" not in info:
        notes.append("info endpoint failed")
        # Fallback to default split only
        columns = fallback_columns(repo_id, "train")
        mapping = suggest_mapping(columns)
        triage = triage_dataset(columns, {}, mapping)
        usable = bool(mapping.get("text"))
        rows.append(
            {
                "repo_id": repo_id,
                "config": "default",
                "split": "train",
                "features": {"columns": columns},
                "mapping_suggested": mapping,
                "usable": usable,
                **triage,
                "notes": "; ".join(notes) or None,
            }
        )
        return rows

    dataset_info = info.get("dataset_info", {}) or {}
    for config_name, cfg in dataset_info.items():
        notes = []
        features = cfg.get("features") or {}
        splits = cfg.get("splits") or {}
        split_names = list(splits.keys()) or ["train"]
        columns = _extract_columns_from_features(features)
        feature_types = _extract_feature_types(features)
        if not columns:
            # fallback to streaming for this config
            columns = fallback_columns(repo_id, split_names[0])
            if columns:
                notes.append("features missing; fallback to streaming")
        mapping = suggest_mapping(columns)
        triage = triage_dataset(columns, feature_types, mapping)
        text_cols = 1 if mapping.get("text") else 0
        usable = text_cols >= min_text_columns
        if not mapping.get("text"):
            notes.append("no text-like column")
        for split in split_names:
            rows.append(
                {
                    "repo_id": repo_id,
                    "config": config_name,
                    "split": split,
                    "features": features if features else {"columns": columns},
                    "mapping_suggested": mapping,
                    "usable": usable,
                    **triage,
                    "notes": "; ".join(notes) if notes else None,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory HF datasets matching 'nepali'")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--approved-output",
        default="sources/nepali_pretrain_sources.jsonl",
        help="Output JSONL for pretrain-accepted sources",
    )
    parser.add_argument(
        "--sft-output",
        default="sources/nepali_sft_sources.jsonl",
        help="Output JSONL for SFT sources",
    )
    parser.add_argument("--limit", type=int, help="Limit number of datasets")
    parser.add_argument("--min-text-columns", type=int, default=1, help="Min text columns to mark usable")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout for info endpoint")
    parser.add_argument("--retry", type=int, default=3, help="Retries for info endpoint")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.approved_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.sft_output), exist_ok=True)

    total_rows = 0
    approved_rows = 0
    sft_rows = 0
    with (
        open(args.output, "w", encoding="utf-8") as f,
        open(args.approved_output, "w", encoding="utf-8") as fa,
        open(args.sft_output, "w", encoding="utf-8") as fs,
    ):
        for ds in iter_datasets("nepali", args.limit):
            repo_id = getattr(ds, "id", None) or getattr(ds, "repo_id", None)
            if not repo_id:
                continue
            info = fetch_info(repo_id, timeout=args.timeout, retries=args.retry)
            rows = build_inventory_rows(
                repo_id,
                info,
                min_text_columns=args.min_text_columns,
                timeout=args.timeout,
                retries=args.retry,
            )
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1
                if row.get("quality_bucket") == "accept" and row.get("task_type") == "pretrain":
                    fa.write(json.dumps(row, ensure_ascii=False) + "\n")
                    approved_rows += 1
                if row.get("task_type") == "sft" and row.get("mapping_suggested", {}).get("text"):
                    fs.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sft_rows += 1

    logger.info("Wrote %d inventory rows to %s", total_rows, args.output)
    logger.info("Wrote %d pretrain-accepted rows to %s", approved_rows, args.approved_output)
    logger.info("Wrote %d SFT rows to %s", sft_rows, args.sft_output)


if __name__ == "__main__":
    main()
