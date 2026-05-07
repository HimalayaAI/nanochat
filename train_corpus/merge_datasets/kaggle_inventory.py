#!/usr/bin/env python3
"""
Inventory Kaggle datasets matching a search term (e.g., "nepali") and emit JSONL.

Requires Kaggle API credentials:
  - ~/.kaggle/kaggle.json
  - or env KAGGLE_USERNAME + KAGGLE_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Kaggle API not available. Install with: pip install kaggle"
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


TEXT_FILE_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".txt"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

SFT_HINTS = {
    "instruction",
    "prompt",
    "response",
    "answer",
    "question",
    "chat",
    "conversation",
    "dialogue",
    "messages",
    "assistant",
    "user",
}
PARALLEL_HINTS = {
    "parallel",
    "translation",
    "bilingual",
    "aligned",
    "source",
    "target",
}
CLASSIFICATION_HINTS = {
    "sentiment",
    "classification",
    "label",
    "category",
    "class",
}
PRETRAIN_HINTS = {
    "corpus",
    "news",
    "wikipedia",
    "wiki",
    "oscar",
    "text",
    "articles",
    "article",
    "dataset",
}
NON_TEXT_HINTS = {
    "image",
    "audio",
    "speech",
    "asr",
    "tts",
    "vision",
    "face",
    "character",
    "hand sign",
    "license plate",
    "stock",
    "price",
    "housing",
    "market",
    "ocr",
}


def iter_kaggle_datasets(
    api: KaggleApi, search: str, limit: Optional[int]
) -> Iterable[Any]:
    page = 1
    count = 0
    while True:
        results = api.dataset_list(search=search, page=page)
        if not results:
            break
        for ds in results:
            yield ds
            count += 1
            if limit and count >= limit:
                return
        page += 1


def dataset_file_list(
    api: KaggleApi,
    ref: str,
    *,
    retries: int,
    backoff_s: float,
) -> List[Dict[str, Any]]:
    files = None
    for attempt in range(retries):
        try:
            files = api.dataset_list_files(ref).files
            break
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429 and attempt < retries - 1:
                sleep_s = min(backoff_s * (2 ** attempt), 60.0)
                logger.warning(
                    "Rate limited on %s; sleeping %.1fs before retry (%d/%d)",
                    ref,
                    sleep_s,
                    attempt + 1,
                    retries,
                )
                time.sleep(sleep_s)
                continue
            raise
    if files is None:
        return []
    file_rows: List[Dict[str, Any]] = []
    for f in files:
        size = getattr(f, "size", None)
        if size is None:
            size = getattr(f, "fileSize", None)
        if size is None:
            size = getattr(f, "totalBytes", None)
        file_rows.append(
            {
                "name": f.name,
                "size": size,
                "type": getattr(f, "fileType", None),
                "creationDate": getattr(f, "creationDate", None),
            }
        )
    return file_rows


def suggest_text_files(files: List[Dict[str, Any]]) -> List[str]:
    candidates: List[str] = []
    for f in files:
        name = (f.get("name") or "").lower()
        for ext in TEXT_FILE_EXTS:
            if name.endswith(ext):
                candidates.append(f.get("name") or "")
                break
    return candidates


def _ext_from_name(name: str) -> str:
    name = name.lower()
    for ext in IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS | TEXT_FILE_EXTS:
        if name.endswith(ext):
            return ext
    return ""


def triage_dataset(
    title: Optional[str],
    subtitle: Optional[str],
    files: List[Dict[str, Any]],
    text_files: List[str],
) -> Dict[str, Any]:
    title_l = (title or "").lower()
    subtitle_l = (subtitle or "").lower()
    text = f"{title_l} {subtitle_l}"

    exts = {_ext_from_name(f.get("name", "") or "") for f in files}
    exts.discard("")

    audio_present = any(ext in AUDIO_EXTS for ext in exts)
    image_present = any(ext in IMAGE_EXTS for ext in exts)
    video_present = any(ext in VIDEO_EXTS for ext in exts)

    if audio_present or "audio" in text or "speech" in text or "asr" in text or "tts" in text:
        modality = "speech"
    elif image_present or video_present or "image" in text or "vision" in text:
        modality = "image"
    else:
        modality = "text_only"

    task_type = "pretrain"
    if any(h in text for h in SFT_HINTS):
        task_type = "sft"
    elif any(h in text for h in PARALLEL_HINTS):
        task_type = "parallel_text"
    elif any(h in text for h in CLASSIFICATION_HINTS):
        task_type = "classification"

    # Additional heuristic: if no text files, likely non-text or tabular
    if not text_files and modality != "text_only":
        quality_bucket = "reject"
    elif modality == "text_only" and task_type == "pretrain" and text_files:
        quality_bucket = "accept"
    else:
        quality_bucket = "review"

    notes = []
    if any(h in text for h in NON_TEXT_HINTS):
        notes.append("non-text hint in title/subtitle")
    if modality != "text_only":
        notes.append(f"modality={modality}")
    if task_type != "pretrain":
        notes.append(f"task_type={task_type}")
    if not text_files:
        notes.append("no obvious text files")

    return {
        "modality": modality,
        "task_type": task_type,
        "quality_bucket": quality_bucket,
        "review_needed": quality_bucket != "accept",
        "triage_notes": "; ".join(notes) if notes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory Kaggle datasets matching a search term")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--search", default="nepali", help="Search term")
    parser.add_argument("--limit", type=int, help="Limit number of datasets")
    parser.add_argument(
        "--approved-output",
        default="sources/kaggle_pretrain_sources.jsonl",
        help="Output JSONL for pretrain-accepted Kaggle sources",
    )
    parser.add_argument(
        "--sft-output",
        default="sources/kaggle_sft_sources.jsonl",
        help="Output JSONL for SFT Kaggle sources",
    )
    parser.add_argument(
        "--skip-files",
        action="store_true",
        help="Skip listing files to avoid rate limits (faster inventory)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Sleep seconds between dataset file requests",
    )
    parser.add_argument(
        "--file-retries",
        type=int,
        default=5,
        help="Retries for dataset file listing",
    )
    parser.add_argument(
        "--file-backoff",
        type=float,
        default=2.0,
        help="Base backoff seconds for dataset file listing",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.approved_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.sft_output), exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    total = 0
    approved = 0
    sft_rows = 0
    with (
        open(args.output, "w", encoding="utf-8") as f,
        open(args.approved_output, "w", encoding="utf-8") as fa,
        open(args.sft_output, "w", encoding="utf-8") as fs,
    ):
        for ds in iter_kaggle_datasets(api, args.search, args.limit):
            ref = getattr(ds, "ref", None)
            if not ref:
                continue
            if args.skip_files:
                files = []
                text_files = []
                notes = "file listing skipped"
            else:
                files = dataset_file_list(
                    api, ref, retries=args.file_retries, backoff_s=args.file_backoff
                )
                text_files = suggest_text_files(files)
                notes = None if text_files else "no obvious text file extensions"
            triage = triage_dataset(
                getattr(ds, "title", None),
                getattr(ds, "subtitle", None),
                files,
                text_files,
            )
            row = {
                "repo_id": ref,  # Kaggle ref, e.g. owner/dataset
                "title": getattr(ds, "title", None),
                "subtitle": getattr(ds, "subtitle", None),
                "size": getattr(ds, "size", None),
                "lastUpdated": getattr(ds, "lastUpdated", None),
                "downloadCount": getattr(ds, "downloadCount", None),
                "voteCount": getattr(ds, "voteCount", None),
                "usabilityRating": getattr(ds, "usabilityRating", None),
                "files": files,
                "text_file_candidates": text_files,
                "notes": notes,
                **triage,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            if triage.get("quality_bucket") == "accept" and triage.get("task_type") == "pretrain":
                fa.write(json.dumps(row, ensure_ascii=False) + "\n")
                approved += 1
            if triage.get("task_type") == "sft":
                fs.write(json.dumps(row, ensure_ascii=False) + "\n")
                sft_rows += 1
            if args.sleep and not args.skip_files:
                time.sleep(args.sleep)

    logger.info("Wrote %d Kaggle inventory rows to %s", total, args.output)
    logger.info("Wrote %d pretrain-accepted rows to %s", approved, args.approved_output)
    logger.info("Wrote %d SFT rows to %s", sft_rows, args.sft_output)


if __name__ == "__main__":
    main()
