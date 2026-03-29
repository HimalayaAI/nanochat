#!/usr/bin/env python3
"""
Estimate total token counts for HF datasets using streaming samples.

- Supports multiple datasets via a YAML/JSON config (similar to tokenizer_sources.yml)
- Optional stratify_by (single field or list) to estimate per-stratum totals
- Optional quality filters (same schema as sample_tokenizer_corpus.py)

Estimation is approximate unless --full-scan is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml
from datasets import load_dataset, get_dataset_infos

# Ensure project root is on sys.path for scripts.* imports
import sys

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from train_corpus.merge_datasets.quality_filters import FilterSpec, normalize_text, passes_quality


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def item_get(item: Any, key: str) -> Any:
    if hasattr(item, "get"):
        try:
            return item.get(key)
        except Exception:
            pass
    try:
        return item[key]
    except Exception:
        return None


def get_field_value(item: Any, field_spec: Any) -> Any:
    if field_spec is None:
        return None
    if isinstance(field_spec, (list, tuple)):
        for spec in field_spec:
            val = get_field_value(item, spec)
            if val is not None:
                return val
        return None
    if isinstance(field_spec, str) and "." in field_spec:
        current = item
        for part in field_spec.split("."):
            if current is None:
                return None
            current = item_get(current, part)
        return current
    if isinstance(field_spec, str):
        return item_get(item, field_spec)
    return None


def load_config(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        return json.loads(Path(path).read_text())
    return yaml.safe_load(Path(path).read_text()) or {}


def load_encoder(name: str):
    if name == "tiktoken":
        try:
            import tiktoken
        except Exception as exc:
            raise SystemExit(
                "tiktoken not available. Install with: pip install tiktoken"
            ) from exc
        return tiktoken.get_encoding("cl100k_base")
    raise ValueError(f"Unsupported token estimator: {name}")


def encode_len(encoder, text: str) -> int:
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return 0


@dataclass
class SourcePlan:
    source_id: str
    source_key: str
    split: str
    config: Optional[str]
    field: Any
    tokens_field: Optional[Any]
    shuffle_buffer: int
    filters: Optional[FilterSpec]
    stratify_by: List[str]
    strata: Dict[str, List[str]]
    include_other: bool
    sample_rows: int
    full_scan: bool


def build_sources(cfg: Dict[str, Any]) -> List[SourcePlan]:
    sources_cfg = cfg.get("sources") or []
    if not sources_cfg:
        raise SystemExit("No sources defined in config")

    default_filters = None
    if isinstance(cfg.get("filters"), dict):
        default_filters = FilterSpec.from_dict(cfg["filters"])

    plans: List[SourcePlan] = []
    for src in sources_cfg:
        source_id = src.get("id")
        if not source_id:
            continue
        split = src.get("split", "train")
        config = src.get("config")
        field = src.get("field", "text")
        tokens_field = src.get("tokens_field")
        shuffle_buffer = int(src.get("shuffle_buffer", 0))

        raw_stratify = src.get("stratify_by")
        if raw_stratify is None:
            stratify_by: List[str] = []
        elif isinstance(raw_stratify, str):
            stratify_by = [raw_stratify]
        elif isinstance(raw_stratify, list):
            stratify_by = [str(v) for v in raw_stratify]
        else:
            raise SystemExit(f"Invalid stratify_by value for {source_id}: {raw_stratify}")

        raw_strata = src.get("strata")
        strata: Dict[str, List[str]] = {}
        if raw_strata is not None:
            if isinstance(raw_strata, dict):
                for key, vals in raw_strata.items():
                    if vals is None:
                        continue
                    strata[str(key)] = [str(v) for v in vals]
            elif isinstance(raw_strata, list):
                if len(stratify_by) != 1:
                    raise SystemExit(
                        f"strata list provided but stratify_by is not a single field for {source_id}"
                    )
                strata[stratify_by[0]] = [str(v) for v in raw_strata]
            else:
                raise SystemExit(f"Invalid strata value for {source_id}: {raw_strata}")

        include_other = bool(src.get("include_other", True))
        sample_rows = int(src.get("sample_rows", cfg.get("sample_rows", 200000)))
        full_scan = bool(src.get("full_scan", cfg.get("full_scan", False)))

        filters = default_filters
        if isinstance(src.get("filters"), dict):
            filters = filters.merge(src["filters"]) if filters else FilterSpec.from_dict(
                src["filters"]
            )

        source_key = f"{source_id}:{config or 'default'}:{split}"
        plans.append(
            SourcePlan(
                source_id=source_id,
                source_key=source_key,
                split=split,
                config=config,
                field=field,
                tokens_field=tokens_field,
                shuffle_buffer=shuffle_buffer,
                filters=filters,
                stratify_by=stratify_by,
                strata=strata,
                include_other=include_other,
                sample_rows=sample_rows,
                full_scan=full_scan,
            )
        )
    return plans


def get_num_rows(source_id: str, config: Optional[str], split: str) -> Optional[int]:
    try:
        infos = get_dataset_infos(source_id)
        if config is None:
            # pick first config
            info = next(iter(infos.values()))
        else:
            info = infos.get(config)
        if not info:
            return None
        split_info = info.splits.get(split)
        if not split_info:
            return None
        return int(split_info.num_examples)
    except Exception:
        return None


def iter_rows(plan: SourcePlan) -> Iterator[Dict[str, Any]]:
    if plan.config:
        ds = load_dataset(plan.source_id, name=plan.config, split=plan.split, streaming=True)
    else:
        ds = load_dataset(plan.source_id, split=plan.split, streaming=True)
    if plan.shuffle_buffer and plan.shuffle_buffer > 0:
        try:
            ds = ds.shuffle(buffer_size=plan.shuffle_buffer, seed=42)
        except Exception:
            logger.warning("Shuffle failed for %s; continuing without shuffle", plan.source_key)
    for row in ds:
        yield row


def normalize_strata_key(values: List[str], fields: List[str]) -> str:
    return "|".join(f"{field}={val}" for field, val in zip(fields, values))


def estimate_source(plan: SourcePlan, encoder) -> Dict[str, Any]:
    total_rows = get_num_rows(plan.source_id, plan.config, plan.split)
    rows_seen = 0
    rows_kept = 0
    token_sum = 0

    strata_tokens: Dict[str, int] = {}
    strata_rows: Dict[str, int] = {}

    for row in iter_rows(plan):
        rows_seen += 1

        text_raw = get_field_value(row, plan.field)
        if not isinstance(text_raw, str):
            continue
        text_norm = normalize_text(text_raw)
        if not text_norm:
            continue
        if plan.filters and not passes_quality(text_norm, plan.filters):
            continue

        if plan.tokens_field:
            tok_val = get_field_value(row, plan.tokens_field)
            try:
                tok_count = int(tok_val)
            except Exception:
                tok_count = 0
        else:
            tok_count = encode_len(encoder, text_norm)

        if tok_count <= 0:
            continue

        rows_kept += 1
        token_sum += tok_count

        if plan.stratify_by:
            values: List[str] = []
            valid = True
            for field in plan.stratify_by:
                allowed = plan.strata.get(field)
                raw_val = get_field_value(row, field)
                if raw_val is None:
                    if plan.include_other and allowed is not None and "other" in allowed:
                        values.append("other")
                    elif plan.include_other and allowed is None:
                        values.append("other")
                    else:
                        valid = False
                        break
                else:
                    val = str(raw_val)
                    if allowed is None:
                        values.append(val)
                    elif val in allowed:
                        values.append(val)
                    elif plan.include_other and "other" in allowed:
                        values.append("other")
                    else:
                        valid = False
                        break
            if valid:
                key = normalize_strata_key(values, plan.stratify_by)
                strata_tokens[key] = strata_tokens.get(key, 0) + tok_count
                strata_rows[key] = strata_rows.get(key, 0) + 1

        if not plan.full_scan and rows_kept >= plan.sample_rows:
            break

    kept_rate = (rows_kept / rows_seen) if rows_seen else 0.0
    avg_tokens_per_kept = (token_sum / rows_kept) if rows_kept else 0.0
    avg_tokens_per_seen = (token_sum / rows_seen) if rows_seen else 0.0

    est_total_tokens = None
    est_kept_rows = None
    if total_rows is not None and rows_seen > 0:
        est_kept_rows = int(total_rows * kept_rate)
        est_total_tokens = int(avg_tokens_per_seen * total_rows)

    strata_estimates = {}
    if plan.stratify_by and strata_rows:
        if total_rows is not None and rows_seen > 0:
            est_kept_rows_total = total_rows * kept_rate
            for key, s_rows in strata_rows.items():
                frac = s_rows / rows_kept if rows_kept else 0.0
                est_rows = est_kept_rows_total * frac
                avg_tok = strata_tokens[key] / s_rows if s_rows else 0.0
                strata_estimates[key] = {
                    "sample_rows": s_rows,
                    "sample_tokens": strata_tokens[key],
                    "avg_tokens_per_row": avg_tok,
                    "estimated_rows": int(est_rows),
                    "estimated_tokens": int(avg_tok * est_rows),
                }
        else:
            for key, s_rows in strata_rows.items():
                avg_tok = strata_tokens[key] / s_rows if s_rows else 0.0
                strata_estimates[key] = {
                    "sample_rows": s_rows,
                    "sample_tokens": strata_tokens[key],
                    "avg_tokens_per_row": avg_tok,
                }

    return {
        "source_key": plan.source_key,
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "kept_rate": kept_rate,
        "total_rows": total_rows,
        "sample_rows": rows_kept,
        "token_sum": token_sum,
        "avg_tokens_per_kept": avg_tokens_per_kept,
        "avg_tokens_per_seen": avg_tokens_per_seen,
        "estimated_kept_rows": est_kept_rows,
        "estimated_total_tokens": est_total_tokens,
        "stratify_fields": plan.stratify_by or None,
        "stratify": strata_estimates or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate token counts for HF datasets")
    parser.add_argument("--config", required=True, help="Path to sources config (YAML/JSON)")
    parser.add_argument("--output", help="Override output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    encoder_name = cfg.get("token_estimator", "tiktoken")
    encoder = load_encoder(encoder_name)

    plans = build_sources(cfg)
    out_path = args.output or cfg.get("stats_path", "data/token_estimates.json")

    results = {"sources": {}, "total": {}}
    total_est = 0
    total_sample_tokens = 0
    for plan in plans:
        logger.info("Estimating tokens for %s", plan.source_key)
        stats = estimate_source(plan, encoder)
        results["sources"][plan.source_key] = stats
        if stats.get("estimated_total_tokens") is not None:
            total_est += stats["estimated_total_tokens"] or 0
        total_sample_tokens += stats.get("token_sum", 0) or 0

    results["total"] = {
        "estimated_total_tokens": total_est or None,
        "sample_tokens": total_sample_tokens,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Wrote token estimates to %s", out_path)


if __name__ == "__main__":
    main()
