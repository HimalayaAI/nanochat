#!/usr/bin/env python3
"""
Infer SFT source schemas from streaming rows and suggest adapter config stubs.

This is intended as a preflight tool before adding/updating SFT source lists.
"""

from __future__ import annotations

import argparse
import json
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset, load_dataset_builder
from huggingface_hub import HfApi


TIMEOUT_EXCEPTION = TimeoutError("timed out while probing dataset row")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer SFT schema from streaming datasets")
    p.add_argument(
        "--config",
        default="train_corpus/configs/sft_sources_deva.yml",
        help="YAML source config path",
    )
    p.add_argument("--max-sources", type=int, default=0, help="Only inspect first N sources (0=all)")
    p.add_argument("--timeout-seconds", type=int, default=45, help="Per-source probe timeout")
    p.add_argument("--output-json", default="", help="Optional report output path")
    return p.parse_args()


@contextmanager
def time_limit(seconds: int):
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # type: ignore[no-untyped-def]
        raise TIMEOUT_EXCEPTION

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_eval_dataset_ids() -> set[str]:
    ids = {"Cognitive-Lab/Aya_Indic_Eval"}
    path = Path("train_corpus/configs/eval_sources.yml")
    if not path.exists():
        return ids
    cfg = load_yaml(str(path))
    for lang in cfg.get("languages", []) or []:
        for src in lang.get("sources", []) or []:
            ds_id = src.get("id")
            if ds_id:
                ids.add(str(ds_id))
    return ids


def infer_adapter_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(row.keys())
    if "interactions" in keys and isinstance(row.get("interactions"), list):
        return {"adapter": "interaction_pairs", "interaction_field": "interactions"}

    if "conversations" in keys and isinstance(row.get("conversations"), list):
        return {"adapter": "sharegpt_conversations", "conversation_field": "conversations"}

    if "messages" in keys and isinstance(row.get("messages"), list):
        return {"adapter": "sharegpt_messages", "message_field": "messages"}

    if "instruction" in keys and "output" in keys:
        return {"adapter": "alpaca"}

    if "question" in keys and "response" in keys:
        return {"adapter": "openorca"}

    if "query" in keys and "pos" in keys:
        out = {"adapter": "qa_pair", "prompt_field": "query", "response_field": "pos"}
        if "task" in keys:
            out["system_field"] = "task"
        return out

    if "prompt" in keys and ("response" in keys or "answer" in keys):
        return {
            "adapter": "qa_pair",
            "prompt_field": "prompt",
            "response_field": "response" if "response" in keys else "answer",
        }

    return {"adapter": "auto"}


def infer_adapter_from_field_names(field_names: Iterable[str]) -> Dict[str, Any]:
    keys = set(field_names)
    if "interactions" in keys:
        return {"adapter": "interaction_pairs", "interaction_field": "interactions"}
    if "conversations" in keys:
        return {"adapter": "sharegpt_conversations", "conversation_field": "conversations"}
    if "messages" in keys:
        return {"adapter": "sharegpt_messages", "message_field": "messages"}
    if "instruction" in keys and "output" in keys:
        return {"adapter": "alpaca"}
    if "question" in keys and "response" in keys:
        return {"adapter": "openorca"}
    if "query" in keys and "pos" in keys:
        out = {"adapter": "qa_pair", "prompt_field": "query", "response_field": "pos"}
        if "task" in keys:
            out["system_field"] = "task"
        return out
    if "prompt" in keys and ("response" in keys or "answer" in keys):
        return {
            "adapter": "qa_pair",
            "prompt_field": "prompt",
            "response_field": "response" if "response" in keys else "answer",
        }
    return {"adapter": "auto"}


def read_stream_row(
    ds_id: str,
    cfg_name: Optional[str],
    split: str,
    data_files: Any,
    timeout_seconds: int,
) -> Dict[str, Any]:
    load_kwargs = {"split": split, "streaming": True}
    if data_files is not None:
        load_kwargs["data_files"] = data_files
    with time_limit(timeout_seconds):
        if cfg_name:
            ds = load_dataset(ds_id, name=cfg_name, **load_kwargs)
        else:
            ds = load_dataset(ds_id, **load_kwargs)
        return next(iter(ds))


def describe_value_type(value: Any) -> str:
    t = type(value).__name__
    if isinstance(value, list):
        if not value:
            return "list[empty]"
        return f"list[{type(value[0]).__name__}]"
    return t


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    sources = list(cfg.get("sources", []) or [])
    if args.max_sources > 0:
        sources = sources[: args.max_sources]

    if not sources:
        raise ValueError(f"No sources found in config: {args.config}")

    eval_ids = get_eval_dataset_ids()
    api = HfApi()
    report: List[Dict[str, Any]] = []

    for idx, src in enumerate(sources, start=1):
        ds_id = src.get("id")
        if not ds_id:
            continue
        cfg_name = src.get("config")
        split = str(src.get("split", "train"))
        data_files = src.get("data_files")
        print(f"\n[{idx}/{len(sources)}] {ds_id} cfg={cfg_name or '-'} split={split}")

        item: Dict[str, Any] = {
            "id": ds_id,
            "config": cfg_name,
            "split": split,
            "data_files": data_files,
            "in_eval_ids": ds_id in eval_ids,
        }

        try:
            info = api.dataset_info(ds_id)
            tags = [t.lower() for t in (info.tags or [])]
            item["benchmark_tag"] = any("benchmark" in t for t in tags)
            item["likes"] = getattr(info, "likes", None)
            item["downloads"] = getattr(info, "downloads", None)
            print(
                f"  benchmark_tag={item['benchmark_tag']} "
                f"in_eval_ids={item['in_eval_ids']} "
                f"likes={item['likes']} downloads={item['downloads']}"
            )
        except Exception as e:
            item["dataset_info_error"] = f"{type(e).__name__}: {e}"
            print(f"  dataset_info_error={item['dataset_info_error']}")

        try:
            cfgs = get_dataset_config_names(ds_id)
            item["available_configs"] = cfgs
            print(f"  available_configs={len(cfgs)}")
        except Exception as e:
            item["config_list_error"] = f"{type(e).__name__}: {e}"
            print(f"  config_list_error={item['config_list_error']}")

        try:
            if cfg_name:
                splits = get_dataset_split_names(ds_id, cfg_name)
            else:
                splits = get_dataset_split_names(ds_id)
            item["available_splits"] = splits
            print(f"  available_splits={splits}")
        except Exception as e:
            item["split_list_error"] = f"{type(e).__name__}: {e}"
            print(f"  split_list_error={item['split_list_error']}")

        try:
            row = read_stream_row(ds_id, cfg_name, split, data_files, args.timeout_seconds)
            keys = list(row.keys())
            key_types = {k: describe_value_type(v) for k, v in row.items()}
            item["stream_keys"] = keys
            item["stream_key_types"] = key_types
            item["suggested_adapter"] = infer_adapter_from_row(row)
            print(f"  stream_keys={keys}")
            print(f"  suggested_adapter={item['suggested_adapter']}")
        except Exception as e:
            item["stream_probe_error"] = f"{type(e).__name__}: {e}"
            print(f"  stream_probe_error={item['stream_probe_error']}")
            try:
                builder = load_dataset_builder(ds_id, name=cfg_name) if cfg_name else load_dataset_builder(ds_id)
                feature_names = list((builder.info.features or {}).keys())
                item["fallback_feature_names"] = feature_names
                item["suggested_adapter"] = infer_adapter_from_field_names(feature_names)
                print(f"  fallback_feature_names={feature_names}")
                print(f"  suggested_adapter={item['suggested_adapter']}")
            except Exception as e2:
                item["feature_fallback_error"] = f"{type(e2).__name__}: {e2}"
                print(f"  feature_fallback_error={item['feature_fallback_error']}")

        report.append(item)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote report: {out}")


if __name__ == "__main__":
    main()
