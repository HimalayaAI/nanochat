#!/usr/bin/env python3
"""
Prepare a mixed SFT corpus from multiple Hugging Face datasets.

Outputs:
1) ShareGPT-style JSONL:
   {"source":"repo", "conversations":[{"from":"human","value":"..."}, ...]}
2) nanochat CustomJSON JSONL:
   [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]

The CustomJSON output is directly consumable by tasks/customjson.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, get_token, login


ROLE_MAP_TO_SHAREGPT = {
    "system": "system",
    "user": "human",
    "human": "human",
    "prompt": "human",
    "assistant": "gpt",
    "gpt": "gpt",
    "model": "gpt",
}

SHAREGPT_TO_CUSTOM = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
}

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F\uA8E0-\uA8FF\U00011B00-\U00011B5F]")


@dataclass
class SourceStats:
    rows_seen: int = 0
    rows_kept: int = 0
    rows_dropped: int = 0
    dedup_dropped: int = 0
    approx_total_chars: int = 0
    approx_total_messages: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed SFT corpus from HF datasets")
    parser.add_argument(
        "--config",
        default="train_corpus/configs/sft_sources.yml",
        help="YAML config with source definitions",
    )
    parser.add_argument(
        "--sharegpt-out",
        default="data/sft_mix/sharegpt_mixture.jsonl",
        help="Output ShareGPT JSONL path",
    )
    parser.add_argument(
        "--customjson-out",
        default="data/sft_mix/customjson_mixture.jsonl",
        help="Output CustomJSON JSONL path",
    )
    parser.add_argument(
        "--stats-out",
        default="data/sft_mix/stats.json",
        help="Output JSON stats path",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=0,
        help="Optional cap on number of sources to process (0 = all)",
    )
    parser.add_argument(
        "--max-conversations-per-source",
        type=int,
        default=0,
        help="Optional hard cap per source, overriding config (0 = use config values)",
    )
    parser.add_argument(
        "--upload-repo",
        default=None,
        help="Optional HF dataset repo id to upload outputs (org/name)",
    )
    parser.add_argument(
        "--upload-prefix",
        default="data",
        help="Path prefix inside the uploaded HF dataset repo",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload mixed SFT corpus (sharegpt + customjson)",
        help="Commit message for HF upload",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for iterable shuffle when enabled in config",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, dict):
        if "text" in content:
            return extract_text(content.get("text"))
        if "content" in content:
            return extract_text(content.get("content"))
        if "value" in content:
            return extract_text(content.get("value"))
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            txt = extract_text(item)
            if txt:
                parts.append(txt)
        return "\n".join(parts).strip()
    return ""


def normalize_raw_messages(raw_messages: Sequence[Any]) -> Optional[List[Dict[str, str]]]:
    messages: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role_raw = msg.get("from", msg.get("role"))
        if role_raw is None:
            continue
        role = ROLE_MAP_TO_SHAREGPT.get(str(role_raw).strip().lower())
        if role is None:
            continue
        content_raw = msg.get("value", msg.get("content"))
        text = extract_text(content_raw)
        if not text:
            continue
        messages.append({"from": role, "value": text})
    if len(messages) < 2:
        return None
    return messages


def conv_from_sharegpt(row: Dict[str, Any], field: str) -> Optional[List[Dict[str, str]]]:
    raw = row.get(field)
    if not isinstance(raw, list):
        return None
    return normalize_raw_messages(raw)


def conv_from_alpaca(
    row: Dict[str, Any],
    instruction_field: str,
    input_field: str,
    output_field: str,
) -> Optional[List[Dict[str, str]]]:
    instruction = extract_text(row.get(instruction_field, ""))
    output = extract_text(row.get(output_field, ""))
    if not instruction or not output:
        return None
    extra_input = extract_text(row.get(input_field, ""))
    if extra_input:
        prompt = f"{instruction}\n\nInput:\n{extra_input}"
    else:
        prompt = instruction
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": output},
    ]


def conv_from_openorca(
    row: Dict[str, Any],
    system_field: str,
    question_field: str,
    response_field: str,
) -> Optional[List[Dict[str, str]]]:
    question = extract_text(row.get(question_field, ""))
    response = extract_text(row.get(response_field, ""))
    if not question or not response:
        return None
    system_prompt = extract_text(row.get(system_field, ""))
    out: List[Dict[str, str]] = []
    if system_prompt:
        out.append({"from": "system", "value": system_prompt})
    out.extend(
        [
            {"from": "human", "value": question},
            {"from": "gpt", "value": response},
        ]
    )
    return out


def conv_from_pair(row: Dict[str, Any], prompt_field: str, response_field: str) -> Optional[List[Dict[str, str]]]:
    prompt = extract_text(row.get(prompt_field, ""))
    response = extract_text(row.get(response_field, ""))
    if not prompt or not response:
        return None
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": response},
    ]


def conv_from_pair_with_system(
    row: Dict[str, Any],
    prompt_field: str,
    response_field: str,
    system_field: Optional[str] = None,
    prepend_prompt_fields: Optional[Sequence[str]] = None,
) -> Optional[List[Dict[str, str]]]:
    prompt_core = extract_text(row.get(prompt_field, ""))
    response = extract_text(row.get(response_field, ""))
    if not prompt_core or not response:
        return None

    prompt_parts: List[str] = []
    for fld in prepend_prompt_fields or []:
        txt = extract_text(row.get(fld, ""))
        if txt:
            prompt_parts.append(txt)
    prompt_parts.append(prompt_core)
    prompt = "\n\n".join(prompt_parts)

    out: List[Dict[str, str]] = []
    if system_field:
        system_text = extract_text(row.get(system_field, ""))
        if system_text:
            out.append({"from": "system", "value": system_text})
    out.extend(
        [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response},
        ]
    )
    return out


def conv_from_interactions(
    row: Dict[str, Any],
    field: str = "interactions",
) -> Optional[List[Dict[str, str]]]:
    interactions = row.get(field)
    if not isinstance(interactions, list) or not interactions:
        return None

    out: List[Dict[str, str]] = []
    for item in interactions:
        user_text = ""
        assistant_text = ""
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text = extract_text(item[0])
            assistant_text = extract_text(item[1])
        elif isinstance(item, dict):
            user_text = extract_text(item.get("user", item.get("prompt", item.get("query", ""))))
            assistant_text = extract_text(
                item.get("assistant", item.get("response", item.get("answer", "")))
            )
        if not user_text or not assistant_text:
            continue
        out.append({"from": "human", "value": user_text})
        out.append({"from": "gpt", "value": assistant_text})

    if len(out) < 2:
        return None
    return out


def autodetect_conversation(row: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    # 1) Standard message arrays
    for key in ("conversations", "messages"):
        conv = conv_from_sharegpt(row, key)
        if conv:
            return conv

    # 2) Alpaca-style
    if "instruction" in row and "output" in row:
        return conv_from_alpaca(row, "instruction", "input", "output")

    # 3) OpenOrca-style
    if "question" in row and "response" in row:
        return conv_from_openorca(row, "system_prompt", "question", "response")

    # 4) Generic pair patterns
    pair_candidates = [
        ("prompt", "response"),
        ("prompt", "answer"),
        ("question", "answer"),
        ("input", "output"),
        ("query", "response"),
    ]
    for p_field, r_field in pair_candidates:
        if p_field in row and r_field in row:
            conv = conv_from_pair(row, p_field, r_field)
            if conv:
                return conv

    return None


def to_customjson_messages(sharegpt_conv: Sequence[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    mapped: List[Dict[str, str]] = []
    for msg in sharegpt_conv:
        role = SHAREGPT_TO_CUSTOM.get(str(msg.get("from", "")).strip().lower())
        text = extract_text(msg.get("value", ""))
        if role is None or not text:
            continue
        mapped.append({"role": role, "content": text})

    if len(mapped) < 2:
        return None

    # Merge leading systems into first user turn.
    sys_chunks: List[str] = []
    while mapped and mapped[0]["role"] == "system":
        sys_chunks.append(mapped.pop(0)["content"])
    if not mapped or mapped[0]["role"] != "user":
        return None
    if sys_chunks:
        mapped[0]["content"] = "\n\n".join(sys_chunks + [mapped[0]["content"]])

    # Strict alternation user/assistant/user/assistant...
    alternated: List[Dict[str, str]] = []
    expected_role = "user"
    for msg in mapped:
        if msg["role"] != expected_role:
            return None
        alternated.append(msg)
        expected_role = "assistant" if expected_role == "user" else "user"

    # End on assistant turn (so it teaches completion).
    if alternated[-1]["role"] != "assistant":
        alternated = alternated[:-1]
        if len(alternated) < 2 or alternated[-1]["role"] != "assistant":
            return None

    return alternated


def canonical_hash(custom_messages: Sequence[Dict[str, str]]) -> str:
    payload = json.dumps(custom_messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def row_matches_filters(row: Dict[str, Any], src_cfg: Dict[str, Any]) -> bool:
    must_equal: Dict[str, Any] = src_cfg.get("must_equal", {}) or {}
    for key, wanted in must_equal.items():
        value = row.get(key)
        if isinstance(wanted, list):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


def sharegpt_matches_script_filter(sharegpt_conv: Sequence[Dict[str, str]], src_cfg: Dict[str, Any]) -> bool:
    # Optional text-level guardrail for script-specific mixtures.
    if not bool(src_cfg.get("require_devanagari", False)):
        return True

    min_chars = int(src_cfg.get("min_devanagari_chars", 16))
    min_ratio = float(src_cfg.get("min_devanagari_ratio", 0.35))
    combined = "\n".join(extract_text(m.get("value", "")) for m in sharegpt_conv).strip()
    if not combined:
        return False

    deva_count = len(DEVANAGARI_RE.findall(combined))
    non_ws_count = sum(1 for ch in combined if not ch.isspace())
    ratio = deva_count / max(1, non_ws_count)
    return deva_count >= min_chars and ratio >= min_ratio


def row_to_sharegpt(row: Dict[str, Any], src_cfg: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    adapter = str(src_cfg.get("adapter", "auto")).strip().lower()
    if adapter == "sharegpt_conversations":
        field = src_cfg.get("conversation_field", "conversations")
        return conv_from_sharegpt(row, field)
    if adapter == "sharegpt_messages":
        field = src_cfg.get("message_field", "messages")
        return conv_from_sharegpt(row, field)
    if adapter == "alpaca":
        return conv_from_alpaca(
            row,
            src_cfg.get("instruction_field", "instruction"),
            src_cfg.get("input_field", "input"),
            src_cfg.get("output_field", "output"),
        )
    if adapter == "openorca":
        return conv_from_openorca(
            row,
            src_cfg.get("system_field", "system_prompt"),
            src_cfg.get("question_field", "question"),
            src_cfg.get("response_field", "response"),
        )
    if adapter == "qa_pair":
        prepend_fields = src_cfg.get("prepend_prompt_fields", []) or []
        if prepend_fields and not isinstance(prepend_fields, list):
            raise ValueError(f"prepend_prompt_fields must be a list for source={src_cfg.get('id')}")
        return conv_from_pair_with_system(
            row,
            src_cfg.get("prompt_field", "prompt"),
            src_cfg.get("response_field", "response"),
            src_cfg.get("system_field"),
            prepend_fields,
        )
    if adapter == "interaction_pairs":
        return conv_from_interactions(row, src_cfg.get("interaction_field", "interactions"))
    if adapter == "auto":
        return autodetect_conversation(row)
    raise ValueError(f"Unknown adapter: {adapter}")


def iter_dataset_rows(src_cfg: Dict[str, Any], seed: int) -> Iterable[Dict[str, Any]]:
    repo_id = src_cfg["id"]
    split = src_cfg.get("split", "train")
    cfg_name = src_cfg.get("config")
    streaming = bool(src_cfg.get("streaming", True))
    shuffle_buffer = int(src_cfg.get("shuffle_buffer", 0))

    load_kwargs = {"split": split, "streaming": streaming}
    if cfg_name:
        ds = load_dataset(repo_id, name=cfg_name, **load_kwargs)
    else:
        ds = load_dataset(repo_id, **load_kwargs)

    if streaming and shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    yield from ds


def upload_outputs(
    repo_id: str,
    files: Sequence[Tuple[Path, str]],
    commit_message: str,
) -> None:
    token = get_token() or os.getenv("HF_TOKEN")
    if not token:
        login()
        token = get_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    for local_path, path_in_repo in files:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    sources: List[Dict[str, Any]] = list(cfg.get("sources", []))
    if not sources:
        raise ValueError(f"No sources found in config: {args.config}")
    if args.max_sources > 0:
        sources = sources[: args.max_sources]

    sharegpt_out = Path(args.sharegpt_out)
    customjson_out = Path(args.customjson_out)
    stats_out = Path(args.stats_out)
    sharegpt_out.parent.mkdir(parents=True, exist_ok=True)
    customjson_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    global_stats: Dict[str, Any] = {
        "config": args.config,
        "sharegpt_out": str(sharegpt_out),
        "customjson_out": str(customjson_out),
        "sources": {},
        "totals": {
            "rows_seen": 0,
            "rows_kept": 0,
            "rows_dropped": 0,
            "dedup_dropped": 0,
            "approx_total_chars": 0,
            "approx_total_messages": 0,
            "unique_conversations": 0,
        },
    }

    with sharegpt_out.open("w", encoding="utf-8") as sharegpt_f, customjson_out.open(
        "w", encoding="utf-8"
    ) as custom_f:
        for src_idx, src_cfg in enumerate(sources):
            if "id" not in src_cfg:
                raise KeyError(f"Source index {src_idx} missing required key: id")
            src_id = src_cfg["id"]
            max_conversations = (
                int(args.max_conversations_per_source)
                if args.max_conversations_per_source > 0
                else int(src_cfg.get("max_conversations", 0))
            )
            src_stats = SourceStats()
            print(f"[{src_idx + 1}/{len(sources)}] Processing {src_id} ...")

            for row in iter_dataset_rows(src_cfg, seed=args.seed):
                if max_conversations > 0 and src_stats.rows_kept >= max_conversations:
                    break
                src_stats.rows_seen += 1

                if not row_matches_filters(row, src_cfg):
                    src_stats.rows_dropped += 1
                    continue

                sharegpt_conv = row_to_sharegpt(row, src_cfg)
                if sharegpt_conv is None:
                    src_stats.rows_dropped += 1
                    continue
                if not sharegpt_matches_script_filter(sharegpt_conv, src_cfg):
                    src_stats.rows_dropped += 1
                    continue

                custom_conv = to_customjson_messages(sharegpt_conv)
                if custom_conv is None:
                    src_stats.rows_dropped += 1
                    continue

                conv_hash = canonical_hash(custom_conv)
                if conv_hash in seen_hashes:
                    src_stats.dedup_dropped += 1
                    continue
                seen_hashes.add(conv_hash)

                conv_id = str(row.get("id", f"{src_id}:{src_stats.rows_seen}"))
                sharegpt_obj = {
                    "id": conv_id,
                    "source": src_id,
                    "conversations": sharegpt_conv,
                }
                sharegpt_f.write(json.dumps(sharegpt_obj, ensure_ascii=False) + "\n")
                custom_f.write(json.dumps(custom_conv, ensure_ascii=False) + "\n")
                src_stats.rows_kept += 1

                src_stats.approx_total_messages += len(custom_conv)
                src_stats.approx_total_chars += sum(len(m["content"]) for m in custom_conv)

            global_stats["sources"][src_id] = {
                "rows_seen": src_stats.rows_seen,
                "rows_kept": src_stats.rows_kept,
                "rows_dropped": src_stats.rows_dropped,
                "dedup_dropped": src_stats.dedup_dropped,
                "approx_total_chars": src_stats.approx_total_chars,
                "approx_total_messages": src_stats.approx_total_messages,
            }
            print(
                f"  kept={src_stats.rows_kept:,} dropped={src_stats.rows_dropped:,} "
                f"dedup={src_stats.dedup_dropped:,}"
            )

            global_stats["totals"]["rows_seen"] += src_stats.rows_seen
            global_stats["totals"]["rows_kept"] += src_stats.rows_kept
            global_stats["totals"]["rows_dropped"] += src_stats.rows_dropped
            global_stats["totals"]["dedup_dropped"] += src_stats.dedup_dropped
            global_stats["totals"]["approx_total_chars"] += src_stats.approx_total_chars
            global_stats["totals"]["approx_total_messages"] += src_stats.approx_total_messages

    global_stats["totals"]["unique_conversations"] = len(seen_hashes)

    stats_out.write_text(json.dumps(global_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote ShareGPT:   {sharegpt_out}")
    print(f"Wrote CustomJSON: {customjson_out}")
    print(f"Wrote stats:      {stats_out}")

    if args.upload_repo:
        prefix = args.upload_prefix.strip("/")
        files_to_upload = [
            (sharegpt_out, f"{prefix}/{sharegpt_out.name}"),
            (customjson_out, f"{prefix}/{customjson_out.name}"),
            (stats_out, f"{prefix}/{stats_out.name}"),
        ]
        upload_outputs(
            repo_id=args.upload_repo,
            files=files_to_upload,
            commit_message=args.commit_message,
        )
        print(f"Uploaded outputs to: https://huggingface.co/datasets/{args.upload_repo}")


if __name__ == "__main__":
    main()
