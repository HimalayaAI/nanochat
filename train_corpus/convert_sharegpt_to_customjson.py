#!/usr/bin/env python3
"""
Convert ShareGPT-style HF datasets to nanochat CustomJSON format.

Input (typical ShareGPT):
  conversations: list[{from: "system|human|gpt", value: "..."}]
or:
  messages: list[{role: "system|user|assistant", content: "..."}]

Output (CustomJSON):
  JSONL where each line is:
    [{"role":"user","content":"..."},{"role":"assistant","content":"..."}...]

This output is directly consumable by tasks/customjson.py in nanochat.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


ROLE_MAP = {
    "system": "system",
    "user": "user",
    "human": "user",
    "prompt": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "model": "assistant",
}


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (int, float)):
        return str(content)
    if isinstance(content, dict):
        if "text" in content:
            return _extract_text(content.get("text"))
        if "content" in content:
            return _extract_text(content.get("content"))
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            txt = _extract_text(part)
            if txt:
                parts.append(txt)
        return "\n".join(parts).strip()
    return ""


def _normalize_message(msg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    role_raw = msg.get("role")
    if role_raw is None:
        role_raw = msg.get("from")
    if role_raw is None:
        return None
    role = ROLE_MAP.get(str(role_raw).strip().lower())
    if role is None:
        return None

    content_raw = msg.get("content")
    if content_raw is None:
        content_raw = msg.get("value")
    text = _extract_text(content_raw)
    if not text:
        return None
    return {"role": role, "content": text}


def _get_raw_messages(row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    for key in ("conversations", "messages"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    return None


def normalize_conversation(
    row: Dict[str, Any],
    *,
    merge_leading_system: bool,
    require_final_assistant: bool,
) -> Optional[List[Dict[str, str]]]:
    raw_messages = _get_raw_messages(row)
    if not raw_messages:
        return None

    messages: List[Dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            continue
        nm = _normalize_message(raw)
        if nm:
            messages.append(nm)
    if len(messages) < 2:
        return None

    # Merge leading system messages into first user message, because CustomJSON
    # expects strict user/assistant alternation starting from index 0.
    if merge_leading_system:
        system_chunks: List[str] = []
        while messages and messages[0]["role"] == "system":
            system_chunks.append(messages.pop(0)["content"])
        if not messages or messages[0]["role"] != "user":
            return None
        if system_chunks:
            messages[0]["content"] = "\n\n".join(system_chunks + [messages[0]["content"]])

    # Validate strict alternation user, assistant, user, assistant...
    expected = "user"
    alternated: List[Dict[str, str]] = []
    for m in messages:
        if m["role"] != expected:
            return None
        alternated.append(m)
        expected = "assistant" if expected == "user" else "user"

    # Keep only samples that teach assistant behavior.
    if not any(m["role"] == "assistant" for m in alternated):
        return None

    if require_final_assistant and alternated and alternated[-1]["role"] != "assistant":
        alternated = alternated[:-1]
        if len(alternated) < 2 or alternated[-1]["role"] != "assistant":
            return None

    return alternated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT-style HF dataset to nanochat CustomJSON JSONL"
    )
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. himalaya-ai/nepali-sft-dataset")
    parser.add_argument("--split", default="train", help="HF split (default: train)")
    parser.add_argument("--config", help="HF dataset config name (optional)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-rows", type=int, help="Optional max input rows to scan")
    parser.add_argument(
        "--no-merge-leading-system",
        action="store_true",
        help="Disable merging leading system messages into first user message",
    )
    parser.add_argument(
        "--allow-final-user",
        action="store_true",
        help="Allow final user turn (default trims/drops so final turn is assistant)",
    )
    args = parser.parse_args()

    load_kwargs = {"split": args.split, "streaming": True}
    if args.config:
        ds = load_dataset(args.repo_id, name=args.config, **load_kwargs)
    else:
        ds = load_dataset(args.repo_id, **load_kwargs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    rows_seen = 0
    rows_written = 0
    dropped = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for row in ds:
            rows_seen += 1
            convo = normalize_conversation(
                row,
                merge_leading_system=not args.no_merge_leading_system,
                require_final_assistant=not args.allow_final_user,
            )
            if convo is None:
                dropped += 1
            else:
                out.write(json.dumps(convo, ensure_ascii=False) + "\n")
                rows_written += 1

            if args.max_rows and rows_seen >= args.max_rows:
                break

    stats = {
        "repo_id": args.repo_id,
        "config": args.config,
        "split": args.split,
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_dropped": dropped,
        "drop_rate": (dropped / rows_seen) if rows_seen else 0.0,
        "output": args.output,
    }
    stats_path = args.output + ".stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(
        "Done.",
        f"seen={rows_seen:,}",
        f"written={rows_written:,}",
        f"dropped={dropped:,}",
        f"stats={stats_path}",
    )


if __name__ == "__main__":
    main()

