#!/usr/bin/env python3
"""
Estimate SFT token budget from CustomJSON/ShareGPT-style JSONL files.

Accepted line formats:
1) CustomJSON line: [{"role":"user","content":"..."}, ...]
2) Wrapped line: {"messages":[{"role":"user","content":"..."}, ...]}

The script uses nanochat's tokenizer.render_conversation(...) for realistic
training token accounting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from nanochat.tokenizer import RustBPETokenizer, get_tokenizer


ROLE_MAP = {
    "user": "user",
    "assistant": "assistant",
    "system": "system",
    "human": "user",
    "gpt": "assistant",
    "model": "assistant",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate token budget from SFT JSONL")
    p.add_argument("--input", required=True, help="Path to JSONL (customjson/sharegpt-wrapped)")
    p.add_argument(
        "--tokenizer-dir",
        default="",
        help="Optional tokenizer directory containing tokenizer.pkl/token_bytes.pt "
        "(otherwise resolved from NANOCHAT_BASE_DIR)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Optional row cap for quick estimate (0=all)")
    p.add_argument("--progress-every", type=int, default=50000, help="Progress print frequency")
    return p.parse_args()


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def normalize_messages(raw_messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role_raw = str(msg.get("role", msg.get("from", ""))).strip().lower()
        role = ROLE_MAP.get(role_raw)
        if role is None:
            continue
        content = _extract_text(msg.get("content", msg.get("value", ""))).strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def parse_line(line: str) -> List[Dict[str, str]]:
    obj = json.loads(line)
    if isinstance(obj, list):
        raw_messages = obj
    elif isinstance(obj, dict):
        raw_messages = obj.get("messages", obj.get("conversations", []))
    else:
        raw_messages = []
    if not isinstance(raw_messages, list):
        return []
    return normalize_messages(raw_messages)


def main() -> None:
    args = parse_args()
    if args.tokenizer_dir:
        tok = RustBPETokenizer.from_directory(args.tokenizer_dir)
    else:
        try:
            tok = get_tokenizer()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}\nSet NANOCHAT_BASE_DIR to your artifacts dir "
                "(or pass --tokenizer-dir explicitly)."
            ) from exc
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = 0
    kept = 0
    total_tokens = 0
    loss_tokens = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            if args.max_rows > 0 and rows > args.max_rows:
                break

            try:
                messages = parse_line(line)
            except Exception:
                continue
            if len(messages) < 2:
                continue
            try:
                ids, loss_mask = tok.render_conversation({"messages": messages})
            except Exception:
                continue

            kept += 1
            total_tokens += len(ids)
            loss_tokens += int(sum(loss_mask))

            if args.progress_every > 0 and kept > 0 and kept % args.progress_every == 0:
                print(
                    f"progress kept={kept:,} total_tokens={total_tokens:,} "
                    f"loss_tokens={loss_tokens:,}"
                )

    avg_total = (total_tokens / kept) if kept else 0.0
    avg_loss = (loss_tokens / kept) if kept else 0.0

    print("\nSFT token estimate")
    print(f"  file:         {path}")
    print(f"  rows_seen:    {rows:,}")
    print(f"  rows_kept:    {kept:,}")
    print(f"  total_tokens: {total_tokens:,}")
    print(f"  loss_tokens:  {loss_tokens:,}")
    print(f"  avg_tokens_per_conversation: {avg_total:.2f}")
    print(f"  avg_loss_tokens_per_conversation: {avg_loss:.2f}")


if __name__ == "__main__":
    main()
