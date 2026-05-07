#!/usr/bin/env python3
"""
Quick generation smoke test for nanochat checkpoints.

This is useful as a pre-SFT gate to verify base checkpoints are generating
reasonable text before starting SFT/rl stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROMPTS = [
    "नेपालको राजधानी के हो?",
    "नेपालको हिमाल किन महत्त्वपूर्ण छ?",
    "Write a short paragraph about machine learning.",
    "What is 17 * 19? Show the steps briefly.",
    "Write a Python function to reverse a string.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test nanochat checkpoint generation")
    parser.add_argument("--source", choices=["base", "sft", "rl"], default="base")
    parser.add_argument("--model-tag", default=None, help="Checkpoint model tag")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step")
    parser.add_argument("--prompt-style", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--prompts-file", default=None, help="Optional .txt/.json/.jsonl file")
    parser.add_argument("--num-prompts", type=int, default=0, help="Limit number of prompts (0=all)")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-type", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--expect-min-new-tokens", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_prompts(path: str | None) -> List[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".txt":
        prompts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON list in {p}")
        prompts = [str(x).strip() for x in obj if str(x).strip()]
    elif suffix == ".jsonl":
        prompts = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("prompt", obj)).strip() if isinstance(obj, dict) else str(obj).strip()
            if text:
                prompts.append(text)
    else:
        raise ValueError(f"Unsupported prompts extension for {p}")
    if not prompts:
        raise ValueError(f"No prompts found in {p}")
    return prompts


def choose_device_type(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_prompt_style(arg: str, source: str) -> str:
    if arg != "auto":
        return arg
    return "chat" if source in {"sft", "rl"} else "plain"


def build_prompt_tokens(tokenizer, prompt: str, prompt_style: str) -> List[int]:
    bos = tokenizer.get_bos_token_id()
    if prompt_style == "plain":
        return tokenizer.encode(prompt, prepend=bos)
    if prompt_style == "chat":
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        ids = [bos, user_start]
        ids.extend(tokenizer.encode(prompt))
        ids.extend([user_end, assistant_start])
        return ids
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def strip_special_tokens(text: str) -> str:
    for tok in [
        "<|bos|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|python_start|>",
        "<|python_end|>",
        "<|output_start|>",
        "<|output_end|>",
    ]:
        text = text.replace(tok, "")
    return text.strip()


def main() -> None:
    args = parse_args()

    from nanochat.checkpoint_manager import load_model
    from nanochat.common import compute_init
    from nanochat.engine import Engine

    prompts = load_prompts(args.prompts_file)
    if args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]
    if not prompts:
        raise ValueError("No prompts to run")

    device_type = choose_device_type(args.device_type)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if ddp_world_size != 1:
        raise RuntimeError("Smoke test expects single-process execution")

    prompt_style = resolve_prompt_style(args.prompt_style, args.source)
    model, tokenizer, meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)

    print(f"Loaded checkpoint source={args.source} tag={args.model_tag or '<auto>'} step={meta.get('step')}")
    print(f"Prompt style: {prompt_style}")
    print(f"Running {len(prompts)} prompt(s)")

    failures = 0
    results: List[Dict[str, Any]] = []
    top_k = args.top_k if args.top_k > 0 else None

    for i, prompt in enumerate(prompts, start=1):
        prompt_tokens = build_prompt_tokens(tokenizer, prompt, prompt_style)
        output_ids, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed + i,
        )
        gen_ids = output_ids[0][len(prompt_tokens) :]
        new_tokens = len(gen_ids)
        text = strip_special_tokens(tokenizer.decode(gen_ids))
        passed = new_tokens >= args.expect_min_new_tokens
        if not passed:
            failures += 1
        results.append(
            {
                "index": i,
                "prompt": prompt,
                "new_tokens": new_tokens,
                "passed": passed,
                "output": text,
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] Prompt {i}/{len(prompts)} | new_tokens={new_tokens}")
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")

    print(f"\nSummary: {len(prompts)-failures}/{len(prompts)} prompts passed")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": args.source,
            "model_tag": args.model_tag,
            "step": meta.get("step"),
            "prompt_style": prompt_style,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "results": results,
            "passed": failures == 0,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report to: {out_path}")

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

