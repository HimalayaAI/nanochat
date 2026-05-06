#!/usr/bin/env python3
"""
Run a quick generation smoke test on a nanochat HF-exported model.

This is intended as a pre-SFT gate:
- loads tokenizer + model with trust_remote_code
- runs a small prompt suite
- verifies the model generates at least one new token per prompt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_PROMPTS = [
    "नेपालको राजधानी के हो?",
    "दुई वाक्यमा हिमालको महत्व बताऊ।",
    "Write a short paragraph about machine learning.",
    "What is 17 * 19? Show quick mental math.",
    "Write a Python function to compute Fibonacci numbers.",
]


SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test HF-exported nanochat model on several prompts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo-id", default=None, help="HF model repo id, e.g. himalaya-ai/himalayagpt-0.5b")
    group.add_argument("--local-dir", default=None, help="Local HF export directory")

    parser.add_argument("--revision", default="main", help="HF revision when using --repo-id")
    parser.add_argument("--prompts-file", default=None, help="Optional .txt/.json/.jsonl prompts file")
    parser.add_argument("--num-prompts", type=int, default=0, help="Limit number of prompts (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max generated tokens per prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device-type", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prompt-style", choices=["auto", "plain", "chat"], default="auto", help="Prompt formatting style")
    parser.add_argument("--expect-min-new-tokens", type=int, default=1, help="Fail prompt if fewer tokens are generated")
    parser.add_argument(
        "--expect-min-response-chars",
        type=int,
        default=1,
        help="Fail prompt if cleaned completion has fewer visible characters",
    )
    parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    return parser.parse_args()


def load_prompts(path: str | None) -> List[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {p}")
    if p.suffix.lower() == ".txt":
        prompts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {p}")
        return prompts
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {p}")
        prompts = [str(x).strip() for x in data if str(x).strip()]
        if not prompts:
            raise ValueError(f"No prompts found in {p}")
        return prompts
    if p.suffix.lower() == ".jsonl":
        prompts: List[str] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "prompt" in obj:
                text = str(obj["prompt"]).strip()
            else:
                text = str(obj).strip()
            if text:
                prompts.append(text)
        if not prompts:
            raise ValueError(f"No prompts found in {p}")
        return prompts
    raise ValueError(f"Unsupported prompts file extension for {p}; use .txt, .json, or .jsonl")


def model_load_kwargs(device_type: str) -> Dict[str, Any]:
    import torch

    use_cuda = torch.cuda.is_available() if device_type == "auto" else device_type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if use_cuda:
        return {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    return {"torch_dtype": torch.float32}


def choose_input_device(model) -> Any:
    # For single-GPU / CPU this is enough and avoids device-map complications.
    return next(model.parameters()).device


def infer_hf_context_window(model, tokenizer) -> int:
    candidates: List[int] = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for key in ("sequence_len", "max_position_embeddings", "n_positions", "max_seq_len"):
            val = getattr(cfg, key, None)
            if isinstance(val, int) and 0 < val < 1_000_000:
                candidates.append(int(val))
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
        candidates.append(int(tok_max))
    if not candidates:
        return 2048
    return min(candidates)


def strip_special_markers(text: str) -> str:
    out = text
    for tok in SPECIAL_TOKENS:
        out = out.replace(tok, "")
    return out.strip()


def _hf_special_id(tokenizer, token: str) -> int | None:
    # Prefer tokenizer's native special-id table when available. Some custom
    # tokenizers can report remapped ids via convert_tokens_to_ids() if the
    # special token is re-added on load.
    special_map = getattr(tokenizer, "_special_to_id", None)
    if isinstance(special_map, dict) and token in special_map:
        token_id = int(special_map[token])
        if token_id >= 0:
            return token_id

    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    if token_id == tokenizer.unk_token_id and token != tokenizer.unk_token:
        return None
    return int(token_id)


def resolve_prompt_style(prompt_style: str, tokenizer) -> str:
    if prompt_style != "auto":
        return prompt_style
    needed = ["<|user_start|>", "<|user_end|>", "<|assistant_start|>"]
    if all(_hf_special_id(tokenizer, t) is not None for t in needed):
        return "chat"
    return "plain"


def maybe_fallback_chat_to_plain(prompt_style: str, tokenizer, embedding_vocab_size: int) -> str:
    if prompt_style != "chat":
        return prompt_style
    for tok in ("<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>"):
        tid = _hf_special_id(tokenizer, tok)
        if tid is None:
            print(f"[warn] Missing special token id for {tok}; falling back to plain prompt style.")
            return "plain"
        if tid < 0 or tid >= embedding_vocab_size:
            print(
                f"[warn] Special token {tok} has id={tid} outside embedding range [0, {embedding_vocab_size - 1}]; "
                "falling back to plain prompt style."
            )
            return "plain"
    return prompt_style


def build_prompt_ids(tokenizer, prompt: str, prompt_style: str) -> List[int]:
    if prompt_style == "plain":
        return tokenizer(prompt, add_special_tokens=False)["input_ids"]

    if prompt_style == "chat":
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = _hf_special_id(tokenizer, "<|bos|>")
        user_start = _hf_special_id(tokenizer, "<|user_start|>")
        user_end = _hf_special_id(tokenizer, "<|user_end|>")
        assistant_start = _hf_special_id(tokenizer, "<|assistant_start|>")
        missing = [
            name
            for name, val in [
                ("<|bos|>", bos),
                ("<|user_start|>", user_start),
                ("<|user_end|>", user_end),
                ("<|assistant_start|>", assistant_start),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                "Chat prompt style requested, but tokenizer is missing special tokens: " + ", ".join(missing)
            )
        ids = [int(bos), int(user_start)]
        ids.extend(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        ids.extend([int(user_end), int(assistant_start)])
        return ids

    raise ValueError(f"Unknown prompt style: {prompt_style}")


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target = args.repo_id if args.repo_id is not None else args.local_dir
    assert target is not None

    prompts = load_prompts(args.prompts_file)
    if args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]
    if not prompts:
        raise ValueError("No prompts to run")

    load_kwargs = model_load_kwargs(args.device_type)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading tokenizer/model from: {target}")
    tok = AutoTokenizer.from_pretrained(
        target,
        trust_remote_code=True,
        revision=args.revision,
    )
    prompt_style = resolve_prompt_style(args.prompt_style, tok)
    model = AutoModelForCausalLM.from_pretrained(
        target,
        trust_remote_code=True,
        revision=args.revision,
        **load_kwargs,
    )
    model.eval()

    model_cfg = getattr(model, "config", None)
    if model_cfg is not None and getattr(model_cfg, "padded_vocab_size", None) is not None:
        embedding_vocab_size = int(model_cfg.padded_vocab_size)
    elif model_cfg is not None and getattr(model_cfg, "vocab_size", None) is not None:
        embedding_vocab_size = int(model_cfg.vocab_size)
    else:
        embedding_vocab_size = len(tok)

    prompt_style = maybe_fallback_chat_to_plain(prompt_style, tok, embedding_vocab_size)
    context_window = infer_hf_context_window(model, tok)
    max_prompt_tokens = max(1, context_window - int(args.max_new_tokens))

    device = choose_input_device(model)
    print(f"Loaded model on device: {device}")
    print(f"Prompt style: {prompt_style}")
    print(f"Context window: {context_window} | max prompt tokens: {max_prompt_tokens}")
    print(f"Running {len(prompts)} prompt(s)")

    results: List[Dict[str, Any]] = []
    failures = 0

    for i, prompt in enumerate(prompts, start=1):
        prompt_ids = build_prompt_ids(tok, prompt, prompt_style)
        if len(prompt_ids) > max_prompt_tokens:
            prompt_ids = prompt_ids[:max_prompt_tokens]

        if prompt_ids:
            min_id = min(prompt_ids)
            max_id = max(prompt_ids)
            if min_id < 0 or max_id >= embedding_vocab_size:
                # Defensive clamp to avoid CUDA gather asserts on malformed ids.
                prompt_ids = [min(max(int(tid), 0), embedding_vocab_size - 1) for tid in prompt_ids]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_len = int(input_ids.shape[-1])
        inputs = {"input_ids": input_ids}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tok.eos_token_id,
            )

        gen_ids = out[0]
        full_text = tok.decode(gen_ids, skip_special_tokens=False)
        completion_ids = gen_ids[prompt_len:]
        completion_raw = tok.decode(completion_ids, skip_special_tokens=False)
        completion_text = strip_special_markers(completion_raw)
        new_tokens = int(gen_ids.shape[-1] - prompt_len)
        passed = (new_tokens >= args.expect_min_new_tokens) and (len(completion_text) >= args.expect_min_response_chars)
        if not passed:
            failures += 1

        results.append(
            {
                "index": i,
                "prompt": prompt,
                "new_tokens": new_tokens,
                "passed": passed,
                "output_full": full_text,
                "completion_raw": completion_raw,
                "completion_text": completion_text,
            }
        )

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] Prompt {i}/{len(prompts)} | new_tokens={new_tokens}")
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion_text if completion_text else '<empty>'}")

    passed_count = len(prompts) - failures
    print(f"\nSummary: {passed_count}/{len(prompts)} prompts passed")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target": target,
            "revision": args.revision,
            "device_type": args.device_type,
            "prompt_style": prompt_style,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "expect_min_new_tokens": args.expect_min_new_tokens,
            "expect_min_response_chars": args.expect_min_response_chars,
            "results": results,
            "passed": failures == 0,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report to: {out_path}")

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
