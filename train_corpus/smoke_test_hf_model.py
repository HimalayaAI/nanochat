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
    parser.add_argument("--expect-min-new-tokens", type=int, default=1, help="Fail prompt if fewer tokens are generated")
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
    model = AutoModelForCausalLM.from_pretrained(
        target,
        trust_remote_code=True,
        revision=args.revision,
        **load_kwargs,
    )
    model.eval()

    device = choose_input_device(model)
    print(f"Loaded model on device: {device}")
    print(f"Running {len(prompts)} prompt(s)")

    results: List[Dict[str, Any]] = []
    failures = 0

    for i, prompt in enumerate(prompts, start=1):
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        gen_ids = out[0]
        full_text = tok.decode(gen_ids, skip_special_tokens=True)
        new_tokens = int(gen_ids.shape[-1] - prompt_len)
        passed = new_tokens >= args.expect_min_new_tokens
        if not passed:
            failures += 1

        results.append(
            {
                "index": i,
                "prompt": prompt,
                "new_tokens": new_tokens,
                "passed": passed,
                "output": full_text,
            }
        )

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] Prompt {i}/{len(prompts)} | new_tokens={new_tokens}")
        print(f"Prompt: {prompt}")
        print(f"Output: {full_text}")

    passed_count = len(prompts) - failures
    print(f"\nSummary: {passed_count}/{len(prompts)} prompts passed")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "target": target,
            "revision": args.revision,
            "device_type": args.device_type,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "expect_min_new_tokens": args.expect_min_new_tokens,
            "results": results,
            "passed": failures == 0,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report to: {out_path}")

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
