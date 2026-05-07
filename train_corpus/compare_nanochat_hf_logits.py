#!/usr/bin/env python3
"""
Compare native nanochat checkpoint logits vs HF-exported logits on the same prompt.

This helps diagnose whether HF conversion is faithful or if generation differences
come from decoding/sampling settings.
"""

from __future__ import annotations

import argparse
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare nanochat vs HF logits on a prompt")
    p.add_argument("--source", choices=["base", "sft", "rl"], default="sft")
    p.add_argument("--model-tag", required=True)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--hf-model", required=True, help="HF repo id or local export directory")
    p.add_argument("--hf-revision", default="main")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--prompt-style", choices=["plain", "chat"], default="chat")
    p.add_argument("--prompt", default="नेपालको राजधानी के हो?")
    p.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="cuda")
    return p.parse_args()


def build_nanochat_prompt_tokens(tokenizer, prompt: str, prompt_style: str) -> List[int]:
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


def _hf_special_id(tokenizer, token: str) -> int:
    special_map = getattr(tokenizer, "_special_to_id", None)
    if isinstance(special_map, dict) and token in special_map:
        return int(special_map[token])
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None:
        raise ValueError(f"Missing HF special token: {token}")
    return int(tid)


def build_hf_prompt_ids(tokenizer, prompt: str, prompt_style: str) -> List[int]:
    if prompt_style == "plain":
        return tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if prompt_style == "chat":
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = _hf_special_id(tokenizer, "<|bos|>")
        user_start = _hf_special_id(tokenizer, "<|user_start|>")
        user_end = _hf_special_id(tokenizer, "<|user_end|>")
        assistant_start = _hf_special_id(tokenizer, "<|assistant_start|>")
        ids = [int(bos), int(user_start)]
        ids.extend(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        ids.extend([int(user_end), int(assistant_start)])
        return ids
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def main() -> None:
    args = parse_args()

    import torch
    from nanochat.checkpoint_manager import load_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device_type)

    raw_model, raw_tok, meta = load_model(
        args.source,
        device=device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    raw_model.eval()
    print(f"Loaded nanochat checkpoint step={meta.get('step')}")

    hf_tok = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        trust_remote_code=args.trust_remote_code,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    hf_model = hf_model.to(device)
    hf_model.eval()
    print(f"Loaded HF model: {args.hf_model}@{args.hf_revision}")

    raw_ids = build_nanochat_prompt_tokens(raw_tok, args.prompt, args.prompt_style)
    hf_ids = build_hf_prompt_ids(hf_tok, args.prompt, args.prompt_style)
    print(f"Prompt style: {args.prompt_style}")
    print(f"Prompt length raw={len(raw_ids)} hf={len(hf_ids)}")
    print(f"Prompt token ids identical: {raw_ids == hf_ids}")
    if raw_ids != hf_ids:
        print("WARNING: prompt tokenization mismatch detected.")

    ids = torch.tensor([raw_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        raw_logits = raw_model(ids)  # [B, T, V]
        hf_logits = hf_model(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True).logits

    raw_last = raw_logits[:, -1, :]
    hf_last = hf_logits[:, -1, :]
    abs_diff = (raw_last - hf_last).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())

    raw_top = torch.topk(raw_last, k=10, dim=-1).indices[0].tolist()
    hf_top = torch.topk(hf_last, k=10, dim=-1).indices[0].tolist()
    overlap = len(set(raw_top) & set(hf_top))
    raw_argmax = int(raw_top[0])
    hf_argmax = int(hf_top[0])

    print(f"Last-step logits max_abs_diff={max_abs:.6e} mean_abs_diff={mean_abs:.6e}")
    print(f"Argmax token raw={raw_argmax} hf={hf_argmax} same={raw_argmax == hf_argmax}")
    print(f"Top-10 overlap: {overlap}/10")
    print(f"raw top10: {raw_top}")
    print(f"hf  top10: {hf_top}")


if __name__ == "__main__":
    main()

