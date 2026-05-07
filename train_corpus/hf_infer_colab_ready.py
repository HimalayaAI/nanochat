#!/usr/bin/env python3
"""
Standalone HF inference helper for HimalayaGPT models.

This script intentionally depends only on:
- torch
- transformers
- huggingface_hub

No nanochat repo internals are required at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


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
    p = argparse.ArgumentParser(description="Run robust HF inference for HimalayaGPT")
    p.add_argument("--repo-id", default="himalaya-ai/himalayagpt-0.5b-it")
    p.add_argument("--revision", default="main")
    p.add_argument("--force-download", action="store_true", help="Force fresh snapshot download from HF")
    p.add_argument("--prompt-style", choices=["auto", "chat", "plain"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--min-new-tokens", type=int, default=1, help="Do not stop before this many generated tokens")
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.08,
        help="Penalty >1.0 discourages repeats (1.0 disables).",
    )
    p.add_argument(
        "--stop-tokens",
        default="<|assistant_end|>,<|output_end|>,<|user_start|>",
        help="Comma-separated special tokens that trigger early stop.",
    )
    p.add_argument(
        "--stop-on-special",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stop when generated token matches --stop-tokens.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--use-device-map",
        action="store_true",
        help="Use transformers/accelerate device_map='auto'. Disabled by default for maximum Colab compatibility.",
    )
    p.add_argument("--prompts-file", default=None, help="Optional .txt file with one prompt per line")
    p.add_argument(
        "--load-mode",
        choices=["manual", "from_pretrained"],
        default="manual",
        help="Model loading strategy. `manual` avoids meta-tensor edge cases on Colab.",
    )
    return p.parse_args()


def _special_id(tokenizer, token: str) -> int | None:
    special_map = getattr(tokenizer, "_special_to_id", None)
    if isinstance(special_map, dict) and token in special_map:
        tid = int(special_map[token])
        if tid >= 0:
            return tid
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None:
        return None
    if tokenizer.unk_token_id is not None and tid == tokenizer.unk_token_id and token != tokenizer.unk_token:
        return None
    return int(tid)


def _strip_special(text: str) -> str:
    out = text
    for tok in SPECIAL_TOKENS:
        out = out.replace(tok, "")
    return out.strip()


def _load_prompts(prompts_file: str | None) -> List[str]:
    if not prompts_file:
        return list(DEFAULT_PROMPTS)
    p = Path(prompts_file)
    lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No prompts in {p}")
    return lines


def _choose_device(arg: str):
    import torch

    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _choose_torch_dtype(dtype_arg: str, device):
    import torch

    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    # auto policy: prefer bf16 if supported, otherwise fp32 (avoid fp16 instability)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _resolve_prompt_style(style_arg: str, tokenizer) -> str:
    if style_arg != "auto":
        return style_arg
    needed = ["<|user_start|>", "<|user_end|>", "<|assistant_start|>"]
    if all(_special_id(tokenizer, tok) is not None for tok in needed):
        return "chat"
    return "plain"


def _build_prompt_ids(tokenizer, prompt: str, prompt_style: str, vocab_size: int) -> List[int]:
    user_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if prompt_style == "plain":
        ids = user_ids
    else:
        bos = _special_id(tokenizer, "<|bos|>")
        u_s = _special_id(tokenizer, "<|user_start|>")
        u_e = _special_id(tokenizer, "<|user_end|>")
        a_s = _special_id(tokenizer, "<|assistant_start|>")
        if None in (bos, u_s, u_e, a_s):
            raise RuntimeError("Chat style requested but special tokens are missing")
        ids = [int(bos), int(u_s)] + user_ids + [int(u_e), int(a_s)]

    # hard clamp safety against malformed token ids
    return [min(max(int(t), 0), vocab_size - 1) for t in ids]


def _top_k_filter(logits, top_k: int):
    import torch

    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    masked = logits.clone()
    masked[masked < v[:, [-1]]] = -float("inf")
    return masked


def _apply_repetition_penalty(logits, token_ids, penalty: float):
    import torch

    if penalty <= 1.0 or token_ids.numel() == 0:
        return logits
    uniq = torch.unique(token_ids)
    penalized = logits.clone()
    penalized[:, uniq] = penalized[:, uniq] / penalty
    return penalized


def _has_meta_tensors(model) -> bool:
    import torch

    for p in model.parameters():
        if isinstance(p, torch.Tensor) and p.is_meta:
            return True
    for b in model.buffers():
        if isinstance(b, torch.Tensor) and b.is_meta:
            return True
    return False


def _load_model_manual(local_dir: str, torch_dtype, device):
    from pathlib import Path

    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    weights_path = Path(local_dir) / "model.safetensors"
    state_dict = load_file(str(weights_path), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "State dict mismatch while manual-loading model.safetensors. "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    return model


def _load_model_from_pretrained(local_dir: str, torch_dtype, device, use_device_map: bool):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if (device.type == "cuda" and use_device_map) else None,
        low_cpu_mem_usage=bool(device.type == "cuda" and use_device_map),
    )
    if device.type == "cpu" or (device.type == "cuda" and not use_device_map):
        if _has_meta_tensors(model):
            print("[warn] Detected meta tensors after load; retrying with device_map='auto' fallback.")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map="auto" if device.type == "cuda" else None,
                low_cpu_mem_usage=bool(device.type == "cuda"),
            )
        else:
            model = model.to(device)
    model.eval()
    return model


def _parse_stop_tokens(raw: str) -> List[str]:
    out = []
    for part in raw.split(","):
        token = part.strip()
        if token:
            out.append(token)
    return out


def _resolve_stop_ids(tokenizer, stop_tokens: List[str]) -> List[int]:
    ids: List[int] = []
    for token in stop_tokens:
        tid = _special_id(tokenizer, token)
        if tid is not None:
            ids.append(int(tid))
    # preserve order but deduplicate
    seen = set()
    uniq = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            uniq.append(tid)
    return uniq


def generate_compat(
    model,
    input_ids,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    stop_ids: List[int],
    stop_on_special: bool,
    seed: int,
):
    import torch
    import torch.nn.functional as F

    ids = input_ids
    rng = None
    if temperature > 0:
        rng = torch.Generator(device=ids.device)
        rng.manual_seed(seed)

    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=attention_mask, return_dict=True).logits[:, -1, :]
        logits = _apply_repetition_penalty(logits, ids[0], penalty=repetition_penalty)
        logits = _top_k_filter(logits, top_k)
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat((ids, next_ids), dim=1)
        if stop_on_special and stop_ids and (ids.shape[1] - input_ids.shape[1]) >= max(0, min_new_tokens):
            next_id = int(next_ids.item())
            if next_id in stop_ids:
                break
    return ids


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if transformers.__version__ == "4.57.0":
        print(
            "[warn] transformers==4.57.0 is yanked on PyPI due to packaging issues. "
            "Prefer transformers>=4.57.1."
        )

    prompts = _load_prompts(args.prompts_file)
    device = _choose_device(args.device)
    torch_dtype = _choose_torch_dtype(args.dtype, device)

    local = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        force_download=args.force_download,
    )
    sha = Path(local).name

    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    if args.load_mode == "manual":
        model = _load_model_manual(local, torch_dtype=torch_dtype, device=device)
    else:
        model = _load_model_from_pretrained(
            local,
            torch_dtype=torch_dtype,
            device=device,
            use_device_map=args.use_device_map,
        )

    cfg = model.config
    vocab_size = int(getattr(cfg, "padded_vocab_size", getattr(cfg, "vocab_size", len(tok))))
    context_window = int(
        min(
            x
            for x in [
                getattr(cfg, "sequence_len", None),
                getattr(cfg, "max_position_embeddings", None),
                getattr(tok, "model_max_length", None),
            ]
            if isinstance(x, int) and x > 0
        )
    )

    prompt_style = _resolve_prompt_style(args.prompt_style, tok)
    max_prompt_tokens = max(1, context_window - args.max_new_tokens)
    stop_tokens = _parse_stop_tokens(args.stop_tokens)
    stop_ids = _resolve_stop_ids(tok, stop_tokens) if args.stop_on_special else []

    print(f"repo={args.repo_id} revision={args.revision} snapshot_sha={sha}")
    print(
        f"device={device} dtype={torch_dtype} prompt_style={prompt_style} "
        f"use_device_map={args.use_device_map} load_mode={args.load_mode}"
    )
    print(f"context_window={context_window} max_prompt_tokens={max_prompt_tokens}")
    print(
        f"generation: temperature={args.temperature} top_k={args.top_k} "
        f"repetition_penalty={args.repetition_penalty} stop_on_special={args.stop_on_special} stop_ids={stop_ids}"
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for i, prompt in enumerate(prompts, 1):
        p_ids = _build_prompt_ids(tok, prompt, prompt_style, vocab_size)[:max_prompt_tokens]
        input_ids = torch.tensor([p_ids], dtype=torch.long, device=next(model.parameters()).device)

        with torch.no_grad():
            out = generate_compat(
                model=model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                stop_ids=stop_ids,
                stop_on_special=args.stop_on_special,
                seed=args.seed + i,
            )

        completion_ids = out[0, input_ids.shape[1] :]
        completion = tok.decode(completion_ids, skip_special_tokens=False)
        completion = _strip_special(completion)

        print(f"\n--- Prompt {i} ---")
        print("Prompt:", prompt)
        print("Completion:", completion if completion else "<empty>")


if __name__ == "__main__":
    main()
