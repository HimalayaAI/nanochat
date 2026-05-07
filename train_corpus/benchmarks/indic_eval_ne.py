#!/usr/bin/env python3
"""
Evaluate Nepali generations on Aya Indic Eval (npi config).

Supports two backends:
1) nanochat checkpoints (base/sft/rl)
2) Hugging Face CausalLM models

This is designed as a lightweight benchmark loop you can run repeatedly during
base training and SFT experiments.

Examples:

  # Evaluate latest base checkpoint
  uv run python -m train_corpus.benchmarks.indic_eval_ne \
    --backend nanochat --source base --model-tag d15_harl_fulltokens_sdpa_bs32

  # Evaluate latest SFT checkpoint
  uv run python -m train_corpus.benchmarks.indic_eval_ne \
    --backend nanochat --source sft --model-tag sft_harl_ne_v1 --prompt-style chat

  # Evaluate an HF-exported model
  uv run python -m train_corpus.benchmarks.indic_eval_ne \
    --backend hf --hf-model himalaya-ai/himalayagpt-0.5b --trust-remote-code
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


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
    parser = argparse.ArgumentParser(description="Evaluate Nepali generation on Aya Indic Eval")

    parser.add_argument("--backend", choices=["nanochat", "hf"], default="nanochat", help="Model backend")
    parser.add_argument("--device-type", choices=["auto", "cuda", "cpu", "mps"], default="auto", help="Execution device")

    # nanochat checkpoint backend
    parser.add_argument("--source", choices=["base", "sft", "rl"], default="base", help="Checkpoint source (nanochat backend)")
    parser.add_argument("--model-tag", default=None, help="Checkpoint model tag (nanochat backend)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (nanochat backend)")

    # HF backend
    parser.add_argument("--hf-model", default=None, help="HF repo id or local folder (hf backend)")
    parser.add_argument("--hf-revision", default="main", help="HF revision")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True for HF loads")
    parser.add_argument(
        "--hf-dtype",
        choices=["auto", "float32", "bfloat16"],
        default="auto",
        help="HF load dtype for CUDA runs. Use `auto` first for conversion parity checks.",
    )

    # dataset
    parser.add_argument("--dataset-id", default="Cognitive-Lab/Aya_Indic_Eval", help="HF dataset id")
    parser.add_argument(
        "--dataset-config",
        default="npi",
        help="Dataset config. For Nepali in Aya Indic Eval use npi (alias ne is auto-mapped).",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--prompt-field", default="inputs", help="Prompt field")
    parser.add_argument("--target-field", default="targets", help="Reference/target field")
    parser.add_argument("--max-examples", type=int, default=200, help="Max rows to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Row offset before sampling")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before truncation")

    # generation
    parser.add_argument("--prompt-style", choices=["auto", "plain", "chat"], default="auto", help="Prompt rendering style")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling (HF backend)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (nanochat and HF compat mode, 0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--hf-generation-mode",
        choices=["compat", "generate"],
        default="compat",
        help="HF backend generation mode: `compat` matches nanochat-style sampling loop.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Override max context window for HF generation (0 = auto-detect from config/tokenizer).",
    )
    parser.add_argument(
        "--no-truncate-context",
        action="store_true",
        help="Disable prompt truncation for overlong HF prompts (will error if prompt exceeds context window).",
    )

    # output
    parser.add_argument(
        "--output-dir",
        default="data/benchmarks/indic_eval_ne",
        help="Output folder for metrics + predictions",
    )
    parser.add_argument("--run-name", default="", help="Optional run name for output filenames")

    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(pred: str, ref: str) -> float:
    pred_tokens = [t for t in normalize_text(pred).split(" ") if t]
    ref_tokens = [t for t in normalize_text(ref).split(" ") if t]
    return _f1_from_sequences(pred_tokens, ref_tokens)


def char_f1(pred: str, ref: str) -> float:
    pred_chars = [c for c in normalize_text(pred) if not c.isspace()]
    ref_chars = [c for c in normalize_text(ref) if not c.isspace()]
    return _f1_from_sequences(pred_chars, ref_chars)


def _f1_from_sequences(pred: Sequence[str], ref: Sequence[str]) -> float:
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_text(pred) == normalize_text(ref))


def strip_special_markers(text: str) -> str:
    out = text
    for tok in SPECIAL_TOKENS:
        out = out.replace(tok, "")
    return out.strip()


def resolve_prompt_style(args: argparse.Namespace) -> str:
    if args.prompt_style != "auto":
        return args.prompt_style
    if args.backend == "nanochat" and args.source in {"sft", "rl"}:
        return "chat"
    return "plain"


def resolve_dataset_config(config_name: str) -> str:
    if config_name.lower() in {"ne", "nep", "nepali"}:
        return "npi"
    return config_name


def choose_device(args: argparse.Namespace):
    import torch

    if args.device_type == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(args.device_type)


def infer_hf_context_window(args: argparse.Namespace, model, tokenizer) -> int:
    if args.context_window and args.context_window > 0:
        return int(args.context_window)

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


def build_nanochat_prompt_tokens(tokenizer, prompt: str, prompt_style: str) -> List[int]:
    if prompt_style == "plain":
        bos = tokenizer.get_bos_token_id()
        return tokenizer.encode(prompt, prepend=bos)

    if prompt_style == "chat":
        bos = tokenizer.get_bos_token_id()
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        ids = [bos, user_start]
        ids.extend(tokenizer.encode(prompt))
        ids.extend([user_end, assistant_start])
        return ids

    raise ValueError(f"Unknown prompt style: {prompt_style}")


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
                "Chat prompt style requested, but tokenizer is missing special tokens: "
                + ", ".join(missing)
            )
        ids = [int(bos), int(user_start)]
        ids.extend(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        ids.extend([int(user_end), int(assistant_start)])
        return ids

    raise ValueError(f"Unknown prompt style: {prompt_style}")


def load_eval_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    config_name = resolve_dataset_config(args.dataset_config)
    ds = load_dataset(args.dataset_id, config_name, split=args.split)

    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)

    start = max(0, args.offset)
    end = start + args.max_examples if args.max_examples > 0 else len(ds)
    end = min(end, len(ds))
    rows = [ds[i] for i in range(start, end)]
    if not rows:
        raise ValueError("No rows selected from dataset; check --offset/--max-examples")

    for i, row in enumerate(rows):
        if args.prompt_field not in row:
            raise KeyError(f"Row {i} missing prompt field: {args.prompt_field}")
        if args.target_field not in row:
            raise KeyError(f"Row {i} missing target field: {args.target_field}")

    return rows


def run_nanochat_eval(args: argparse.Namespace, prompt_style: str, rows: List[Dict[str, Any]], device):
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine

    model, tokenizer, meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)

    print(f"Loaded nanochat model source={args.source} tag={args.model_tag or '<auto>'} step={meta.get('step')}")
    print(f"Prompt style: {prompt_style}")

    top_k = args.top_k if args.top_k > 0 else None
    outputs: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = str(row[args.prompt_field])
        target = str(row[args.target_field])
        prompt_tokens = build_nanochat_prompt_tokens(tokenizer, prompt, prompt_style)
        generated_batch, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            seed=args.seed + idx,
        )
        completion_ids = generated_batch[0][len(prompt_tokens) :]
        prediction_raw = tokenizer.decode(completion_ids)
        prediction = strip_special_markers(prediction_raw)
        outputs.append(
            {
                "row_index": idx,
                "id": row.get("id"),
                "prompt": prompt,
                "target": target,
                "prediction_raw": prediction_raw,
                "prediction": prediction,
            }
        )
        if (idx + 1) % 20 == 0 or (idx + 1) == len(rows):
            print(f"Progress: {idx + 1}/{len(rows)}")

    model_info = {
        "backend": "nanochat",
        "source": args.source,
        "model_tag": args.model_tag,
        "step": meta.get("step"),
        "model_config": meta.get("model_config"),
    }
    return outputs, model_info


def run_hf_eval(args: argparse.Namespace, prompt_style: str, rows: List[Dict[str, Any]], device):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.hf_model:
        raise ValueError("--hf-model is required when --backend hf")

    model_kwargs: Dict[str, Any] = {"revision": args.hf_revision, "trust_remote_code": args.trust_remote_code}
    if device.type == "cuda":
        if args.hf_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif args.hf_dtype == "float32":
            model_kwargs["torch_dtype"] = torch.float32
        else:
            model_kwargs["torch_dtype"] = "auto"
        model_kwargs["device_map"] = "auto"
    else:
        if args.hf_dtype == "bfloat16":
            raise ValueError("--hf-dtype bfloat16 is only supported on CUDA devices")
        model_kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"HF load kwargs: {model_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, **model_kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()

    context_window = infer_hf_context_window(args, model, tokenizer)
    max_prompt_tokens = context_window - args.max_new_tokens
    if max_prompt_tokens < 1:
        raise ValueError(
            f"max_new_tokens ({args.max_new_tokens}) must be smaller than context window ({context_window}). "
            "Reduce --max-new-tokens or set a larger --context-window override."
        )

    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    truncate_context = not args.no_truncate_context

    print(f"Loaded HF model: {args.hf_model}@{args.hf_revision}")
    print(f"Prompt style: {prompt_style}")
    print(f"HF context window: {context_window} | max prompt tokens (after reserve): {max_prompt_tokens}")

    do_sample = args.temperature > 0.0
    outputs: List[Dict[str, Any]] = []
    truncated_count = 0
    invalid_id_rows = 0

    for idx, row in enumerate(rows):
        prompt = str(row[args.prompt_field])
        target = str(row[args.target_field])

        prompt_ids = build_hf_prompt_ids(tokenizer, prompt, prompt_style)
        if len(prompt_ids) > max_prompt_tokens:
            if not truncate_context:
                raise ValueError(
                    f"Row {idx} prompt token length {len(prompt_ids)} exceeds max allowed {max_prompt_tokens}. "
                    "Re-run with smaller --max-new-tokens, larger --context-window, or remove --no-truncate-context."
                )
            # Keep the prompt prefix (instruction + leading context) and reserve space for generation.
            prompt_ids = prompt_ids[:max_prompt_tokens]
            truncated_count += 1

        if prompt_ids:
            min_id = min(prompt_ids)
            max_id = max(prompt_ids)
            if min_id < 0 or max_id >= vocab_size:
                invalid_id_rows += 1
                # Defensive clamp: avoid device-side assert from invalid embedding indices.
                prompt_ids = [min(max(int(tid), 0), vocab_size - 1) for tid in prompt_ids]

        input_ids = torch.tensor([prompt_ids], dtype=torch.long)
        if device.type in {"cuda", "mps"}:
            input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "temperature": args.temperature if do_sample else None,
            "top_p": args.top_p if do_sample else None,
            "top_k": args.top_k if do_sample and args.top_k > 0 else None,
            "pad_token_id": tokenizer.eos_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if args.hf_generation_mode == "compat":
            ids = input_ids
            rng = None
            if do_sample:
                rng = torch.Generator(device=ids.device)
                rng.manual_seed(args.seed + idx)
            with torch.no_grad():
                for _ in range(args.max_new_tokens):
                    model_out = model(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True)
                    logits = model_out.logits[:, -1, :]
                    if args.top_k and args.top_k > 0:
                        v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
                        logits = logits.clone()
                        logits[logits < v[:, [-1]]] = -float("inf")
                    if do_sample:
                        logits = logits / args.temperature
                        probs = F.softmax(logits, dim=-1)
                        next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
                    else:
                        next_ids = torch.argmax(logits, dim=-1, keepdim=True)
                    ids = torch.cat((ids, next_ids), dim=1)
            out = ids
        else:
            with torch.no_grad():
                out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        completion_ids = out[0, input_ids.shape[1] :]
        prediction_raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
        prediction = strip_special_markers(prediction_raw)
        outputs.append(
            {
                "row_index": idx,
                "id": row.get("id"),
                "prompt": prompt,
                "target": target,
                "prediction_raw": prediction_raw,
                "prediction": prediction,
            }
        )

        if (idx + 1) % 20 == 0 or (idx + 1) == len(rows):
            print(f"Progress: {idx + 1}/{len(rows)}")

    if truncated_count:
        print(f"Truncated overlong prompts: {truncated_count}/{len(rows)}")
    if invalid_id_rows:
        print(f"Rows with out-of-range token ids (clamped defensively): {invalid_id_rows}/{len(rows)}")

    model_info = {
        "backend": "hf",
        "hf_model": args.hf_model,
        "hf_revision": args.hf_revision,
        "hf_dtype": args.hf_dtype,
        "trust_remote_code": args.trust_remote_code,
        "hf_generation_mode": args.hf_generation_mode,
    }
    return outputs, model_info


def score_predictions(records: List[Dict[str, Any]]) -> Dict[str, float]:
    em_vals = [exact_match(r["prediction"], r["target"]) for r in records]
    tok_f1_vals = [token_f1(r["prediction"], r["target"]) for r in records]
    char_f1_vals = [char_f1(r["prediction"], r["target"]) for r in records]
    return {
        "exact_match": mean(em_vals),
        "token_f1": mean(tok_f1_vals),
        "char_f1": mean(char_f1_vals),
        "num_examples": len(records),
    }


def with_scores(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rec in records:
        em = exact_match(rec["prediction"], rec["target"])
        tf1 = token_f1(rec["prediction"], rec["target"])
        cf1 = char_f1(rec["prediction"], rec["target"])
        rec2 = dict(rec)
        rec2.update(
            {
                "exact_match": em,
                "token_f1": tf1,
                "char_f1": cf1,
            }
        )
        out.append(rec2)
    return out


def make_run_id(args: argparse.Namespace, prompt_style: str) -> str:
    if args.run_name:
        return args.run_name
    model_part = args.hf_model if args.backend == "hf" else f"{args.source}_{args.model_tag or 'auto'}"
    model_part = model_part.replace("/", "_").replace(" ", "_")
    return f"{model_part}_{prompt_style}_{args.dataset_config}_{args.max_examples}"


def main() -> None:
    args = parse_args()
    prompt_style = resolve_prompt_style(args)
    device = choose_device(args)

    rows = load_eval_rows(args)
    print(f"Loaded {len(rows)} rows from {args.dataset_id}/{resolve_dataset_config(args.dataset_config)}:{args.split}")

    if args.backend == "nanochat":
        records, model_info = run_nanochat_eval(args, prompt_style, rows, device)
    else:
        records, model_info = run_hf_eval(args, prompt_style, rows, device)

    scored_records = with_scores(records)
    metrics = score_predictions(scored_records)

    run_id = make_run_id(args, prompt_style)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{run_id}.predictions.jsonl"
    summary_path = out_dir / f"{run_id}.summary.json"

    with pred_path.open("w", encoding="utf-8") as f:
        for rec in scored_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "dataset": {
            "id": args.dataset_id,
            "config": resolve_dataset_config(args.dataset_config),
            "split": args.split,
            "prompt_field": args.prompt_field,
            "target_field": args.target_field,
            "max_examples": args.max_examples,
            "offset": args.offset,
            "shuffle": args.shuffle,
        },
        "generation": {
            "prompt_style": prompt_style,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "model": model_info,
        "metrics": metrics,
        "files": {
            "predictions_jsonl": str(pred_path),
            "summary_json": str(summary_path),
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\nIndic Eval (Nepali) summary")
    print(f"  exact_match: {metrics['exact_match']:.4f}")
    print(f"  token_f1:    {metrics['token_f1']:.4f}")
    print(f"  char_f1:     {metrics['char_f1']:.4f}")
    print(f"  num_examples:{int(metrics['num_examples'])}")
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote summary:     {summary_path}")


if __name__ == "__main__":
    main()
