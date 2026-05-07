#!/usr/bin/env python3
"""
Sanity checks for nanochat Hugging Face model exports.

This script validates:
1) `auto_map` schema in config files (to prevent Hub parser warnings).
2) End-to-end `transformers` loading with `trust_remote_code=True`.
3) A short generation smoke test.

Examples:
  uv run python -m train_corpus.validate_hf_model_export \
    --repo-id himalaya-ai/himalayagpt-0.5b

  uv run python -m train_corpus.validate_hf_model_export \
    --local-dir /data/.../hf_export/himalayagpt-0.5b
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate nanochat HF export")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo-id", default=None, help="HF model repo id (e.g. org/model)")
    group.add_argument("--local-dir", default=None, help="Local export directory")

    parser.add_argument("--revision", default="main", help="HF revision when using --repo-id")
    parser.add_argument("--prompt", default="नेपालको राजधानी ", help="Prompt for generation smoke test")
    parser.add_argument("--max-new-tokens", type=int, default=48, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling top_p")
    parser.add_argument("--seed", type=int, default=42, help="Torch RNG seed")
    parser.add_argument("--skip-generate", action="store_true", help="Only run config checks and model/tokenizer load")
    parser.add_argument(
        "--device-type",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for smoke test",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_from_repo(repo_id: str, filename: str, revision: str) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
        token=token,
    )
    return _read_json(Path(local_path))


def _validate_auto_map_is_string(obj: Dict[str, Any], filename: str, key: str, required: bool = True) -> None:
    auto_map = obj.get("auto_map")
    if not isinstance(auto_map, dict):
        raise ValueError(f"{filename}: expected `auto_map` dict, got {type(auto_map).__name__}")
    if key not in auto_map:
        if required:
            raise KeyError(f"{filename}: missing `auto_map.{key}`")
        return
    value = auto_map[key]
    if not isinstance(value, str):
        raise TypeError(
            f"{filename}: `auto_map.{key}` must be a string, got {type(value).__name__} ({value!r})"
        )


def _validate_tokenizer_auto_map(value: Any, filename: str) -> str:
    # Transformers AutoTokenizer expects either:
    # - a string class path, or
    # - [slow_class_path, fast_class_path_or_none]
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise TypeError(
                f"{filename}: `auto_map.AutoTokenizer` list/tuple must have 2 entries, got {len(value)}"
            )
        slow_cls, fast_cls = value
        if not isinstance(slow_cls, str):
            raise TypeError(
                f"{filename}: slow tokenizer class path must be string, got {type(slow_cls).__name__}"
            )
        if fast_cls is not None and not isinstance(fast_cls, str):
            raise TypeError(
                f"{filename}: fast tokenizer class path must be string or null, got {type(fast_cls).__name__}"
            )
        return slow_cls
    raise TypeError(
        f"{filename}: `auto_map.AutoTokenizer` must be string or [slow, fast], got {type(value).__name__}"
    )


def _check_local_required_files(local_dir: Path) -> None:
    required = [
        "config.json",
        "tokenizer_config.json",
        "model.safetensors",
        "configuration_nanochat.py",
        "modeling_nanochat.py",
        "tokenization_nanochat.py",
        "tokenizer.pkl",
    ]
    missing = [name for name in required if not (local_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required export files in {local_dir}: {missing}")


def _get_device_kwargs(device_type: str) -> Dict[str, Any]:
    import torch

    use_cuda = torch.cuda.is_available() if device_type == "auto" else device_type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False")
    if use_cuda:
        return {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    return {"torch_dtype": torch.float32}


def _run_load_and_generate(target: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, skip_generate: bool, device_type: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = _get_device_kwargs(device_type)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tok = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(target, trust_remote_code=True, **kwargs)

    if skip_generate:
        print("Load test passed (tokenizer + model).")
        return

    inputs = tok(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    text = tok.decode(out[0], skip_special_tokens=True)
    print("Generation output:")
    print(text)


def main() -> None:
    args = parse_args()

    if args.local_dir:
        local_dir = Path(args.local_dir)
        _check_local_required_files(local_dir)
        config = _read_json(local_dir / "config.json")
        tokenizer_config = _read_json(local_dir / "tokenizer_config.json")
        target = str(local_dir)
        location = f"local export: {local_dir}"
    else:
        assert args.repo_id is not None
        config = _load_json_from_repo(args.repo_id, "config.json", args.revision)
        tokenizer_config = _load_json_from_repo(args.repo_id, "tokenizer_config.json", args.revision)
        target = args.repo_id
        location = f"hub repo: {args.repo_id}@{args.revision}"

    _validate_auto_map_is_string(config, "config.json", "AutoConfig", required=True)
    _validate_auto_map_is_string(config, "config.json", "AutoModelForCausalLM", required=True)
    _validate_auto_map_is_string(config, "config.json", "AutoTokenizer", required=True)

    tokenizer_auto_map = tokenizer_config.get("auto_map", {}).get("AutoTokenizer", None)
    if tokenizer_auto_map is None:
        raise KeyError("tokenizer_config.json: missing `auto_map.AutoTokenizer`")
    tokenizer_slow_cls = _validate_tokenizer_auto_map(tokenizer_auto_map, "tokenizer_config.json")
    config_auto_cls = config.get("auto_map", {}).get("AutoTokenizer")
    if tokenizer_slow_cls != config_auto_cls:
        raise ValueError(
            "AutoTokenizer mapping mismatch: "
            f"config.json has {config_auto_cls!r}, tokenizer_config.json slow path has {tokenizer_slow_cls!r}"
        )

    print(f"Config checks passed for {location}")
    _run_load_and_generate(
        target=target,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        skip_generate=args.skip_generate,
        device_type=args.device_type,
    )
    print("HF export validation passed.")


if __name__ == "__main__":
    main()
