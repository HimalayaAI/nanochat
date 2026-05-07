#!/usr/bin/env python3
"""
Export a nanochat checkpoint to a Hugging Face model repo format.

This export uses custom `trust_remote_code` files, so consumers can load via:

  from transformers import AutoModelForCausalLM, AutoTokenizer
  tok = AutoTokenizer.from_pretrained("<repo>", trust_remote_code=True)
  mdl = AutoModelForCausalLM.from_pretrained("<repo>", trust_remote_code=True)

The model architecture stays faithful to nanochat (residual scalars, value embeddings,
smear/backout, rotary, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict

import torch
from huggingface_hub import HfApi, get_token, login
from safetensors.torch import save_file

from nanochat.checkpoint_manager import load_model
from nanochat.common import get_base_dir


CONFIG_PY = dedent(
    """
    from transformers import PretrainedConfig


    class NanochatConfig(PretrainedConfig):
        model_type = "nanochat"
        attribute_map = {
            "hidden_size": "n_embd",
            "num_hidden_layers": "n_layer",
            "num_attention_heads": "n_head",
            "num_key_value_heads": "n_kv_head",
            "max_position_embeddings": "sequence_len",
        }

        def __init__(
            self,
            vocab_size=32768,
            padded_vocab_size=32768,
            sequence_len=2048,
            n_layer=24,
            n_head=12,
            n_kv_head=12,
            n_embd=1536,
            window_pattern="SSSL",
            # Standard HF aliases (accepted so configs remain loadable even if
            # written with generic field names by external tooling).
            hidden_size=None,
            num_hidden_layers=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            max_position_embeddings=None,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
            **kwargs,
        ):
            if hidden_size is not None:
                n_embd = hidden_size
            if num_hidden_layers is not None:
                n_layer = num_hidden_layers
            if num_attention_heads is not None:
                n_head = num_attention_heads
            if num_key_value_heads is not None:
                n_kv_head = num_key_value_heads
            if max_position_embeddings is not None:
                sequence_len = max_position_embeddings

            super().__init__(
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                **kwargs,
            )
            self.vocab_size = vocab_size
            self.padded_vocab_size = padded_vocab_size
            self.sequence_len = sequence_len
            self.n_layer = n_layer
            self.n_head = n_head
            self.n_kv_head = n_kv_head
            self.n_embd = n_embd
            self.window_pattern = window_pattern

            # Mirror common HF config keys for generation/cache utilities and
            # generic ecosystem tools that expect canonical names.
            self.hidden_size = self.n_embd
            self.num_hidden_layers = self.n_layer
            self.num_attention_heads = self.n_head
            self.num_key_value_heads = self.n_kv_head
            self.max_position_embeddings = self.sequence_len
            self.head_dim = self.n_embd // self.n_head
            self.intermediate_size = 4 * self.n_embd
            self.is_decoder = True
            self.use_cache = use_cache
            self.tie_word_embeddings = False
    """
).strip() + "\n"


MODELING_PY = dedent(
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from .configuration_nanochat import NanochatConfig


    def _norm(x):
        return F.rms_norm(x, (x.size(-1),))


    def _detect_compute_dtype(device):
        if device.type == "cuda":
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(idx)
            if (major, minor) >= (8, 0):
                return torch.bfloat16
        return torch.float32


    class Linear(nn.Linear):
        def forward(self, x):
            return F.linear(x, self.weight.to(dtype=x.dtype))


    def _has_ve(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2


    def _apply_rotary(x, cos, sin):
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], dim=-1)


    def _sdpa_attention(q, k, v, window_size, enable_gqa):
        # q/k/v are (B, H, T, D)
        t_q = q.size(2)
        t_k = k.size(2)
        left_window = window_size[0]

        # Full causal attention when the window covers full context.
        if (left_window < 0 or left_window >= t_q) and t_q == t_k:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

        # Single-token decode path.
        if t_q == 1:
            if left_window >= 0 and left_window < t_k:
                start = max(0, t_k - (left_window + 1))
                k = k[:, :, start:, :]
                v = v[:, :, start:, :]
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

        # Build explicit causal (+ optional sliding-window) mask.
        device = q.device
        row_idx = (t_k - t_q) + torch.arange(t_q, device=device).unsqueeze(1)
        col_idx = torch.arange(t_k, device=device).unsqueeze(0)
        mask = col_idx <= row_idx
        if left_window >= 0 and left_window < t_k:
            mask = mask & ((row_idx - col_idx) <= left_window)

        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


    def _flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
        if not causal:
            raise NotImplementedError("Nanochat HF export currently supports only causal attention")
        # SDPA fallback mirroring nanochat.flash_attention semantics.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        enable_gqa = q.size(1) != k.size(1)
        y = _sdpa_attention(q, k, v, window_size=window_size, enable_gqa=enable_gqa)
        return y.transpose(1, 2)


    class NanochatAttention(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.layer_idx = layer_idx
            self.n_head = config.n_head
            self.n_kv_head = config.n_kv_head
            self.n_embd = config.n_embd
            self.head_dim = self.n_embd // self.n_head
            self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
            self.ve_gate_channels = 12
            self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if _has_ve(layer_idx, config.n_layer) else None

        def forward(self, x, ve, cos_sin, window_size):
            bsz, seqlen, _ = x.size()
            q = self.c_q(x).view(bsz, seqlen, self.n_head, self.head_dim)
            k = self.c_k(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)

            if ve is not None:
                ve = ve.view(bsz, seqlen, self.n_kv_head, self.head_dim)
                gate = 3.0 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve

            cos, sin = cos_sin
            q = _apply_rotary(q, cos, sin)
            k = _apply_rotary(k, cos, sin)
            q = 1.2 * _norm(q)
            k = 1.2 * _norm(k)

            y = _flash_attn_func(q, k, v, causal=True, window_size=window_size)
            y = y.contiguous().view(bsz, seqlen, -1)
            return self.c_proj(y)


    class NanochatMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

        def forward(self, x):
            return self.c_proj(F.relu(self.c_fc(x)).square())


    class NanochatBlock(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            self.attn = NanochatAttention(config, layer_idx)
            self.mlp = NanochatMLP(config)

        def forward(self, x, ve, cos_sin, window_size):
            x = x + self.attn(_norm(x), ve, cos_sin, window_size)
            x = x + self.mlp(_norm(x))
            return x


    class NanochatBackbone(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.transformer = nn.ModuleDict(
                {
                    "wte": nn.Embedding(config.padded_vocab_size, config.n_embd),
                    "h": nn.ModuleList([NanochatBlock(config, i) for i in range(config.n_layer)]),
                }
            )
            self.lm_head = Linear(config.n_embd, config.padded_vocab_size, bias=False)
            self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
            self.smear_gate = Linear(24, 1, bias=False)
            self.smear_lambda = nn.Parameter(torch.zeros(1))
            self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
            head_dim = config.n_embd // config.n_head
            kv_dim = config.n_kv_head * head_dim
            self.value_embeds = nn.ModuleDict(
                {
                    str(i): nn.Embedding(config.padded_vocab_size, kv_dim)
                    for i in range(config.n_layer)
                    if _has_ve(i, config.n_layer)
                }
            )
            self.window_sizes = self._compute_window_sizes(config)
            self.rotary_seq_len = config.sequence_len * 10
            self.register_buffer("cos", torch.empty(1), persistent=False)
            self.register_buffer("sin", torch.empty(1), persistent=False)
            self._refresh_rotary()

        def _precompute_rotary_embeddings(self, seq_len, head_dim, device, dtype):
            channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            inv_freq = 1.0 / (100000 ** (channel_range / head_dim))
            t = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs = torch.outer(t, inv_freq)
            cos = freqs.cos()[None, :, None, :].to(dtype=dtype)
            sin = freqs.sin()[None, :, None, :].to(dtype=dtype)
            return cos, sin

        def _refresh_rotary(self, device=None, dtype=None):
            head_dim = self.config.n_embd // self.config.n_head
            if device is None:
                device = self.transformer.wte.weight.device
            if dtype is None:
                dtype = self.transformer.wte.weight.dtype
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=device, dtype=dtype)
            self.cos = cos
            self.sin = sin

        def _compute_window_sizes(self, config):
            pattern = config.window_pattern.upper()
            long_window = config.sequence_len
            short_window = -(-long_window // 4 // 128) * 128
            lut = {"L": (long_window, 0), "S": (short_window, 0)}
            out = []
            for layer_idx in range(config.n_layer):
                ch = pattern[layer_idx % len(pattern)]
                out.append(lut[ch])
            out[-1] = (long_window, 0)
            return out

        def forward(self, input_ids):
            bsz, seqlen = input_ids.shape
            compute_dtype = _detect_compute_dtype(input_ids.device)
            if self.cos.device != input_ids.device or self.cos.dtype != compute_dtype:
                self._refresh_rotary(device=input_ids.device, dtype=compute_dtype)

            if seqlen > self.cos.size(1):
                raise ValueError(
                    f"Sequence length {seqlen} exceeds rotary cache length {self.cos.size(1)}. "
                    "Re-export with larger sequence_len if needed."
                )
            cos_sin = self.cos[:, :seqlen], self.sin[:, :seqlen]

            x = self.transformer["wte"](input_ids)
            x = x.to(compute_dtype)
            x = _norm(x)
            if seqlen > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

            x0 = x
            backout_layer = self.config.n_layer // 2
            x_backout = None
            for i, block in enumerate(self.transformer["h"]):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](input_ids).to(x.dtype) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i])
                if i == backout_layer:
                    x_backout = x

            if x_backout is not None:
                x = x - self.backout_lambda.to(x.dtype) * x_backout
            x = _norm(x)
            logits = self.lm_head(x)[..., :self.config.vocab_size].float()
            softcap = 15.0
            logits = softcap * torch.tanh(logits / softcap)
            return logits


    class NanochatForCausalLM(PreTrainedModel):
        config_class = NanochatConfig
        base_model_prefix = "model"
        main_input_name = "input_ids"
        _tied_weights_keys = []

        def __init__(self, config):
            super().__init__(config)
            self.model = NanochatBackbone(config)

        @property
        def all_tied_weights_keys(self):
            # Compatibility shim for some transformers/accelerate versions that
            # access `model.all_tied_weights_keys` during device_map inference.
            return {k: None for k in getattr(self, "_tied_weights_keys", [])}

        def get_input_embeddings(self):
            return self.model.transformer["wte"]

        def set_input_embeddings(self, value):
            self.model.transformer["wte"] = value

        def get_output_embeddings(self):
            return self.model.lm_head

        def set_output_embeddings(self, value):
            self.model.lm_head = value

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            use_cache=None,
            past_key_values=None,
            return_dict=None,
            **kwargs,
        ):
            if input_ids is None:
                raise ValueError("input_ids must be provided")
            logits = self.model(input_ids)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-1,
                )

            if return_dict is False:
                return (loss, logits) if loss is not None else (logits,)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
            )

        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
            return {"input_ids": input_ids}
    """
).strip() + "\n"


TOKENIZER_PY = dedent(
    """
    import os
    import pickle
    import shutil
    from typing import Dict, List, Optional, Tuple

    from transformers import PreTrainedTokenizer


    class NanochatTokenizer(PreTrainedTokenizer):
        # Use `vocab_file` (not `tokenizer_file`) to avoid collision with the
        # internal `tokenizer_file` reserved for fast-tokenizer JSON handling.
        vocab_files_names = {"vocab_file": "tokenizer.pkl"}
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self, vocab_file=None, **kwargs):
            default_name = self.vocab_files_names["vocab_file"]
            if vocab_file is None:
                vocab_file = default_name

            # Resolve both absolute and module-relative tokenizer paths so loading
            # works from local dirs and HF cache snapshots.
            candidate_paths = [vocab_file]
            if not os.path.isabs(vocab_file):
                module_dir = os.path.dirname(__file__)
                candidate_paths.append(os.path.join(module_dir, vocab_file))
                candidate_paths.append(os.path.join(module_dir, default_name))
            resolved = next((p for p in candidate_paths if p and os.path.exists(p)), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"Tokenizer file not found. Tried: {candidate_paths}. "
                    "Ensure tokenizer.pkl is present in the model repo."
                )

            self.vocab_file = resolved
            with open(resolved, "rb") as f:
                self._enc = pickle.load(f)
            self._special_to_id = dict(getattr(self._enc, "_special_tokens", {}))
            self._id_to_special = {v: k for k, v in self._special_to_id.items()}
            bos = kwargs.pop("bos_token", "<|bos|>")
            eos = kwargs.pop("eos_token", bos)
            pad = kwargs.pop("pad_token", bos)
            super().__init__(bos_token=bos, eos_token=eos, pad_token=pad, **kwargs)

        @property
        def vocab_size(self) -> int:
            return int(self._enc.n_vocab)

        def get_vocab(self) -> Dict[str, int]:
            vocab = {str(i): i for i in range(self.vocab_size)}
            for token, token_id in self._special_to_id.items():
                tid = int(token_id)
                if 0 <= tid < self.vocab_size:
                    vocab[token] = tid
            return vocab

        def _tokenize(self, text: str, **kwargs) -> List[str]:
            return [str(i) for i in self._enc.encode_ordinary(text)]

        def _convert_token_to_id(self, token: str) -> int:
            if token in self._special_to_id:
                return int(self._special_to_id[token])
            try:
                return int(token)
            except ValueError:
                if self.unk_token_id is not None:
                    return int(self.unk_token_id)
                # Fallback to BOS so conversion never yields out-of-range IDs.
                return int(self._special_to_id.get("<|bos|>", 0))

        def _convert_id_to_token(self, index: int) -> str:
            if index in self._id_to_special:
                return self._id_to_special[index]
            return str(index)

        def convert_tokens_to_string(self, tokens: List[str]) -> str:
            ids = []
            for token in tokens:
                if token in self._special_to_id:
                    continue
                ids.append(int(token))
            return self._enc.decode(ids)

        def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
        ) -> str:
            pieces = []
            buf = []
            for token_id in token_ids:
                token_id = int(token_id)
                if token_id in self._id_to_special:
                    if buf:
                        pieces.append(self._enc.decode(buf))
                        buf = []
                    if not skip_special_tokens:
                        pieces.append(self._id_to_special[token_id])
                else:
                    buf.append(token_id)
            if buf:
                pieces.append(self._enc.decode(buf))
            return "".join(pieces)

        def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
        ) -> List[int]:
            if token_ids_1 is None:
                return token_ids_0
            return token_ids_0 + token_ids_1

        def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
            os.makedirs(save_directory, exist_ok=True)
            name = "tokenizer.pkl" if filename_prefix is None else f"{filename_prefix}-tokenizer.pkl"
            out = os.path.join(save_directory, name)
            shutil.copy2(self.vocab_file, out)
            return (out,)
    """
).strip() + "\n"


README_TEMPLATE = dedent(
    """
    ---
    library_name: transformers
    tags:
      - nanochat
      - causal-lm
      - trust-remote-code
    ---

    # {repo_name}

    Exported from nanochat checkpoints with custom `transformers` remote code.

    ## Load

    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    repo = "{repo_name}"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    prompt = "नेपालको राजधानी "
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=64)
    print(tok.decode(out[0], skip_special_tokens=True))
    ```

    ## Notes

    - This repo uses custom model/tokenizer code (`trust_remote_code=True`).
    - Recommended runtime: `transformers>=4.57.1` (avoid `4.57.0`, which was yanked on PyPI).
    - A standalone helper is included at `run_standalone_inference.py` for HF-only usage.
    - Checkpoint source: `{source}`
    - Model tag: `{model_tag}`
    - Step: `{step}`
    """
).strip() + "\n"


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def infer_torch_dtype(state_dict: Dict[str, torch.Tensor]) -> str:
    for value in state_dict.values():
        return str(value.dtype).replace("torch.", "")
    return "float32"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export nanochat checkpoint to HF model format")
    parser.add_argument("--source", default="base", choices=["base", "sft", "rl"], help="Checkpoint family")
    parser.add_argument("--model-tag", default=None, help="Model tag (e.g. d24_harl_r12)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    parser.add_argument(
        "--output-dir",
        default="data/hf_model_export",
        help="Local export directory",
    )
    parser.add_argument("--upload-repo", default=None, help="Optional HF model repo to upload")
    parser.add_argument("--private", action="store_true", help="Create private model repo on upload")
    parser.add_argument(
        "--commit-message",
        default="Export nanochat checkpoint with custom HF compatibility",
        help="Commit message for HF upload",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, tokenizer, meta = load_model(
        args.source,
        device=device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )

    # HF wrapper class stores the nanochat backbone under `self.model`, so prefix keys accordingly.
    state_dict = {f"model.{k}": v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out_dir / "model.safetensors"))

    model_cfg = dict(meta["model_config"])
    model_cfg["padded_vocab_size"] = int(model.transformer["wte"].weight.shape[0])
    bos_id = int(tokenizer.get_bos_token_id())

    hf_config: Dict[str, Any] = {
        "architectures": ["NanochatForCausalLM"],
        "model_type": "nanochat",
        "auto_map": {
            "AutoConfig": "configuration_nanochat.NanochatConfig",
            "AutoModelForCausalLM": "modeling_nanochat.NanochatForCausalLM",
            "AutoTokenizer": "tokenization_nanochat.NanochatTokenizer",
        },
        "vocab_size": int(model_cfg["vocab_size"]),
        "padded_vocab_size": int(model_cfg["padded_vocab_size"]),
        "sequence_len": int(model_cfg["sequence_len"]),
        "n_layer": int(model_cfg["n_layer"]),
        "n_head": int(model_cfg["n_head"]),
        "n_kv_head": int(model_cfg["n_kv_head"]),
        "n_embd": int(model_cfg["n_embd"]),
        # Standard HF aliases for better compatibility with tooling that
        # expects canonical transformer names.
        "hidden_size": int(model_cfg["n_embd"]),
        "num_hidden_layers": int(model_cfg["n_layer"]),
        "num_attention_heads": int(model_cfg["n_head"]),
        "num_key_value_heads": int(model_cfg["n_kv_head"]),
        "max_position_embeddings": int(model_cfg["sequence_len"]),
        "window_pattern": model_cfg.get("window_pattern", "SSSL"),
        # Nanochat HF wrapper does not currently emit/update KV cache state.
        "use_cache": False,
        "bos_token_id": bos_id,
        "eos_token_id": bos_id,
        "pad_token_id": bos_id,
        # Keep HF defaults in fp32 for numerical stability and closer parity
        # with native nanochat inference on non-FA3 hardware.
        "torch_dtype": "float32",
        "transformers_version": "4.57.0",
    }
    write_text(out_dir / "config.json", json.dumps(hf_config, ensure_ascii=False, indent=2) + "\n")

    generation_config = {
        "bos_token_id": bos_id,
        "eos_token_id": bos_id,
        "pad_token_id": bos_id,
        "do_sample": True,
        "use_cache": False,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    write_text(out_dir / "generation_config.json", json.dumps(generation_config, ensure_ascii=False, indent=2) + "\n")

    base_dir = Path(get_base_dir())
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer_pkl = tokenizer_dir / "tokenizer.pkl"
    token_bytes = tokenizer_dir / "token_bytes.pt"
    if not tokenizer_pkl.exists():
        raise FileNotFoundError(
            f"Tokenizer encoding not found at {tokenizer_pkl}. Train/export tokenizer first."
        )
    shutil.copy2(tokenizer_pkl, out_dir / "tokenizer.pkl")
    if token_bytes.exists():
        shutil.copy2(token_bytes, out_dir / "token_bytes.pt")

    tokenizer_config = {
        "tokenizer_class": "NanochatTokenizer",
        "auto_map": {
            # AutoTokenizer currently expects a pair [slow_class, fast_class].
            # We only implement the slow tokenizer for this custom code path.
            "AutoTokenizer": ["tokenization_nanochat.NanochatTokenizer", None],
        },
        "bos_token": "<|bos|>",
        "eos_token": "<|bos|>",
        "pad_token": "<|bos|>",
        "model_max_length": int(model_cfg["sequence_len"]),
    }
    special_tokens_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|bos|>",
        "pad_token": "<|bos|>",
    }
    write_text(out_dir / "tokenizer_config.json", json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n")
    write_text(out_dir / "special_tokens_map.json", json.dumps(special_tokens_map, ensure_ascii=False, indent=2) + "\n")

    write_text(out_dir / "configuration_nanochat.py", CONFIG_PY)
    write_text(out_dir / "modeling_nanochat.py", MODELING_PY)
    write_text(out_dir / "tokenization_nanochat.py", TOKENIZER_PY)
    # Ship a standalone HF-only inference helper inside the model repo.
    # This avoids requiring the nanochat codebase for downstream users.
    standalone_infer = Path(__file__).with_name("hf_infer_colab_ready.py")
    if standalone_infer.exists():
        shutil.copy2(standalone_infer, out_dir / "run_standalone_inference.py")

    repo_name = args.upload_repo or "local/nanochat-export"
    model_tag = args.model_tag or meta.get("model_tag", "auto")
    step = args.step if args.step is not None else meta.get("step", "latest")
    readme = README_TEMPLATE.format(
        repo_name=repo_name,
        source=args.source,
        model_tag=model_tag,
        step=step,
    )
    write_text(out_dir / "README.md", readme)

    print(f"Exported HF-compatible folder to: {out_dir}")

    if args.upload_repo:
        token = get_token() or os.getenv("HF_TOKEN")
        if not token:
            login()
            token = get_token()
        api = HfApi(token=token)
        api.create_repo(args.upload_repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.upload_repo,
            repo_type="model",
            folder_path=str(out_dir),
            commit_message=args.commit_message,
        )
        print(f"Uploaded model export to: https://huggingface.co/{args.upload_repo}")


if __name__ == "__main__":
    main()
