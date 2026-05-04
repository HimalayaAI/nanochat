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
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
            **kwargs,
        ):
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
    """
).strip() + "\n"


MODELING_PY = dedent(
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    from configuration_nanochat import NanochatConfig


    def _norm(x):
        return F.rms_norm(x, (x.size(-1),))


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

        def _layer_mask(self, t, left_window, device):
            if left_window < 0 or left_window >= t:
                return None
            i = torch.arange(t, device=device)
            causal = i[:, None] >= i[None, :]
            local = (i[:, None] - i[None, :]) < left_window
            return causal & local

        def forward(self, x, ve, cos_sin, left_window):
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

            q = q.transpose(1, 2)  # B,H,T,D
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if self.n_kv_head != self.n_head:
                rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)

            attn_mask = self._layer_mask(seqlen, left_window, x.device)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=(attn_mask is None),
            )
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
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

        def forward(self, x, ve, cos_sin, left_window):
            x = x + self.attn(_norm(x), ve, cos_sin, left_window)
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

        def _refresh_rotary(self):
            head_dim = self.config.n_embd // self.config.n_head
            weight = self.transformer.wte.weight
            device = weight.device
            dtype = weight.dtype
            t = torch.arange(self.rotary_seq_len, dtype=torch.float32, device=device)
            ch = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            inv = 1.0 / (100000 ** (ch / head_dim))
            freqs = torch.outer(t, inv)
            cos = freqs.cos()[None, :, None, :].to(dtype=dtype)
            sin = freqs.sin()[None, :, None, :].to(dtype=dtype)
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
            if self.cos.device != input_ids.device or self.cos.dtype != self.transformer.wte.weight.dtype:
                self._refresh_rotary()
            cos_sin = self.cos[:, :seqlen], self.sin[:, :seqlen]

            x = self.transformer["wte"](input_ids)
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
                x = block(x, ve, cos_sin, self.window_sizes[i][0])
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

        def __init__(self, config):
            super().__init__(config)
            self.model = NanochatBackbone(config)

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
        vocab_files_names = {"tokenizer_file": "tokenizer.pkl"}
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self, tokenizer_file=None, **kwargs):
            default_name = self.vocab_files_names["tokenizer_file"]
            if tokenizer_file is None:
                tokenizer_file = default_name

            # Resolve both absolute and module-relative tokenizer paths so loading
            # works from local dirs and HF cache snapshots.
            candidate_paths = [tokenizer_file]
            if not os.path.isabs(tokenizer_file):
                module_dir = os.path.dirname(__file__)
                candidate_paths.append(os.path.join(module_dir, tokenizer_file))
                candidate_paths.append(os.path.join(module_dir, default_name))
            resolved = next((p for p in candidate_paths if p and os.path.exists(p)), None)
            if resolved is None:
                raise FileNotFoundError(
                    f"Tokenizer file not found. Tried: {candidate_paths}. "
                    "Ensure tokenizer.pkl is present in the model repo."
                )

            self.tokenizer_file = resolved
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
            return {str(i): i for i in range(self.vocab_size)}

        def _tokenize(self, text: str, **kwargs) -> List[str]:
            return [str(i) for i in self._enc.encode_ordinary(text)]

        def _convert_token_to_id(self, token: str) -> int:
            if token in self._special_to_id:
                return int(self._special_to_id[token])
            return int(token)

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
            shutil.copy2(self.tokenizer_file, out)
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
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    prompt = "नेपालको राजधानी "
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=64)
    print(tok.decode(out[0], skip_special_tokens=True))
    ```

    ## Notes

    - This repo uses custom model/tokenizer code (`trust_remote_code=True`).
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
        "window_pattern": model_cfg.get("window_pattern", "SSSL"),
        "bos_token_id": bos_id,
        "eos_token_id": bos_id,
        "pad_token_id": bos_id,
        "torch_dtype": infer_torch_dtype(state_dict),
        "transformers_version": "4.57.0",
    }
    write_text(out_dir / "config.json", json.dumps(hf_config, ensure_ascii=False, indent=2) + "\n")

    generation_config = {
        "bos_token_id": bos_id,
        "eos_token_id": bos_id,
        "pad_token_id": bos_id,
        "do_sample": True,
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
            "AutoTokenizer": "tokenization_nanochat.NanochatTokenizer",
        },
        "tokenizer_file": "tokenizer.pkl",
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
