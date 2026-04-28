"""
Unified Flash Attention interface with automatic FA3 / FA4 / SDPA switching.

Exports `flash_attn` module that matches the FA3 call sites in this repo, while
choosing the fastest available backend at runtime:
- FA3 (Hopper, via kernels hub)
- FA4 (flash-attn-4 CuTe backend)
- SDPA fallback (works everywhere)

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import inspect
import os
import warnings
import torch
import torch.nn.functional as F
from types import SimpleNamespace


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


def _load_flash_attention_4():
    """Try to load Flash Attention 4 (CuTe backend)."""
    if not torch.cuda.is_available():
        return None
    try:
        # flash-attn-4 exposes this path
        from flash_attn.cute import flash_attn_func as fa4_func
        sig = inspect.signature(fa4_func)
        return SimpleNamespace(func=fa4_func, sig=sig)
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
_fa4 = _load_flash_attention_4()
HAS_FA4 = _fa4 is not None


def _get_requested_backend():
    """
    Optional backend override through env var:
      NANOCHAT_ATTN_BACKEND in {auto, fa3, fa4, sdpa}
    """
    value = os.getenv("NANOCHAT_ATTN_BACKEND", "auto").strip().lower()
    allowed = {"auto", "fa3", "fa4", "sdpa"}
    return value if value in allowed else "auto"


def _resolve_backend():
    """
    Decide once which backend to use by default.
    Priority in auto mode: FA3 (Hopper bf16) -> FA4 -> SDPA.
    """
    requested = _get_requested_backend()
    from nanochat.common import COMPUTE_DTYPE

    if requested == "fa3":
        assert HAS_FA3, "NANOCHAT_ATTN_BACKEND=fa3 but FA3 is unavailable"
        assert COMPUTE_DTYPE == torch.bfloat16, (
            f"NANOCHAT_ATTN_BACKEND=fa3 requires bf16, got {COMPUTE_DTYPE}"
        )
        return "fa3"

    if requested == "fa4":
        assert HAS_FA4, "NANOCHAT_ATTN_BACKEND=fa4 but FA4 is unavailable"
        return "fa4"

    if requested == "sdpa":
        return "sdpa"

    # auto
    if HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16:
        return "fa3"
    if HAS_FA4 and torch.cuda.is_available():
        return "fa4"
    return "sdpa"


ATTN_BACKEND = _resolve_backend()
USE_FA3 = ATTN_BACKEND == "fa3"
USE_FA4 = ATTN_BACKEND == "fa4"
_fa4_runtime_fallback_warned = False
_fa4_runtime_enabled = USE_FA4

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:
    def _dynamo_disable(fn):
        return fn


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def _fa4_attention(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Run FA4 with a conservative call contract.
    We only use FA4 for full-context causal attention in training.
    """
    if not causal:
        raise RuntimeError("FA4 path currently expects causal=True in this wrapper")

    # In this codebase, full-context layers are represented as (sequence_len, 0),
    # not only (-1, -1). Treat both as full-context.
    left, right = window_size
    seq_len = q.size(1)  # q is (B, T, H, D)
    is_full_context = (right == 0) and (left < 0 or left >= seq_len)
    if not is_full_context:
        raise RuntimeError("FA4 path currently supports full-context only in this wrapper")

    kwargs = {}
    params = _fa4.sig.parameters
    if "causal" in params:
        kwargs["causal"] = causal
    # For full-context, omit window_size and let backend take its fast causal path.
    if "dropout_p" in params:
        kwargs["dropout_p"] = 0.0
    if "softmax_scale" in params:
        kwargs["softmax_scale"] = None

    out = _fa4.func(q, k, v, **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


@_dynamo_disable
def _fa4_attention_nodynamo(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Run FA4 outside TorchDynamo tracing.
    This avoids Dynamo/CUTLASS tracing incompatibilities on some stacks.
    """
    return _fa4_attention(q, k, v, causal=causal, window_size=window_size)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if USE_FA4:
        global _fa4_runtime_fallback_warned, _fa4_runtime_enabled
        if not _fa4_runtime_enabled:
            # FA4 failed earlier in this process; stick to SDPA to avoid repeated exceptions/recompiles.
            pass
        else:
            try:
                return _fa4_attention_nodynamo(q, k, v, causal=causal, window_size=window_size)
            except Exception as e:
                # Safe fallback so experiments can continue even when FA4 edge-cases appear.
                _fa4_runtime_enabled = False
                if not _fa4_runtime_fallback_warned:
                    warnings.warn(f"FA4 attention call failed once, falling back to SDPA. Error: {e}")
                    _fa4_runtime_fallback_warned = True

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
    # NOTE: FA4 decode/KV-cache path is not wired yet in this wrapper.
    # Inference remains on SDPA unless FA3 is active.

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
