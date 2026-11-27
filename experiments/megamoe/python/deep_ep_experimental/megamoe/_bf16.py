# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026
# Modified by guofanfeng 2026

"""BF16 validation, collective execution and per-call autograd state."""

from ctypes import c_float
from dataclasses import dataclass
import math

import torch
from ._autograd import BF16Function

from deep_ep._native import require_native
from deep_ep._runtime import integer


@dataclass(frozen=True)
class Call:
    max_recv: int
    limit: float
    save_type: str
    activation: str = "swiglu"
    beta2: float = 0.0


_KIMI_SITU_BETA_GATE_DEFAULT = 4.0
_KIMI_SITU_BETA_UP_DEFAULT = 25.0


def _resolve_activation(activation, clamp):
    """Normalize (activation, activation_clamp) into (name, beta1, beta2).

    SwiGLU keeps the dsv4 hard clamp: clamp is a nonnegative float or None.
    SiTU-GLU (Kimi K3) softcaps both branches with beta * tanh(x / beta):
    clamp is a (beta_gate, beta_up) tuple, beta_gate must be positive, and
    beta_up of None or 0 selects the linear up branch (0.0 sentinel end to
    end). clamp=None defaults to Kimi K3's (4.0, 25.0). A bare float with
    situglu is rejected so the two families cannot be mixed up.
    """
    name = str(activation).strip().lower().replace("-", "").replace("_", "")
    if name not in ("swiglu", "situglu"):
        raise NotImplementedError(f"activation must be 'swiglu' or 'situglu', got {activation!r}")
    if name == "swiglu":
        if isinstance(clamp, (tuple, list)):
            raise TypeError("activation_clamp for swiglu must be a float or None, not a tuple")
        limit = 0.0 if clamp is None else float(clamp)
        if limit < 0 or not math.isfinite(limit) or not math.isfinite(c_float(limit).value):
            raise ValueError("activation_clamp must be a finite nonnegative float or None")
        return name, limit, 0.0
    if isinstance(clamp, (int, float)) and not isinstance(clamp, bool):
        raise TypeError(
            "activation='situglu' requires activation_clamp as a (gate, up) tuple of softcap betas, "
            f"got a float {clamp!r}"
        )
    if clamp is None:
        betas = (_KIMI_SITU_BETA_GATE_DEFAULT, _KIMI_SITU_BETA_UP_DEFAULT)
    else:
        if not isinstance(clamp, (tuple, list)) or len(clamp) != 2:
            raise ValueError(f"activation_clamp for situglu must be (beta_gate, beta_up), got {clamp!r}")
        betas = (
            _KIMI_SITU_BETA_GATE_DEFAULT if clamp[0] is None else float(clamp[0]),
            0.0 if clamp[1] is None else float(clamp[1]),
        )
    if betas[0] <= 0:
        raise ValueError(f"activation_clamp[0] (beta_gate) must be positive, got {betas[0]!r}")
    if betas[1] < 0:
        raise ValueError(f"activation_clamp[1] (beta_up) must be None/0 (linear up) or positive, got {betas[1]!r}")
    if not all(math.isfinite(v) and math.isfinite(c_float(v).value) for v in betas):
        raise ValueError("activation_clamp betas must be finite and representable as float32")
    return name, betas[0], betas[1]


def validate_buffer_inputs(b, bs):
    for name, tensor, dtype, width in (
        ("x", b.x, torch.bfloat16, b.hidden),
        ("topk_idx", b.topk_idx, torch.int32, b.num_topk),
        ("topk_weights", b.topk_weights, None, b.num_topk),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[0] < bs
            or tensor.shape[0] > b.num_max_tokens_per_rank
            or tensor.shape[1] != width
            or tensor.device != b._device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous [rows,{width}] on the buffer device, within source capacity")
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name} has unsupported dtype")
    if b.topk_weights.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("topk_weights must be bfloat16 or float32")


def prepare_plan(b, bs, activation, beta1, beta2, save_type):
    from .api import _extension

    mode = b.gmm1_save_type if save_type is None else save_type
    if not b.need_grad:
        mode = "none"
    key = (
        bs,
        b.hidden,
        b.intermediate_hidden,
        b.num_topk,
        b.num_experts,
        b.max_recv_tokens,
        b.topk_weights.dtype,
        b.need_grad,
        b.ranks_per_node,
        mode,
        activation,
        beta1,
        beta2,
    )
    if b._plan_key != key:
        native = require_native("BF16 MegaMoE")
        address = native.megamoe_workspace_address(b._runtime.native)
        plan = _extension().BF16Plan(
            address,
            b._runtime.native.capacity,
            b._runtime.rank,
            b._runtime.world_size,
            b.x[:bs],
            b.need_grad,
            b.ranks_per_node,
        )
        plan.prepare(
            bs,
            b.hidden,
            b.intermediate_hidden,
            b.num_topk,
            b.num_experts // b.group.size(),
            b.max_recv_tokens,
            beta1,
            mode,
            b.topk_weights.dtype == torch.float32,
            activation,
            beta2,
        )
        b._plan, b._plan_key, b._plan_bs = plan, key, bs
    return b._plan


def run(y, w1, w2, b, shared_w1, shared_w2, stats, activation, clamp, fast_math):
    from .api import SymmBuffer

    if not isinstance(b, SymmBuffer) or b.mma_type != "bf16xbf16":
        raise TypeError("bf16_mega_moe requires a BF16 SymmBuffer")
    if type(b.need_grad) is not bool:
        raise TypeError("need_grad must be bool")
    if b.gmm1_save_type not in ("bf16", "mxfp8", "none"):
        raise ValueError("gmm1_save_type must be bf16, mxfp8 or none")
    integer("max_recv_tokens", b.max_recv_tokens, 1, (1 << 31) - 1)
    if shared_w1 is not None or shared_w2 is not None:
        raise NotImplementedError("BF16 shared experts are not implemented")
    if fast_math is not True:
        raise NotImplementedError("BF16 supports the default math configuration only")
    activation, beta1, beta2 = _resolve_activation(activation, clamp)
    if not isinstance(y, torch.Tensor) or y.ndim != 2 or y.dtype != torch.bfloat16 or y.device != b._device:
        raise ValueError("y must be BF16 [M,H] on the buffer device")
    bs = y.shape[0]
    if not 0 < bs <= b.num_max_tokens_per_rank or y.shape[1] != b.hidden or not y.is_contiguous():
        raise ValueError("y must be contiguous [M,H] within the source capacity")
    if y.requires_grad:
        raise ValueError("y must not require gradients before this call")
    validate_buffer_inputs(b, bs)
    x, ids, p = b.x[:bs], b.topk_idx[:bs], b.topk_weights[:bs]
    e, f, h = b.num_experts // b.group.size(), b.intermediate_hidden, b.hidden
    canonical = tuple(w1.shape) == (e, 2 * f, h) and tuple(w2.shape) == (e, h, f)
    legacy = tuple(w1.shape) == (e, h, 2 * f) and tuple(w2.shape) == (e, f, h)
    if not (canonical or legacy):
        raise ValueError("BF16 weights must match [E,2F,H]/[E,H,F] or [E,H,2F]/[E,F,H]")
    if any(t.dtype != torch.bfloat16 or t.device != b._device or not t.is_contiguous() for t in (w1, w2)):
        raise ValueError("BF16 weights must be contiguous on the buffer device")
    if any(y.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (x, ids, p, w1, w2)):
        raise ValueError("y must not overlap BF16 inputs")
    if stats is not None:
        if (
            stats.dtype != torch.int32
            or stats.device != b._device
            or tuple(stats.shape) != (e,)
            or not stats.is_contiguous()
        ):
            raise ValueError("Cumulative stats must be contiguous local-expert int32 on the buffer device")
        if any(stats.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (x, ids, p, w1, w2, y)):
            raise ValueError("Cumulative stats must not overlap inputs or output")
    gradients = torch.is_grad_enabled() and any(t.requires_grad for t in (x, p, w1, w2))
    if gradients and not b.need_grad:
        raise ValueError("Use a SymmBuffer with need_grad=True for differentiable BF16 calls")
    mode = b.gmm1_save_type if gradients else "none"
    if activation == "situglu" and mode == "mxfp8":
        raise NotImplementedError("SiTU-GLU with gmm1_save_type='mxfp8' is not supported yet")
    config = Call(b.max_recv_tokens, beta1, mode, activation, beta2)
    with b._lock:
        runtime = b._runtime
        runtime.ensure_alive()
        caller = runtime.use_bf16_caller_stream()
        plan = prepare_plan(b, bs, activation, beta1, beta2, mode)
        if gradients:
            _, counts = BF16Function.apply(b, plan, config, x, ids, p, w1, w2, y)
        else:
            capture = getattr(b, "_timer_capture", None)
            if capture is not None:
                capture.claim()
            result = plan.forward(y, x, ids, p, w1, w2, config.max_recv, beta1, mode, capture is not None, False,
                                  activation, beta2)
            if capture is not None:
                capture.forward = result.timer
            counts = result.counts
        if stats is not None:
            stats.add_(counts)
            stats.record_stream(caller)
