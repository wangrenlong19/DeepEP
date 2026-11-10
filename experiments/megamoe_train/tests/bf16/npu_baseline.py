# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026

"""NPU small-op reference with MegaMoE's declared numerical boundaries.

BigOp and the auxiliary contract reference share FP32 activation/derivative
arithmetic and ordered FP32 route sums, with BF16 GEMM inputs and outputs.
CPU FP64 mathematical truth is implemented independently in cpu_reference.py.
"""

import torch


BIGOP_REFERENCE_VERSION = "fp32-boundaries-v3-topk-dtype"


def _grouped_matmul(input_tokens, weight, tokens_per_expert):
    """BigOp split-M grouped GEMM; group_list contains counts, not prefix sums."""
    import torch_npu

    group_list = tokens_per_expert.to(torch.int64).to(input_tokens.device)
    # A single split-M weight retains its expert axis, including E=1.
    weight_list = (
        [weight.contiguous()]
        if weight.shape[0] == 1
        else [weight[expert].contiguous() for expert in range(weight.shape[0])]
    )
    return torch_npu.npu_grouped_matmul(
        [input_tokens.contiguous()],
        weight_list,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]


def _grouped_wgrad(input_tokens, grad_output, tokens_per_expert):
    """BigOp grouped weight gradient; preserve the transposed left-hand view."""
    import torch_npu

    group_list = tokens_per_expert.to(torch.int64).to(input_tokens.device)
    return torch_npu.npu_grouped_matmul(
        [input_tokens.T],
        [grad_output],
        group_list=group_list,
        split_item=3,
        group_type=2,
        group_list_type=1,
    )[0]


def _round_to_e4m3fn(value):
    """BigOp saturated E4M3FN arithmetic with ties-to-even rounding."""
    magnitude = value.abs().clamp(max=448.0)
    tiny = torch.finfo(torch.float32).tiny
    exponent = torch.floor(torch.log2(magnitude.clamp_min(tiny))).clamp(-6, 8)
    quantum = torch.where(magnitude < 2.0**-6, 2.0**-9, torch.exp2(exponent - 3.0))
    rounded = (torch.round(magnitude / quantum) * quantum).clamp(max=448.0)
    return torch.where(value < 0, -rounded, rounded)


def _mxfp8_saved_roundtrip(value):
    """BigOp BF16 reciprocal scaling, E4M3 RINT and BF16 decoding per 32 values."""
    if value.shape[-1] % 32:
        raise ValueError("MXFP8 saved-state baseline requires last dimension % 32 == 0")
    blocks = value.reshape(-1, value.shape[-1] // 32, 32)
    maximum = blocks.float().abs().amax(dim=-1, keepdim=True)
    exponent = (torch.floor(torch.log2(maximum.clamp_min(torch.finfo(torch.float32).tiny))) - 8.0).clamp(-127, 127)
    exponent = torch.where(maximum == 0, -127.0, exponent)
    scale = torch.exp2(exponent)
    reciprocal = torch.exp2(-exponent).to(torch.bfloat16)
    normalized = (blocks * reciprocal).to(torch.bfloat16).float()
    return (_round_to_e4m3fn(normalized) * scale).to(torch.bfloat16).reshape(value.shape)


def _activation_operands(gate_up, limit, raw=None):
    """Clip FP32 operands; MXFP8 derivative masks use raw forward decisions."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate_mask = up_mask = 1.0
    if 0 < limit < torch.finfo(torch.float32).max:
        raw_gate, raw_up = (gate_up if raw is None else raw).float().chunk(2, dim=-1)
        gate_mask = raw_gate <= limit
        up_mask = (raw_up >= -limit) & (raw_up <= limit)
        gate = torch.where(raw_gate > limit, limit, gate)
        up = torch.where(raw_up < -limit, -limit, up)
        up = torch.where(raw_up > limit, limit, up)
        # Decoding can cross a boundary even when the raw value was inside.
        # Clip the reconstructed operand, retaining the original gradient mask.
        gate, up = gate.clamp_max(limit), up.clamp(-limit, limit)
    return gate, up, gate_mask, up_mask


def _swiglu_fp32(gate_up, limit, raw=None):
    gate, up, _, _ = _activation_operands(gate_up, limit, raw)
    silu = gate / (1.0 + torch.exp(-gate))
    return silu * up


def _situ_fp32(gate_up, limit, raw=None, act_betas=(4.0, 25.0)):
    """SiTU-GLU forward in FP32: A(g)*B(u), softcap(x, b) = b*tanh(x/b).

    SiTU-GLU never takes the hard clamp (the runner rejects limit > 0), so
    ``limit`` and the MXFP8 ``raw`` masks are accepted for signature parity
    only. beta_up == 0 keeps the up branch linear.
    """
    gate, up, _, _ = _activation_operands(gate_up, limit, raw)
    beta_gate, beta_up = (float(value) for value in act_betas)
    gate_act = beta_gate * torch.tanh(gate / beta_gate) / (1.0 + torch.exp(-gate))
    up_act = up if beta_up == 0 else beta_up * torch.tanh(up / beta_up)
    return gate_act * up_act


def _situ_backward_fp32(grad_h, gate_up, limit, raw=None, act_betas=(4.0, 25.0)):
    """SiTU-GLU derivative in FP32; only dGate/dUp are BF16 outputs.

    dGate = dA * A'(g) with A'(g) = (1 - t1^2)*sig + beta_gate*t1*sig*(1 - sig);
    dUp = dA * A(g) * B'(u) with B'(u) = 1 - tanh(u/beta_up)^2 (linear up: 1).
    """
    gate, up, _, _ = _activation_operands(gate_up, limit, raw)
    beta_gate, beta_up = (float(value) for value in act_betas)
    t1 = torch.tanh(gate / beta_gate)
    sigmoid = 1.0 / (1.0 + torch.exp(-gate))
    gate_act = beta_gate * t1 * sigmoid
    da = grad_h.float() * (up if beta_up == 0 else beta_up * torch.tanh(up / beta_up))
    dg = da * ((1.0 - t1 * t1) * sigmoid + beta_gate * t1 * sigmoid * (1.0 - sigmoid))
    if beta_up == 0:
        du = da
    else:
        t2 = torch.tanh(up / beta_up)
        du = da * gate_act * (1.0 - t2 * t2)
    return torch.cat((dg, du), dim=-1).to(gate_up.dtype)


def _swiglu_backward_fp32(grad_h, gate_up, limit, raw=None):
    """Keep dH and the derivative in FP32; only dGate/dUp are BF16 outputs."""
    gate, up, gate_mask, up_mask = _activation_operands(gate_up, limit, raw)
    denominator = 1.0 + torch.exp(-gate)
    silu = gate / denominator
    sigmoid = torch.ones_like(gate) / denominator
    grad_h = grad_h.float()
    dg = (grad_h * up) * (sigmoid * (1.0 + gate * (1.0 - sigmoid))) * gate_mask
    du = grad_h * silu * up_mask
    return torch.cat((dg, du), dim=-1).to(gate_up.dtype)


def _activation_pair(activation):
    """Return the (forward, backward) FP32 activation helpers for a family."""
    if activation == "situglu":
        return _situ_fp32, _situ_backward_fp32
    if activation == "swiglu":
        return _swiglu_fp32, _swiglu_backward_fp32
    raise ValueError(f"activation must be swiglu or situglu, got {activation!r}")


@torch.no_grad()
def run_baseline(
    inputs,
    device,
    expert_start,
    save_mode="bf16",
    swiglu_limit=0.0,
    *,
    reference="megamoe",
    trace=None,
    activation="swiglu",
    act_betas=(4.0, 25.0),
):
    """Run the selected reference with public w1=[E,2F,H], w2=[E,H,F].

    Pack expert-major, then source-rank/token order. The reference names select
    reporting roles, not different rounding rules. Forward stays uncompressed;
    backward reuses saved Gate/Up or recomputes them for save=none.
    ``activation`` selects the FFN activation family (SiTU-GLU softcaps both
    branches and requires ``swiglu_limit <= 0``).
    """
    if save_mode not in ("none", "bf16", "mxfp8"):
        raise ValueError(f"Unresolved save mode: {save_mode}")
    if reference not in ("megamoe", "bigop"):
        raise ValueError(f"Unknown NPU reference: {reference}")
    act_forward, act_backward = _activation_pair(activation)
    if activation == "situglu" and swiglu_limit > 0:
        raise ValueError("swiglu_limit is incompatible with activation='situglu'")
    ids = inputs["ids"]
    indices = [(ids == expert_start + expert).nonzero(as_tuple=True) for expert in range(len(inputs["w1"]))]
    counts = torch.tensor([rows.numel() for rows, _ in indices], dtype=torch.int64, device="cpu")
    rows = torch.cat([rows for rows, _ in indices])
    slots = torch.cat([slots for _, slots in indices])
    result = {
        "ids": ids,
        "routes": [],
        "counts": counts,
        "reference": reference,
        "topk_weight_dtype": inputs["p"].dtype,
    }
    if trace is not None:
        trace.update(rows=rows.clone(), slots=slots.clone(), counts=counts.clone())
    if not rows.numel():
        result.update(
            dw1=torch.zeros(inputs["w1"].shape, dtype=torch.bfloat16, device="cpu"),
            dw2=torch.zeros(inputs["w2"].shape, dtype=torch.bfloat16, device="cpu"),
        )
        return result

    def tensor(value):
        return value.to(device=device, dtype=torch.bfloat16).contiguous()

    x, dy = tensor(inputs["x"][rows]), tensor(inputs["dy"][rows])
    p = inputs["p"][rows, slots, None].to(device=device).contiguous()
    weight1 = tensor(inputs["w1"].transpose(1, 2))
    weight2 = tensor(inputs["w2"].transpose(1, 2))
    forward_z = _grouped_matmul(x, weight1, counts)
    forward_h = act_forward(forward_z, swiglu_limit, act_betas=act_betas)
    forward_hp = (forward_h * p.float()).to(forward_z.dtype)
    y = _grouped_matmul(forward_hp, weight2, counts)

    raw_z = _grouped_matmul(x, weight1, counts) if save_mode == "none" else forward_z
    z = _mxfp8_saved_roundtrip(raw_z) if save_mode == "mxfp8" else raw_z
    raw = raw_z if save_mode == "mxfp8" else None
    h = act_forward(z, swiglu_limit, raw, act_betas=act_betas)
    hp = (h * p.float()).to(z.dtype)
    dhp = _grouped_matmul(dy, weight2.transpose(1, 2).contiguous(), counts)
    dw2 = _grouped_wgrad(hp, dy, counts).reshape_as(weight2).transpose(1, 2).contiguous()
    dp = (dhp.float() * h).sum(dim=-1).to(p.dtype)
    grad_swiglu = dhp.float() * p.float()
    dz = act_backward(grad_swiglu, z, swiglu_limit, raw, act_betas=act_betas)
    dx = _grouped_matmul(dz, weight1.transpose(1, 2).contiguous(), counts)
    dw1 = _grouped_wgrad(x, dz, counts).reshape_as(weight1).transpose(1, 2).contiguous()
    if trace is not None:
        # These are reference-side NPU intermediates, not fused-kernel dumps.
        trace.update(
            (name, value.detach().cpu().clone())
            for name, value in dict(raw_z=raw_z, forward_y=y, z=z, h=h, hp=hp, dhp=dhp, dp=dp, dz=dz).items()
        )
    result.update(dw1=dw1.cpu(), dw2=dw2.cpu())
    y, dx, dp = y.cpu(), dx.cpu(), dp.cpu()
    offset = 0
    for expert, (expert_rows, expert_slots) in enumerate(indices):
        stop = offset + expert_rows.numel()
        if stop > offset:
            result["routes"].append(
                {
                    "rows": expert_rows,
                    "slots": expert_slots,
                    "y": y[offset:stop],
                    "dx": dx[offset:stop],
                    "dp": dp[offset:stop],
                }
            )
        else:
            result["dw1"][expert].zero_()
            result["dw2"][expert].zero_()
        offset = stop
    return result


@torch.no_grad()
def unpermute_routes(buffer, ids, device, *, reference="megamoe", topk_weight_dtype=torch.bfloat16):
    """Combine transported routes using the selected numerical contract.

    buffer=[M,TopK,2H+1] carries BF16 y/dx and input-dtype dp exactly as FP32.
    Both reference roles combine y and dx in TopK order in FP32, then cast once.
    """
    if reference not in ("megamoe", "bigop"):
        raise ValueError(f"Unknown NPU reference: {reference}")
    m, topk = ids.shape
    hidden = (buffer.shape[-1] - 1) // 2
    if topk_weight_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Route-weight gradients require BF16 or FP32")
    result = {"dtopk_weights": buffer[:, :, -1].to(topk_weight_dtype).clone()}
    routes = buffer[:, :, : 2 * hidden].to(device=device, dtype=torch.bfloat16)
    y = torch.zeros((m, hidden), dtype=torch.float32, device=device)
    dx = torch.zeros((m, hidden), dtype=torch.float32, device=device)
    for slot in range(topk):
        y += routes[:, slot, :hidden].float()
        dx += routes[:, slot, hidden:].float()
    return {**result, "y": y.to(torch.bfloat16).cpu(), "dx": dx.to(torch.bfloat16).cpu()}
