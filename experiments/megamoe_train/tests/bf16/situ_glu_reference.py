#!/usr/bin/env python3
"""SiTU-GLU (Kimi K3) golden reference for the MegaMoe BF16 pipeline.

Definitions (matching kernels/mega_moe_bf16 epilogues):

    softcap(x, b) = b * tanh(x / b)
    A(g) = softcap(g, beta_gate) * sigmoid(g)
    B(u) = softcap(u, beta_up)
    y    = A(g) * B(u)                       |y| <= beta_gate * beta_up
    A'(g) = (1 - t1^2) * sig + beta_gate * t1 * sig * (1 - sig)
            with t1 = tanh(g / beta_gate), sig = sigmoid(g)

Rounding conventions mirror the eager reference graph (BigOp golden /
torch eager small-op graph): every elementwise op materializes BF16, so
the forward and backward goldens below apply one BF16 rounding per op
(see situ_glu_forward / situ_glu_grad). The FP32 helpers (situ_gate_act /
situ_up_act / situ_gate_grad / situ_up_grad) keep the unrounded formulas
for derivative checks.

All inputs are float32 numpy arrays; outputs are float32 with BF16 values
(representable exactly).
"""

from __future__ import annotations

import numpy as np

KIMI_K3_BETA_GATE = 4.0
KIMI_K3_BETA_UP = 25.0


def float32_to_bf16_bits(data: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(data, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float32(data: np.ndarray) -> np.ndarray:
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def bf16_round(data: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(float32_to_bf16_bits(data))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # exp overflow for very negative x saturates sigmoid to 0, matching the
    # kernel's saturating vector Exp; silence the float32 warning.
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x.astype(np.float32))


def situ_gate_act(
    gate: np.ndarray, beta_gate: float = KIMI_K3_BETA_GATE
) -> np.ndarray:
    """A(g) = beta_gate * tanh(g / beta_gate) * sigmoid(g), FP32."""
    g = np.asarray(gate, dtype=np.float32)
    return (
        np.float32(beta_gate) * _tanh(g / np.float32(beta_gate)) * _sigmoid(g)
    )


def situ_up_act(up: np.ndarray, beta_up: float = KIMI_K3_BETA_UP) -> np.ndarray:
    """B(u) = beta_up * tanh(u / beta_up), FP32."""
    u = np.asarray(up, dtype=np.float32)
    return np.float32(beta_up) * _tanh(u / np.float32(beta_up))


def situ_glu_forward(
    gate: np.ndarray,
    up: np.ndarray,
    beta_gate: float = KIMI_K3_BETA_GATE,
    beta_up: float = KIMI_K3_BETA_UP,
) -> np.ndarray:
    """Kernel/golden forward: the eager per-op BF16 chain.

    The reference graph (beta*tanh(gate/beta)*sigmoid(gate) * beta_up*
    tanh(up/beta_up)) materializes every elementwise op as BF16; each R()
    below is one of those roundings.  A single-rounding FP32 chain differs
    by ~0.4% per element and fails the backward double-hundred gate.

    beta_up=None (reference implementation's unspecified linear_beta)
    keeps the up branch linear: the chain is the gate chain + R(A * up).
    """
    b1 = np.float32(beta_gate)
    d1 = bf16_round(gate / b1)
    t1 = bf16_round(_tanh(d1))
    a1 = bf16_round(b1 * t1)
    sg = bf16_round(_sigmoid(gate))
    a = bf16_round(a1 * sg)
    if beta_up is None:
        return bf16_round(a * up)
    b2 = np.float32(beta_up)
    e1 = bf16_round(up / b2)
    t2 = bf16_round(_tanh(e1))
    b = bf16_round(b2 * t2)
    return bf16_round(a * b)


def situ_gate_grad(
    gate: np.ndarray, beta_gate: float = KIMI_K3_BETA_GATE
) -> np.ndarray:
    """A'(g) in FP32 (see module docstring)."""
    g = np.asarray(gate, dtype=np.float32)
    b = np.float32(beta_gate)
    t1 = _tanh(g / b)
    sig = _sigmoid(g)
    return (1.0 - t1 * t1) * sig + b * t1 * sig * (1.0 - sig)


def situ_up_grad(up: np.ndarray, beta_up: float = KIMI_K3_BETA_UP) -> np.ndarray:
    """B'(u) = 1 - tanh(u / beta_up)^2 in FP32."""
    u = np.asarray(up, dtype=np.float32)
    t2 = _tanh(u / np.float32(beta_up))
    return 1.0 - t2 * t2


def situ_glu_grad(
    gate: np.ndarray,
    up: np.ndarray,
    d_h: np.ndarray,
    beta_gate: float = KIMI_K3_BETA_GATE,
    beta_up: float = KIMI_K3_BETA_UP,
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel-backward golden: the eager/autograd per-op BF16 chain.

    Every autograd node materializes BF16, so the chain is:
      forward: d1=R(g/b1) t1=R(tanh d1) a1=R(b1*t1) sg=R(sig g) A=R(a1*sg)
               e1=R(u/b2) t2=R(tanh e1) B=R(b2*t2)
      dA=R(G*B) dB=R(G*A)                       (MulBackward of y=A*B)
      dUp:   R(R(R(dB*b2)*(1-t2^2))/b2)         TanhBackward, DivBackward
      dGate: R(R(dA*sg)*sg*(1-sg))              SigmoidBackward (one node)
             + R(R(R(R(dA*sg)*b1)*(1-t1^2))/b1) TanhBackward, DivBackward
             summed and rounded once            (AddBackward)

    beta_up=None (reference linear_beta unspecified) keeps the up branch
    linear: dA = R(G*u), dUp = R(G*A), same gate branch.
    """
    b1 = np.float32(beta_gate)
    g = np.asarray(gate, dtype=np.float32)
    u = np.asarray(up, dtype=np.float32)
    g_out = np.asarray(d_h, dtype=np.float32)
    # rounded forward intermediates (gate chain, shared by both variants)
    d1 = bf16_round(g / b1)
    t1 = bf16_round(_tanh(d1))
    a1 = bf16_round(b1 * t1)
    sg = bf16_round(_sigmoid(g))
    a = bf16_round(a1 * sg)
    if beta_up is None:
        # linear up branch (reference linear_beta unspecified): y = A*u
        #   dA = R(G*u), dUp = R(G*A)
        d_a = bf16_round(g_out * u)
        d_up = bf16_round(g_out * a)
    else:
        b2 = np.float32(beta_up)
        e1 = bf16_round(u / b2)
        t2 = bf16_round(_tanh(e1))
        b = bf16_round(b2 * t2)
        # y-grad split
        d_a = bf16_round(g_out * b)
        d_b = bf16_round(g_out * a)
        # up branch
        d_t2 = bf16_round(d_b * b2)
        d_uge = bf16_round(d_t2 * (1.0 - t2 * t2))
        d_up = bf16_round(d_uge / b2)
    # gate branch (shared)
    d_a1 = bf16_round(d_a * sg)
    d_sg = bf16_round(d_a * a1)
    d_g_sig = bf16_round(d_sg * sg * (1.0 - sg))          # SigmoidBackward node
    d_t1 = bf16_round(d_a1 * b1)
    d_g_tanh = bf16_round(d_t1 * (1.0 - t1 * t1))          # TanhBackward node
    d_g_div = bf16_round(d_g_tanh / b1)                    # DivBackward
    d_gate = bf16_round(d_g_sig + d_g_div)
    return d_gate, d_up
