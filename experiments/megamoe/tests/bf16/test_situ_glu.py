# Copyright (c) 2026, wangrenlong

"""SiTU-GLU parameter contract tests and reference identity checks.

The math contract lives in situ_glu_reference.py (cherry-picked from the
20261124-fix-precis line): softcap(x, b) = b * tanh(x / b), A(g) =
softcap(g, beta_gate) * sigmoid(g), B(u) = softcap(u, beta_up) or the linear
u when beta_up is None/0. The Python API reuses activation_clamp, overloaded
by activation: swiglu keeps the dsv4 float clamp, situglu takes a
(gate, up) tuple of betas.

The NPU accuracy gate (kernel vs the FP32-through single-rounding
convention, double-hundred style) runs through the device bench; these tests
pin the Python contract and the reference math that gate compares against.
"""

import numpy as np
import pytest

from deep_ep_experimental.megamoe._bf16 import _resolve_activation
from situ_glu_reference import (
    KIMI_K3_BETA_GATE,
    KIMI_K3_BETA_UP,
    situ_gate_act,
    situ_gate_grad,
    situ_glu_forward,
    situ_glu_grad,
    situ_up_act,
    situ_up_grad,
)


@pytest.mark.parametrize(
    ("activation", "clamp", "expected"),
    [
        ("swiglu", None, ("swiglu", 0.0, 0.0)),
        ("swiglu", 7.0, ("swiglu", 7.0, 0.0)),
        ("SITU_GLU", (4.0, 25.0), ("situglu", 4.0, 25.0)),
        ("situ-glu", None, ("situglu", KIMI_K3_BETA_GATE, KIMI_K3_BETA_UP)),
        ("situglu", (5.0, None), ("situglu", 5.0, 0.0)),
        ("situglu", (5.0, 0), ("situglu", 5.0, 0.0)),
    ],
)
def test_resolve_activation(activation, clamp, expected):
    assert _resolve_activation(activation, clamp) == expected


def test_resolve_activation_rejects():
    with pytest.raises(NotImplementedError):
        _resolve_activation("relu", None)
    with pytest.raises(TypeError):
        _resolve_activation("situglu", 4.0)  # bare float: tuple required
    with pytest.raises(TypeError):
        _resolve_activation("swiglu", (4.0, 25.0))  # tuple reserved for situglu
    with pytest.raises(ValueError):
        _resolve_activation("situglu", (0.0, 25.0))  # beta_gate must be positive
    with pytest.raises(ValueError):
        _resolve_activation("situglu", (4.0, -1.0))  # beta_up is None/0 or positive
    with pytest.raises(ValueError):
        _resolve_activation("swiglu", -1.0)
    with pytest.raises(ValueError):
        _resolve_activation("swiglu", float("nan"))


def test_reference_formulas():
    """The FP32 helpers must satisfy the module-contract identities."""
    rng = np.random.default_rng(0)
    g = (rng.standard_normal(1024) * 8.0).astype(np.float32)
    u = (rng.standard_normal(1024) * 8.0).astype(np.float32)
    d_h = rng.standard_normal(1024).astype(np.float32)
    # softcap saturation bounds |y| <= beta_gate * beta_up
    y = situ_gate_act(g) * situ_up_act(u)
    assert float(np.max(np.abs(y))) <= KIMI_K3_BETA_GATE * KIMI_K3_BETA_UP + 1e-3
    # linear up keeps the raw branch
    y_lin = situ_gate_act(g) * u
    assert np.allclose(y_lin, situ_glu_forward(g, u, beta_up=None), atol=2e-2, rtol=2e-2)
    # derivative identities against a finite difference
    eps = np.float32(1e-3)
    fd_g = (situ_gate_act(g + eps) - situ_gate_act(g - eps)) / (2 * eps)
    assert np.allclose(fd_g, situ_gate_grad(g), atol=2e-2, rtol=2e-2)
    fd_u = (situ_up_act(u + eps) - situ_up_act(u - eps)) / (2 * eps)
    assert np.allclose(fd_u, situ_up_grad(u), atol=2e-2, rtol=2e-2)
    # grad chain returns finite dGate/dUp
    d_gate, d_up = situ_glu_grad(g, u, d_h)
    assert np.all(np.isfinite(d_gate)) and np.all(np.isfinite(d_up))
