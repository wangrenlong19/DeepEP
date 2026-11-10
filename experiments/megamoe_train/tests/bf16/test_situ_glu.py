# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""CPU-only SiTU-GLU unit tests (no NPU extension required).

Covers:
1. the Python-side activation resolver `_resolve_activation` (extracted from
   the training bridge module via ast, same technique as
   test_reference_contract);
2. the numpy golden in situ_glu_reference.py against float64 direct formulas,
   the |y| <= beta_gate * beta_up bound, and central finite differences for
   the gradient formulas;
3. bridge-level activation threading: forward passes the resolved family and
   betas to the executor, and backward replays them from the autograd
   context.

Run:  python test_situ_glu.py
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

TEST_DIR = Path(__file__).resolve().parent
BRIDGE_PY = (
    TEST_DIR.parent.parent / "python" / "deep_ep_experimental" / "megamoe_train" / "bf16.py"
)
sys.path.insert(0, str(TEST_DIR))

from situ_glu_reference import (  # noqa: E402
    KIMI_K3_BETA_GATE,
    KIMI_K3_BETA_UP,
    bf16_round,
    situ_glu_forward,
    situ_glu_grad,
    situ_gate_act,
    situ_gate_grad,
    situ_up_act,
    situ_up_grad,
)


def load_bridge_source():
    return ast.parse(BRIDGE_PY.read_text(encoding="utf-8"))


def load_resolver():
    """Import _resolve_activation without loading the NPU extension."""
    tree = load_bridge_source()
    wanted = {"_resolve_activation"}
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("_KIMI_SITU_BETA")
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "resolver", "exec"), namespace)
    return namespace["_resolve_activation"]


def load_bridge():
    """Import the autograd bridge without loading the NPU extension."""
    tree = load_bridge_source()
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MegaMoeBF16TrainFunction")
    namespace = {
        "torch": SimpleNamespace(autograd=SimpleNamespace(Function=object)),
        "once_differentiable": lambda func: func,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "autograd_bridge", "exec"), namespace)
    return namespace[cls.name]


class Context:
    def save_for_backward(self, *values):
        self.saved_tensors = values

    def mark_non_differentiable(self, *values):
        pass


class RecordingExecutor:
    """Record forward/backward activation arguments for the bridge contract."""

    def __init__(self):
        self.forward_args = None
        self.backward_args = None

    def forward(self, *args):
        self.forward_args = args
        return SimpleNamespace(
            y=object(), counts=object(), saved_state=object(), dispatch_output=object()
        )

    def backward(self, *args):
        self.backward_args = args
        return SimpleNamespace(dx=object(), dtopk_weights=object(), dw1=object(), dw2=object())


class RecordingRuntime:
    def validate_topk_weight_type(self, dtype, stage):
        self.stage = stage


class RecordingOperator:
    def __init__(self):
        self._executor = RecordingExecutor()
        self._runtime = RecordingRuntime()

    def _check_ready(self):
        pass


class Tensor:
    dtype = "bf16-mock"


class GradMock:
    def contiguous(self):
        return self


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.resolve = load_resolver()

    def test_default_is_swiglu_with_kimi_betas(self):
        name, bg, bu = self.resolve("swiglu", 0.0, None)
        self.assertEqual((name, bg, bu), ("swiglu", 4.0, 25.0))

    def test_situglu_normalizes_name_and_defaults(self):
        for spelling in ("situglu", "SiTU-GLU", "situ_glu", " SiTUGlu "):
            name, bg, bu = self.resolve(spelling, 0.0, None)
            self.assertEqual(name, "situglu")
            self.assertEqual((bg, bu), (4.0, 25.0))

    def test_situglu_rejects_swiglu_limit(self):
        with self.assertRaisesRegex(ValueError, "swiglu_limit"):
            self.resolve("situglu", 7.0, None)
        # limit=0 (or None-ish) stays legal
        self.assertEqual(self.resolve("situglu", 0.0, None)[0], "situglu")

    def test_situglu_custom_betas_and_validation(self):
        name, bg, bu = self.resolve("situglu", 0.0, (2.0, 30.0))
        self.assertEqual((name, bg, bu), ("situglu", 2.0, 30.0))
        with self.assertRaisesRegex(ValueError, "act_betas"):
            self.resolve("situglu", 0.0, (2.0,))
        with self.assertRaisesRegex(ValueError, "beta_gate"):
            self.resolve("situglu", 0.0, (0.0, 25.0))
        with self.assertRaisesRegex(ValueError, "beta_up"):
            self.resolve("situglu", 0.0, (4.0, -1.0))

    def test_situglu_linear_beta_unspecified(self):
        # Reference linear_beta semantics: beta_up None (or 0) keeps the up
        # branch linear and is normalized to the 0.0 sentinel end to end.
        name, bg, bu = self.resolve("situglu", 0.0, (4.0, None))
        self.assertEqual((name, bg, bu), ("situglu", 4.0, 0.0))
        name, bg, bu = self.resolve("situglu", 0.0, (4.0, 0))
        self.assertEqual((name, bg, bu), ("situglu", 4.0, 0.0))
        name, bg, bu = self.resolve("situglu", 0.0, (None, None))
        self.assertEqual((name, bg, bu), ("situglu", 4.0, 0.0))
        # swiglu ignores the sentinel too
        self.assertEqual(self.resolve("swiglu", 0.0, (None, None))[1:], (4.0, 0.0))

    def test_unknown_activation_rejected(self):
        with self.assertRaisesRegex(ValueError, "swiglu.*situglu|situglu.*swiglu"):
            self.resolve("geglu", 0.0, None)

    def test_swiglu_ignores_betas(self):
        name, bg, bu = self.resolve("swiglu", 5.0, (1.0, 2.0))
        self.assertEqual((name, bg, bu), ("swiglu", 1.0, 2.0))


class ReferenceFormulaTests(unittest.TestCase):
    """Golden math against independent float64 formulas."""

    def setUp(self):
        rng = np.random.default_rng(20261022)
        self.gate = rng.normal(0.0, 6.0, (256, 512)).astype(np.float32)
        self.up = rng.normal(0.0, 20.0, (256, 512)).astype(np.float32)

    def test_forward_matches_float64_direct_formula(self):
        g64 = self.gate.astype(np.float64)
        u64 = self.up.astype(np.float64)
        bg, bu = float(KIMI_K3_BETA_GATE), float(KIMI_K3_BETA_UP)
        expected = bg * np.tanh(g64 / bg) / (1.0 + np.exp(-g64)) * bu * np.tanh(u64 / bu)
        y = situ_glu_forward(self.gate, self.up)
        # The per-op BF16 chain compounds to at most ~2% relative error
        # against the float64 truth (measured max 1.84% on the test set).
        np.testing.assert_allclose(y, expected, rtol=2.5e-2, atol=1e-2)

    def test_output_bound_is_beta_product(self):
        y = situ_glu_forward(self.gate * 1000.0, self.up * 1000.0)  # deep saturation
        self.assertTrue(np.all(np.abs(y) <= KIMI_K3_BETA_GATE * KIMI_K3_BETA_UP))

    def test_output_values_are_bf16_representable(self):
        y = situ_glu_forward(self.gate, self.up)
        np.testing.assert_array_equal(y, bf16_round(y))

    def test_gate_gradient_matches_finite_differences(self):
        eps = np.float32(1e-2)
        bg = float(KIMI_K3_BETA_GATE)
        idx = (slice(0, 32), slice(0, 64))
        g = self.gate[idx]
        a_plus = situ_gate_act(g + eps, KIMI_K3_BETA_GATE)
        a_minus = situ_gate_act(g - eps, KIMI_K3_BETA_GATE)
        numeric = ((a_plus - a_minus) / np.float32(2.0 * eps)).astype(np.float64)
        analytic = situ_gate_grad(g, KIMI_K3_BETA_GATE).astype(np.float64)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-2, atol=2e-3)
        self.assertGreater(bg, 0)  # silence unused-var pattern for readers

    def test_full_grad_matches_finite_differences(self):
        # Finite-difference the UNROUNDED FP32 forward; differencing the
        # BF16-quantized golden directly is invalid because one quantization
        # step (up to 0.125 near |y|~16) dwarfs any eps=1e-2 perturbation.
        eps = np.float32(1e-3)
        idx = (slice(0, 16), slice(0, 64))
        g, u = self.gate[idx], self.up[idx]
        y = situ_gate_act(g) * situ_up_act(u)
        y_plus_g = situ_gate_act(g + eps) * situ_up_act(u)
        y_minus_g = situ_gate_act(g - eps) * situ_up_act(u)
        y_plus_u = situ_gate_act(g) * situ_up_act(u + eps)
        y_minus_u = situ_gate_act(g) * situ_up_act(u - eps)
        numeric_dg = (y_plus_g - y_minus_g) / np.float32(2.0 * eps)
        numeric_du = (y_plus_u - y_minus_u) / np.float32(2.0 * eps)
        np.testing.assert_allclose(situ_gate_grad(g) * situ_up_act(u), numeric_dg, rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(situ_gate_act(g) * situ_up_grad(u), numeric_du, rtol=2e-2, atol=2e-2)
        del y

    def test_linear_beta_forward_and_grad(self):
        # linear_beta unspecified: up branch linear, only the gate is
        # softcapped.  Chain forward == analytic A(g)*u within rounding;
        # analytic grads are dGate = A'(g)*u, dUp = A(g).
        idx = (slice(0, 32), slice(0, 128))
        g, u = self.gate[idx], self.up[idx]
        y = situ_glu_forward(g, u, beta_up=None)
        expected = situ_gate_act(g) * u
        np.testing.assert_allclose(y, expected, rtol=2.5e-2, atol=1e-2)
        rng = np.random.default_rng(11)
        d_h = rng.normal(0.0, 2.0, g.shape).astype(np.float32)
        d_gate, d_up = situ_glu_grad(g, u, d_h, beta_up=None)
        # finite-difference on the unrounded formula (loose vs the chain)
        eps = np.float32(1e-3)
        num_dg = ((situ_gate_act(g + eps) * u) - (situ_gate_act(g - eps) * u)) / np.float32(2.0 * eps)
        num_du = situ_gate_act(g)
        np.testing.assert_allclose(
            d_gate, d_h * num_dg, rtol=0.15, atol=0.02 * np.abs(d_h * num_dg).max()
        )
        np.testing.assert_allclose(
            d_up, d_h * num_du, rtol=0.15, atol=0.02 * np.abs(d_h * num_du).max()
        )

    def test_bf16_materialized_grad_close_to_fp32_analytic(self):
        # The per-op BF16 chain accumulates a few ulps of rounding per
        # element; near-zero analytic values make per-element relative error
        # meaningless, so gate on rtol plus an absolute floor scaled to the
        # tensor magnitude (chain noise stays within a few BF16 ulps of the
        # value scale; a wrong FORMULA produces O(1)-relative failures).
        idx = (slice(0, 32), slice(0, 128))
        g, u = self.gate[idx], self.up[idx]
        rng = np.random.default_rng(7)
        d_h = rng.normal(0.0, 2.0, g.shape).astype(np.float32)
        d_gate, d_up = situ_glu_grad(g, u, d_h)
        analytic_dg = d_h * situ_gate_grad(g) * situ_up_act(u)
        analytic_du = d_h * situ_gate_act(g) * situ_up_grad(u)
        np.testing.assert_allclose(d_gate, analytic_dg, rtol=0.15, atol=0.02 * np.abs(analytic_dg).max())
        np.testing.assert_allclose(d_up, analytic_du, rtol=0.15, atol=0.02 * np.abs(analytic_du).max())


@unittest.skipIf(torch is None, "CPU Torch is required; no NPU or ATK is needed")
class BridgeActivationThreadingTests(unittest.TestCase):
    """The bridge must carry activation settings from forward into backward."""

    def test_situglu_args_thread_forward_and_backward(self):
        bridge = load_bridge()
        op = RecordingOperator()
        ctx = Context()
        with patch.dict("os.environ", {}, clear=True):
            bridge.forward(
                ctx, op, Tensor(), *(Tensor() for _ in range(4)),
                65536, 0.0, "bf16", "situglu", 4.0, 25.0,
            )
            self.assertEqual(op._executor.forward_args[-3:], ("situglu", 4.0, 25.0))
            # Linear-beta sentinel flows to the executor verbatim.
            bridge.forward(
                ctx, op, Tensor(), *(Tensor() for _ in range(4)),
                65536, 0.0, "bf16", "situglu", 4.0, 0.0,
            )
            self.assertEqual(op._executor.forward_args[-3:], ("situglu", 4.0, 0.0))
            grads = bridge.backward(ctx, GradMock(), None)
        self.assertEqual(grads[0], None)
        self.assertEqual(grads[2], None)
        self.assertEqual(len(grads), 12)
        self.assertEqual(grads[-3:], (None, None, None))
        # The backward replays the ctx values, not a re-resolved default.
        self.assertEqual(op._executor.backward_args[-3:], ("situglu", 4.0, 0.0))

    def test_swiglu_defaults_thread_through(self):
        bridge = load_bridge()
        op = RecordingOperator()
        ctx = Context()
        with patch.dict("os.environ", {}, clear=True):
            bridge.forward(
                ctx, op, Tensor(), *(Tensor() for _ in range(4)),
                65536, 0.0, "mxfp8", "swiglu", 4.0, 25.0,
            )
            self.assertEqual(op._executor.forward_args[-3:], ("swiglu", 4.0, 25.0))
            bridge.backward(ctx, GradMock(), None)
        self.assertEqual(op._executor.backward_args[-3:], ("swiglu", 4.0, 25.0))


if __name__ == "__main__":
    unittest.main()
