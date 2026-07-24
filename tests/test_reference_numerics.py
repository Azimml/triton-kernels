"""
CPU numerics tests for the PyTorch *reference* implementations of the elementwise
kernels (RMSNorm, SwiGLU). These functions are pure PyTorch - they define the
ground truth the Triton kernels are validated against - so they run on CPU.

The reference functions live in modules that also declare Triton ``@jit`` kernels.
Those declarations are lazy (no GPU work at import), but Triton itself must be
importable, so the whole module is skipped if Triton is not installed. When it is
installed (as in this repo's environment) the tests run without a GPU.
"""

import pytest
import torch

pytest.importorskip("triton", reason="reference functions live alongside Triton @jit kernels")

from triton_kernels.rmsnorm import (  # noqa: E402  (import after importorskip)
    rmsnorm_residual_torch,
    rmsnorm_torch,
)
from triton_kernels.swiglu import swiglu_torch  # noqa: E402


class TestRMSNormReference:
    """RMSNorm reference matches the y = x * rsqrt(mean(x^2)+eps) * w definition."""

    @pytest.mark.parametrize("shape", [(4, 16), (2, 8, 64), (1, 128)])
    def test_matches_explicit_formula(self, shape: tuple):
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.float32)
        weight = torch.randn(shape[-1], dtype=torch.float32)
        eps = 1e-6

        y = rmsnorm_torch(x, weight, eps=eps)

        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = x * rms * weight
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)

    def test_identity_weight_normalizes_to_unit_rms(self):
        """With unit weight and no eps, each row's output has unit mean-square."""
        torch.manual_seed(1)
        x = torch.randn(8, 256, dtype=torch.float32)
        weight = torch.ones(256, dtype=torch.float32)
        y = rmsnorm_torch(x, weight, eps=0.0)
        ms = y.pow(2).mean(dim=-1)
        torch.testing.assert_close(ms, torch.ones_like(ms), rtol=1e-4, atol=1e-4)

    def test_shape_and_dtype_preserved(self):
        x = torch.randn(3, 5, 32, dtype=torch.float16)
        weight = torch.ones(32, dtype=torch.float16)
        y = rmsnorm_torch(x, weight)
        assert y.shape == x.shape
        assert y.dtype == torch.float16


class TestRMSNormResidualReference:
    """Fused residual reference equals RMSNorm applied to (x + residual)."""

    def test_equals_rmsnorm_of_sum(self):
        torch.manual_seed(2)
        x = torch.randn(4, 64, dtype=torch.float32)
        residual = torch.randn(4, 64, dtype=torch.float32)
        weight = torch.randn(64, dtype=torch.float32)

        fused = rmsnorm_residual_torch(x, residual, weight)
        separate = rmsnorm_torch(x + residual, weight)
        torch.testing.assert_close(fused, separate, rtol=1e-5, atol=1e-5)


class TestSwiGLUReference:
    """SwiGLU reference equals silu(gate) * up = gate * sigmoid(gate) * up."""

    @pytest.mark.parametrize("shape", [(16,), (4, 32), (2, 8, 64)])
    def test_matches_explicit_formula(self, shape: tuple):
        torch.manual_seed(3)
        gate = torch.randn(shape, dtype=torch.float32)
        up = torch.randn(shape, dtype=torch.float32)

        y = swiglu_torch(gate, up)

        expected = gate * torch.sigmoid(gate) * up
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)

    def test_zero_gate_gives_zero_output(self):
        """silu(0) = 0, so a zero gate zeroes the output regardless of up."""
        gate = torch.zeros(4, 16)
        up = torch.randn(4, 16)
        torch.testing.assert_close(swiglu_torch(gate, up), torch.zeros_like(up))

    def test_shape_and_dtype_preserved(self):
        gate = torch.randn(3, 128, dtype=torch.float16)
        up = torch.randn(3, 128, dtype=torch.float16)
        y = swiglu_torch(gate, up)
        assert y.shape == gate.shape
        assert y.dtype == torch.float16
