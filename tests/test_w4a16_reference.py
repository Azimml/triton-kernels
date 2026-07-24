"""
CPU reference-math tests for the W4A16 (4-bit weight-only) reference module.

These exercise `reference/w4a16_reference.py`, which is pure PyTorch and defines
the exact numerics the Triton `w4a16_gemm` kernel must match. Everything here
runs on CPU (no GPU / Triton), so it is part of the always-on portable suite.
"""

import pytest
import torch

from reference.w4a16_reference import (
    dequantize_int4,
    pack_int4,
    quantize_weight_int4_grouped,
    unpack_int4,
)


class TestPackUnpackRoundTrip:
    """Packing 4-bit nibbles into int32 and back must be lossless in [0, 15]."""

    @pytest.mark.parametrize("K", [8, 64, 128, 256])
    @pytest.mark.parametrize("N", [1, 16, 128])
    def test_round_trip_identity(self, K: int, N: int):
        q = torch.randint(0, 16, (K, N), dtype=torch.int32)
        packed = pack_int4(q)
        restored = unpack_int4(packed, K)
        assert torch.equal(q, restored)

    def test_packed_shape_is_k_over_eight(self):
        q = torch.randint(0, 16, (256, 32), dtype=torch.int32)
        packed = pack_int4(q)
        assert packed.shape == (256 // 8, 32)
        assert packed.dtype == torch.int32

    def test_nibble_order_low_k_in_low_bits(self):
        """Nibble j (K = 8*i + j) lives in bits [4j, 4j+3] of the int32."""
        # Column of 8 known values -> one int32 whose bytes we can read back.
        q = torch.arange(8, dtype=torch.int32).reshape(8, 1)  # [0, 1, ..., 7]
        packed = pack_int4(q)
        word = int(packed[0, 0].item())
        for j in range(8):
            assert (word >> (4 * j)) & 0xF == j

    def test_requires_k_divisible_by_eight(self):
        with pytest.raises(ValueError, match="divisible by 8"):
            pack_int4(torch.zeros(12, 4, dtype=torch.int32))


class TestGroupedQuantizeDequantize:
    """Grouped 4-bit quantize -> dequantize round-trip error is bounded."""

    @pytest.mark.parametrize("group_size", [64, 128])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_error_bounded_by_half_step(self, group_size: int, symmetric: bool):
        torch.manual_seed(0)
        K, N = 256, 64
        weight = torch.randn(K, N, dtype=torch.float32)
        packed, scales, zeros = quantize_weight_int4_grouped(
            weight, group_size=group_size, symmetric=symmetric
        )
        restored = dequantize_int4(packed, scales, zeros, group_size, K)

        # Each element quantizes to the nearest of 16 levels within its group, so
        # the reconstruction error is bounded by half that group's step (scale).
        g = torch.arange(K) // group_size
        step = scales.float()[g]  # (K, N) per-element scale
        assert torch.all((weight - restored).abs() <= step / 2 + 1e-4)

    def test_output_formats(self):
        K, N = 128, 32
        weight = torch.randn(K, N, dtype=torch.float16)
        packed, scales, zeros = quantize_weight_int4_grouped(weight, group_size=64)
        assert packed.shape == (K // 8, N)
        assert packed.dtype == torch.int32
        assert scales.shape == (K // 64, N)
        assert zeros.shape == (K // 64, N)

    def test_symmetric_uses_fixed_zero_point(self):
        """Symmetric quantization pins every group's zero-point at 8."""
        weight = torch.randn(128, 16, dtype=torch.float32)
        _, _, zeros = quantize_weight_int4_grouped(weight, group_size=64, symmetric=True)
        assert torch.all(zeros == 8.0)

    def test_rejects_group_size_not_dividing_k(self):
        with pytest.raises(ValueError, match="divisible by group_size"):
            quantize_weight_int4_grouped(torch.randn(100, 8), group_size=64)
