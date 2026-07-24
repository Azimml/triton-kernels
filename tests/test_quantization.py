"""
Tests for quantization utilities.

Validates symmetric and asymmetric INT8 quantization, dequantization, and
round-trip error. These are pure-CPU numerics tests (no GPU / Triton).
"""

import pytest
import torch

from triton_kernels.quantization import (
    QMAX,
    QMIN,
    QuantizedLinear,
    QuantScheme,
    calculate_quantization_error,
    dequantize,
    quantize,
    quantize_asymmetric,
    quantize_symmetric,
    quantize_weight_per_channel,
    quantize_weight_per_channel_asymmetric,
)


class TestQuantizeSymmetric:
    """Test suite for symmetric quantization."""

    @pytest.mark.parametrize("shape", [(1024,), (256, 256), (1024, 4096), (4096, 11008)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_output_dtype(self, shape: tuple, dtype: torch.dtype):
        """Verify quantized output is INT8 and scale is FP32."""
        tensor = torch.randn(shape, dtype=dtype)
        quantized, scale = quantize_symmetric(tensor)

        assert quantized.dtype == torch.int8
        assert scale.dtype == torch.float32

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096)])
    def test_output_shape_per_tensor(self, shape: tuple):
        """Verify shapes for per-tensor quantization."""
        tensor = torch.randn(shape, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor, dim=None)

        assert quantized.shape == shape
        assert scale.shape == ()  # Scalar for per-tensor

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096)])
    def test_output_shape_per_channel(self, shape: tuple):
        """Verify shapes for per-channel quantization."""
        tensor = torch.randn(shape, dtype=torch.float16)

        # Per-output-channel (dim=1 for rows)
        quantized, scale = quantize_symmetric(tensor, dim=1)
        assert quantized.shape == shape
        assert scale.shape == (shape[0],)

        # Per-input-channel (dim=0 for columns)
        quantized, scale = quantize_symmetric(tensor, dim=0)
        assert quantized.shape == shape
        assert scale.shape == (shape[1],)

    def test_value_range(self):
        """Verify quantized values are in valid INT8 range."""
        tensor = torch.randn(1024, 1024, dtype=torch.float16) * 10
        quantized, _ = quantize_symmetric(tensor)

        assert quantized.min() >= -128
        assert quantized.max() <= 127

    def test_zero_tensor(self):
        """Test quantization of zero tensor."""
        tensor = torch.zeros(256, 256, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)

        assert (quantized == 0).all()
        # Scale should be 1 (not 0) to avoid division issues
        assert scale.item() == 1.0

    def test_uniform_tensor(self):
        """Test quantization of uniform value tensor."""
        tensor = torch.full((256, 256), 5.0, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)

        # All values should be the same (5 / scale rounded)
        assert (quantized == quantized[0, 0]).all()


class TestDequantize:
    """Test suite for dequantization."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_output_dtype(self, dtype: torch.dtype):
        """Verify dequantized output has correct dtype."""
        tensor = torch.randn(256, 256, dtype=torch.float32)
        quantized, scale = quantize_symmetric(tensor)
        restored = dequantize(quantized, scale, dtype=dtype)

        assert restored.dtype == dtype

    def test_shape_preserved(self):
        """Verify dequantization preserves shape."""
        shapes = [(1024,), (256, 256), (4096, 11008)]
        for shape in shapes:
            tensor = torch.randn(shape, dtype=torch.float16)
            quantized, scale = quantize_symmetric(tensor)
            restored = dequantize(quantized, scale)

            assert restored.shape == shape


class TestRoundTrip:
    """Test suite for quantization round-trip error."""

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096), (4096, 4096)])
    def test_round_trip_error_per_tensor(self, shape: tuple):
        """Test round-trip error is within acceptable bounds for per-tensor."""
        tensor = torch.randn(shape, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)

        # For 8-bit symmetric quantization, max error ≈ scale / 2
        # Relative error should be small for normally distributed data
        error = calculate_quantization_error(tensor, quantized, scale)

        # SNR should be > 30 dB for 8-bit quantization
        assert error["snr_db"] > 30, f"SNR too low: {error['snr_db']:.1f} dB"

        # Relative error should be < 5%
        assert error["relative_error_pct"] < 5, (
            f"Relative error too high: {error['relative_error_pct']:.1f}%"
        )

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096)])
    def test_round_trip_error_per_channel(self, shape: tuple):
        """Test round-trip error for per-channel quantization."""
        tensor = torch.randn(shape, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor, dim=1)

        error = calculate_quantization_error(tensor, quantized, scale, dim=0)

        # Per-channel should have even better accuracy
        assert error["snr_db"] > 35, f"SNR too low: {error['snr_db']:.1f} dB"

    def test_small_values(self):
        """Test quantization of small values."""
        tensor = torch.randn(256, 256, dtype=torch.float16) * 1e-3
        quantized, scale = quantize_symmetric(tensor)
        restored = dequantize(quantized, scale, dtype=torch.float16)

        # Should still maintain reasonable relative accuracy
        # Use float32 for computation to avoid float16 precision issues (1e-10 is 0 in float16)
        tensor_f32 = tensor.float()
        restored_f32 = restored.float()
        # Only consider non-zero values to avoid 0/0 = NaN
        nonzero_mask = tensor_f32.abs() > 1e-6
        rel_error = (tensor_f32[nonzero_mask] - restored_f32[nonzero_mask]).abs() / tensor_f32[
            nonzero_mask
        ].abs()
        # Allow larger relative error for very small values
        assert rel_error.median() < 0.1

    def test_large_values(self):
        """Test quantization of large values."""
        tensor = torch.randn(256, 256, dtype=torch.float16) * 100
        quantized, scale = quantize_symmetric(tensor)

        error = calculate_quantization_error(tensor, quantized, scale)
        assert error["snr_db"] > 30


class TestQuantizeWeightPerChannel:
    """Test suite for weight-specific quantization."""

    @pytest.mark.parametrize(
        "out_features,in_features",
        [
            (256, 256),
            (4096, 4096),
            (4096, 11008),
            (11008, 4096),
        ],
    )
    def test_weight_shapes(self, out_features: int, in_features: int):
        """Test quantization of various weight matrix shapes."""
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        weight_int8, scale = quantize_weight_per_channel(weight)

        assert weight_int8.shape == (out_features, in_features)
        assert weight_int8.dtype == torch.int8
        assert scale.shape == (out_features,)
        assert scale.dtype == torch.float32

    def test_different_scales_per_row(self):
        """Verify each row can have different scale."""
        # Create weight with rows of different magnitudes
        weight = torch.zeros(4, 16, dtype=torch.float16)
        weight[0] = torch.randn(16) * 1
        weight[1] = torch.randn(16) * 10
        weight[2] = torch.randn(16) * 100
        weight[3] = torch.randn(16) * 0.1

        weight_int8, scale = quantize_weight_per_channel(weight)

        # Scales should vary significantly
        scale_ratios = scale.max() / scale.min()
        assert scale_ratios > 10, "Scales should vary across rows"

    def test_per_channel_beats_per_tensor_on_heterogeneous_rows(self):
        """
        When rows span very different magnitudes, a single per-tensor scale is
        dominated by the largest row and crushes the small ones to a few INT8
        codes. A per-channel scale sizes each row independently, so it must
        achieve strictly higher SNR on this pathological weight.
        """
        torch.manual_seed(0)
        weight = torch.stack(
            [
                torch.randn(512) * 100.0,  # large-magnitude row
                torch.randn(512) * 1.0,
                torch.randn(512) * 0.01,  # tiny row, starved by a shared scale
                torch.randn(512) * 10.0,
            ]
        )

        q_pt, s_pt = quantize_symmetric(weight, dim=None)
        err_pt = calculate_quantization_error(weight, q_pt, s_pt)

        q_pc, s_pc = quantize_weight_per_channel(weight)
        err_pc = calculate_quantization_error(weight, q_pc, s_pc, dim=0)

        assert err_pc["snr_db"] > err_pt["snr_db"] + 3.0, (
            f"per-channel {err_pc['snr_db']:.1f} dB vs per-tensor {err_pt['snr_db']:.1f} dB"
        )


class TestQuantizedLinear:
    """Test suite for QuantizedLinear module."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        in_features, out_features = 256, 128
        batch_size = 4

        # Create and quantize
        linear = QuantizedLinear(in_features, out_features)
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        linear.quantize_weights(weight)

        # Forward pass
        x = torch.randn(batch_size, in_features, dtype=torch.float16)
        y = linear(x)

        assert y.shape == (batch_size, out_features)
        assert y.dtype == torch.float16

    def test_from_linear(self):
        """Test conversion from nn.Linear."""
        in_features, out_features = 256, 128

        # Create pretrained linear
        original = torch.nn.Linear(in_features, out_features, bias=False)

        # Convert to quantized
        quantized = QuantizedLinear.from_linear(original)

        # Compare outputs
        x = torch.randn(4, in_features)
        y_original = original(x)
        y_quantized = quantized(x)

        # Should be close (within quantization error)
        torch.testing.assert_close(y_original, y_quantized, rtol=0.05, atol=0.05)

    def test_with_bias(self):
        """Test QuantizedLinear with bias."""
        in_features, out_features = 256, 128

        original = torch.nn.Linear(in_features, out_features, bias=True)
        quantized = QuantizedLinear.from_linear(original)

        x = torch.randn(4, in_features)
        y_original = original(x)
        y_quantized = quantized(x)

        torch.testing.assert_close(y_original, y_quantized, rtol=0.05, atol=0.05)

    def test_not_quantized_error(self):
        """Test that forward raises error if not quantized."""
        linear = QuantizedLinear(256, 128)
        x = torch.randn(4, 256)

        with pytest.raises(RuntimeError, match="not quantized"):
            linear(x)

    def test_llm_shapes(self):
        """Test with typical LLM weight shapes."""
        # Hidden to hidden (attention projections)
        shapes = [
            (4096, 4096),  # LLaMA 7B Q/K/V/O
            (4096, 11008),  # LLaMA 7B gate/up
            (11008, 4096),  # LLaMA 7B down
        ]

        for out_features, in_features in shapes:
            linear = QuantizedLinear(in_features, out_features)
            weight = torch.randn(out_features, in_features, dtype=torch.float16)
            linear.quantize_weights(weight)

            x = torch.randn(1, 1024, in_features, dtype=torch.float16)
            y = linear(x)

            assert y.shape == (1, 1024, out_features)


class TestCalculateQuantizationError:
    """Test suite for error calculation."""

    def test_zero_error_identity(self):
        """Integer-valued tensor with unit scale reconstructs almost exactly."""
        # Integers in [-100, 100] with scale=1 are representable exactly in INT8,
        # so the only error is the float16 cast on the way back out.
        tensor = torch.randint(-100, 101, (256, 256)).to(torch.float32)
        quantized = tensor.round().clamp(-128, 127).to(torch.int8)
        scale = torch.tensor(1.0)

        error = calculate_quantization_error(tensor, quantized, scale)
        # Reconstruction is exact for this contrived case -> very high SNR.
        assert error["max_abs_error"] == 0.0
        assert error["snr_db"] > 100

    def test_error_metrics_reasonable(self):
        """Test that error metrics are in reasonable ranges."""
        tensor = torch.randn(256, 256, dtype=torch.float16)
        quantized, scale = quantize_symmetric(tensor)

        error = calculate_quantization_error(tensor, quantized, scale)

        assert "max_abs_error" in error
        assert "mean_abs_error" in error
        assert "relative_error_pct" in error
        assert "snr_db" in error

        # All values should be positive/non-negative
        assert error["max_abs_error"] >= 0
        assert error["mean_abs_error"] >= 0
        assert error["relative_error_pct"] >= 0
        # SNR should be positive for any meaningful quantization
        assert error["snr_db"] > 0


class TestQuantizeAsymmetric:
    """Test suite for asymmetric (affine / zero-point) INT8 quantization."""

    @pytest.mark.parametrize("shape", [(1024,), (256, 256), (1024, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_output_dtypes(self, shape: tuple, dtype: torch.dtype):
        """Quantized tensor is INT8, scale is FP32, zero-point is INT8."""
        tensor = torch.randn(shape, dtype=dtype)
        quantized, scale, zero_point = quantize_asymmetric(tensor)

        assert quantized.dtype == torch.int8
        assert scale.dtype == torch.float32
        assert zero_point.dtype == torch.int8

    def test_value_range(self):
        """Quantized values stay inside the signed INT8 range."""
        tensor = torch.randn(512, 512) * 10
        quantized, _, zero_point = quantize_asymmetric(tensor)

        assert quantized.min() >= QMIN
        assert quantized.max() <= QMAX
        assert zero_point.min() >= QMIN
        assert zero_point.max() <= QMAX

    @pytest.mark.parametrize("shape", [(256, 256), (1024, 4096)])
    def test_per_channel_shapes(self, shape: tuple):
        """Per-channel scale and zero-point have the reduced-dim shape."""
        tensor = torch.randn(shape, dtype=torch.float16)

        quantized, scale, zero_point = quantize_asymmetric(tensor, dim=1)
        assert quantized.shape == shape
        assert scale.shape == (shape[0],)
        assert zero_point.shape == (shape[0],)

    def test_round_trip_per_tensor(self):
        """Round-trip error is bounded by ~half a quantization step."""
        tensor = torch.randn(512, 512, dtype=torch.float32)
        quantized, scale, zero_point = quantize_asymmetric(tensor)
        restored = dequantize(quantized, scale, dtype=torch.float32, zero_point=zero_point)

        # Max error of correct affine quantization is <= scale / 2 (+ fp slack).
        max_err = (tensor - restored).abs().max().item()
        assert max_err <= scale.item() / 2 + 1e-4

    def test_round_trip_per_channel(self):
        """Per-channel asymmetric round-trips accurately."""
        tensor = torch.randn(128, 512, dtype=torch.float32)
        quantized, scale, zero_point = quantize_asymmetric(tensor, dim=1)
        restored = dequantize(quantized, scale, dtype=torch.float32, dim=0, zero_point=zero_point)
        # Per-row error bounded by that row's scale / 2.
        row_err = (tensor - restored).abs().amax(dim=1)
        assert torch.all(row_err <= scale / 2 + 1e-4)

    def test_zero_is_representable(self):
        """A true zero maps back to (near) zero under the affine scheme."""
        # Range is forced to include 0, so an actual 0.0 entry must dequantize
        # back to within half a step of 0.0 (no systematic bias at zero).
        tensor = torch.rand(256, 256) * 5.0 + 2.0  # all positive, min > 0
        tensor[0, 0] = 0.0  # plant a real zero
        quantized, scale, zero_point = quantize_asymmetric(tensor)
        restored = dequantize(quantized, scale, dtype=torch.float32, zero_point=zero_point)
        assert abs(restored[0, 0].item()) <= scale.item() / 2 + 1e-4

    @pytest.mark.parametrize("offset", [1.0, 5.0, 20.0])
    def test_beats_symmetric_on_skewed_data(self, offset: float):
        """
        On distributions not centred at zero, asymmetric quantization achieves
        strictly higher SNR than symmetric, because it does not waste half its
        codes on an unused negative range.
        """
        # All-positive, offset-from-zero data (e.g. post-ReLU-like weights).
        tensor = torch.rand(1024, 1024, dtype=torch.float32) + offset

        q_sym, s_sym = quantize_symmetric(tensor)
        err_sym = calculate_quantization_error(tensor, q_sym, s_sym)

        q_asym, s_asym, zp = quantize_asymmetric(tensor)
        err_asym = calculate_quantization_error(tensor, q_asym, s_asym, zero_point=zp)

        # Asymmetric should be clearly better (>= 3 dB) as the offset grows the
        # gap widens, but even the mildest offset must win.
        assert err_asym["snr_db"] > err_sym["snr_db"] + 3.0, (
            f"offset={offset}: asym {err_asym['snr_db']:.1f} dB vs sym {err_sym['snr_db']:.1f} dB"
        )

    def test_symmetric_data_parity(self):
        """On zero-centred data, asymmetric is at least as good as symmetric."""
        tensor = torch.randn(1024, 1024, dtype=torch.float32)

        q_sym, s_sym = quantize_symmetric(tensor)
        err_sym = calculate_quantization_error(tensor, q_sym, s_sym)

        q_asym, s_asym, zp = quantize_asymmetric(tensor)
        err_asym = calculate_quantization_error(tensor, q_asym, s_asym, zero_point=zp)

        # Asymmetric uses the full 256 codes vs 255 for symmetric, so it should
        # never be materially worse on symmetric data.
        assert err_asym["snr_db"] >= err_sym["snr_db"] - 0.5

    def test_constant_tensor(self):
        """A constant tensor quantizes without NaN/inf (degenerate range)."""
        tensor = torch.full((128, 128), 3.5, dtype=torch.float32)
        quantized, scale, zero_point = quantize_asymmetric(tensor)
        restored = dequantize(quantized, scale, dtype=torch.float32, zero_point=zero_point)

        assert torch.isfinite(restored).all()


class TestQuantizeWeightPerChannelAsymmetric:
    """Test the per-output-channel asymmetric weight helper."""

    @pytest.mark.parametrize(
        "out_features,in_features",
        [(256, 256), (4096, 4096), (4096, 11008)],
    )
    def test_weight_shapes(self, out_features: int, in_features: int):
        weight = torch.randn(out_features, in_features, dtype=torch.float16)
        w_int8, scale, zero_point = quantize_weight_per_channel_asymmetric(weight)

        assert w_int8.shape == (out_features, in_features)
        assert w_int8.dtype == torch.int8
        assert scale.shape == (out_features,)
        assert zero_point.shape == (out_features,)

    def test_recovers_skewed_rows(self):
        """Each row is reconstructed within its own quantization step."""
        # Rows with different offsets/scales - the pathological case for a
        # single symmetric per-tensor scale.
        weight = torch.stack(
            [
                torch.rand(256) * 0.1 + 5.0,  # tight, high offset
                torch.rand(256) * 4.0 - 2.0,  # wide, centred
                torch.rand(256) * 0.01,  # tiny
            ]
        )
        w_int8, scale, zero_point = quantize_weight_per_channel_asymmetric(weight)
        restored = dequantize(w_int8, scale, dtype=torch.float32, dim=0, zero_point=zero_point)
        row_err = (weight - restored).abs().amax(dim=1)
        assert torch.all(row_err <= scale / 2 + 1e-4)


class TestQuantizeDispatch:
    """Test the unified `quantize()` dispatcher and QuantScheme enum."""

    def test_symmetric_returns_none_zero_point(self):
        tensor = torch.randn(256, 256)
        quantized, scale, zero_point = quantize(tensor, scheme=QuantScheme.SYMMETRIC)
        assert zero_point is None
        assert quantized.dtype == torch.int8

    def test_asymmetric_returns_zero_point(self):
        tensor = torch.randn(256, 256)
        quantized, scale, zero_point = quantize(tensor, scheme=QuantScheme.ASYMMETRIC)
        assert zero_point is not None
        assert zero_point.dtype == torch.int8

    def test_string_scheme_accepted(self):
        tensor = torch.randn(64, 64)
        _, _, zp_sym = quantize(tensor, scheme="symmetric")
        _, _, zp_asym = quantize(tensor, scheme="asymmetric")
        assert zp_sym is None and zp_asym is not None

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError):
            quantize(torch.randn(8, 8), scheme="nonexistent")

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="8-bit"):
            quantize_symmetric(torch.randn(8, 8), bits=4)
        with pytest.raises(ValueError, match="8-bit"):
            quantize_asymmetric(torch.randn(8, 8), bits=16)

    def test_out_of_range_dim_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            quantize_symmetric(torch.randn(8, 8), dim=2)
        with pytest.raises(ValueError, match="out of range"):
            quantize_asymmetric(torch.randn(8, 8), dim=5)

    def test_negative_dim_accepted(self):
        """A valid negative axis (e.g. -1) is accepted like torch semantics."""
        q, scale = quantize_symmetric(torch.randn(4, 8), dim=-1)
        assert scale.shape == (4,)


class TestQuantizedLinearAsymmetric:
    """QuantizedLinear with the asymmetric scheme, end-to-end on CPU."""

    def test_forward_shape_and_dtype(self):
        linear = QuantizedLinear(256, 128, scheme="asymmetric")
        weight = torch.rand(128, 256, dtype=torch.float16) + 1.0  # skewed
        linear.quantize_weights(weight)

        x = torch.randn(4, 256, dtype=torch.float16)
        y = linear(x)
        assert y.shape == (4, 128)
        assert y.dtype == torch.float16
        assert linear.scheme is QuantScheme.ASYMMETRIC

    def test_asymmetric_more_accurate_on_skewed_weights(self):
        """
        For a linear layer with skewed (all-positive) weights, the asymmetric
        QuantizedLinear reproduces the FP reference more faithfully than the
        symmetric one.
        """
        torch.manual_seed(0)
        linear = torch.nn.Linear(256, 128, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.rand(128, 256) * 2.0 + 1.0)  # all positive

        q_sym = QuantizedLinear.from_linear(linear, scheme="symmetric")
        q_asym = QuantizedLinear.from_linear(linear, scheme="asymmetric")

        x = torch.randn(16, 256)
        y_ref = linear(x)
        err_sym = (y_ref - q_sym(x)).abs().mean().item()
        err_asym = (y_ref - q_asym(x)).abs().mean().item()

        assert err_asym < err_sym, f"asym {err_asym:.5f} vs sym {err_sym:.5f}"

    def test_zero_point_buffer_shape_stable(self):
        """The zero-point buffer exists for both schemes (stable state_dict)."""
        sym = QuantizedLinear(64, 32, scheme="symmetric")
        asym = QuantizedLinear(64, 32, scheme="asymmetric")
        assert sym.weight_zero_point.shape == asym.weight_zero_point.shape == (32,)
        # Symmetric leaves it at zero after quantization.
        sym.quantize_weights(torch.randn(32, 64))
        assert torch.all(sym.weight_zero_point == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
