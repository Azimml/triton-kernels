"""
Quantization utilities for INT8 inference.

Provides both symmetric and asymmetric (zero-point / affine) quantization for
weights, commonly used in W8A16 inference where weights are INT8 but activations
remain in FP16.

Quantization schemes
---------------------
Symmetric (zero-point fixed at 0):
    scale = max(|tensor|) / 127
    q     = round(tensor / scale).clamp(-128, 127)
    x     = q * scale

Asymmetric / affine (learned per-group zero-point):
    scale = (max - min) / 255
    zp    = round(-128 - min / scale).clamp(-128, 127)   # integer zero-point
    q     = round(tensor / scale + zp).clamp(-128, 127)
    x     = (q - zp) * scale

Symmetric wastes half its codes on distributions that are not centred at zero
(e.g. post-activation weights, GEGLU gates). Asymmetric quantization shifts the
representable window to cover [min, max] exactly, using the full 256 codes, which
lowers error on skewed distributions at the cost of storing one extra zero-point
per channel. Both give ~2x memory reduction for weights.

The dequantization happens in registers during matmul, so the memory traffic
reduction directly translates to speedup for memory-bound GEMMs.

References:
- LLM.int8() - https://arxiv.org/abs/2208.07339
- "Quantizing deep convolutional networks for efficient inference" (Krishnamoorthi,
  2018) - https://arxiv.org/abs/1806.08342 (affine / asymmetric scheme)
"""

from enum import Enum

import torch

# INT8 quantized range (signed 8-bit).
QMIN = -128
QMAX = 127


def _validate_bits_and_dim(tensor: torch.Tensor, bits: int, dim: int | None) -> None:
    """Validate the common ``bits`` / ``dim`` arguments with clear error messages.

    Raised as ``ValueError`` (not ``assert``) so the checks survive ``python -O``
    and give callers an actionable message rather than a bare traceback.
    """
    if bits != 8:
        raise ValueError(f"Only 8-bit quantization is supported, got bits={bits}")
    if dim is not None and not (-tensor.dim() <= dim < tensor.dim()):
        raise ValueError(
            f"dim={dim} is out of range for a {tensor.dim()}D tensor "
            f"(valid: {-tensor.dim()}..{tensor.dim() - 1}, or None for per-tensor)"
        )


class QuantScheme(str, Enum):
    """Weight-only quantization scheme.

    SYMMETRIC:  zero-point fixed at 0, uses the range [-127, 127].
    ASYMMETRIC: learned integer zero-point, uses the full range [-128, 127].
    """

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


def quantize_symmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization of tensor to INT8.

    Computes per-tensor or per-channel quantization using symmetric scheme:
    - scale = max(|tensor|) / (2^(bits-1) - 1)
    - quantized = round(tensor / scale).clamp(-128, 127)

    Args:
        tensor: Input tensor to quantize (typically FP16 or FP32 weights).
        bits: Number of bits for quantization (default: 8).
        dim: Dimension for per-channel quantization. If None, uses per-tensor.
             For weight matrices (out_features, in_features), use dim=0 for
             per-output-channel quantization.

    Returns:
        Tuple of (quantized_int8, scale):
        - quantized_int8: INT8 tensor of same shape as input
        - scale: FP32 scale factor(s) for dequantization

    Example:
        >>> weight = torch.randn(4096, 4096, dtype=torch.float16)
        >>> weight_int8, scale = quantize_symmetric(weight)
        >>> weight_fp16 = dequantize(weight_int8, scale)
        >>> error = (weight - weight_fp16).abs().max()
    """
    _validate_bits_and_dim(tensor, bits, dim)

    # Max value for symmetric INT8: 127 (we use symmetric range -127 to 127)
    qmax = QMAX

    # Compute scale
    if dim is None:
        # Per-tensor quantization
        max_val = tensor.abs().max()
        scale = max_val / qmax
    else:
        # Per-channel quantization
        max_val = tensor.abs().amax(dim=dim, keepdim=True)
        scale = max_val / qmax

    # Avoid division by zero
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # Quantize: round and clamp
    quantized = torch.round(tensor / scale).clamp(QMIN, QMAX).to(torch.int8)

    # Scale should be FP32 for numerical precision during dequantization
    scale = scale.float()

    # Squeeze scale if per-channel
    if dim is not None:
        scale = scale.squeeze(dim)

    return quantized, scale


def quantize_asymmetric(
    tensor: torch.Tensor,
    bits: int = 8,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric (affine, zero-point) quantization of tensor to INT8.

    Unlike the symmetric scheme, this maps the *observed* range [min, max] onto
    the full INT8 range [-128, 127] using an integer zero-point, so no codes are
    wasted when the distribution is not centred at zero:

        scale = (max - min) / 255
        zp    = round(-128 - min / scale).clamp(-128, 127)
        q     = round(tensor / scale + zp).clamp(-128, 127)

    Dequantization is ``x = (q - zp) * scale`` (see :func:`dequantize`).

    Args:
        tensor: Input tensor to quantize (typically FP16 or FP32 weights).
        bits: Number of bits for quantization (default: 8).
        dim: Dimension for per-channel quantization. If None, uses per-tensor.
             For weight matrices (out_features, in_features), use dim=1 for
             per-output-channel (per-row) quantization.

    Returns:
        Tuple of (quantized_int8, scale, zero_point):
        - quantized_int8: INT8 tensor of same shape as input
        - scale: FP32 scale factor(s)
        - zero_point: INT8 integer zero-point(s), same shape as scale

    Example:
        >>> weight = torch.rand(4096, 4096, dtype=torch.float16)  # skewed to [0, 1)
        >>> w_int8, scale, zp = quantize_asymmetric(weight, dim=1)
        >>> w_restored = dequantize(w_int8, scale, dim=0, zero_point=zp)
    """
    _validate_bits_and_dim(tensor, bits, dim)

    # Number of representable steps between the two INT8 extremes.
    q_range = QMAX - QMIN  # 255

    if dim is None:
        min_val = tensor.min()
        max_val = tensor.max()
    else:
        min_val = tensor.amin(dim=dim, keepdim=True)
        max_val = tensor.amax(dim=dim, keepdim=True)

    # Force the range to include 0 so that a true zero is exactly representable
    # (matches TFLite/Krishnamoorthi affine quantization).
    min_val = torch.minimum(min_val, torch.zeros_like(min_val))
    max_val = torch.maximum(max_val, torch.zeros_like(max_val))

    scale = (max_val - min_val) / q_range
    # Constant tensors (max == min) get a unit scale to avoid division by zero.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # Integer zero-point: the INT8 code that maps back to floating-point 0.
    zero_point = torch.round(QMIN - min_val / scale).clamp(QMIN, QMAX)

    quantized = torch.round(tensor / scale + zero_point).clamp(QMIN, QMAX).to(torch.int8)

    scale = scale.float()
    zero_point = zero_point.to(torch.int8)

    if dim is not None:
        scale = scale.squeeze(dim)
        zero_point = zero_point.squeeze(dim)

    return quantized, scale, zero_point


def quantize(
    tensor: torch.Tensor,
    scheme: QuantScheme | str = QuantScheme.SYMMETRIC,
    bits: int = 8,
    dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Unified quantization entry point dispatching on ``scheme``.

    Args:
        tensor: Input tensor to quantize.
        scheme: ``QuantScheme.SYMMETRIC`` or ``QuantScheme.ASYMMETRIC`` (strings
            ``"symmetric"`` / ``"asymmetric"`` are also accepted).
        bits: Number of bits (default: 8).
        dim: Per-channel dimension, or None for per-tensor.

    Returns:
        Tuple of (quantized_int8, scale, zero_point). ``zero_point`` is ``None``
        for the symmetric scheme and an INT8 tensor for the asymmetric scheme, so
        the return shape is uniform across both paths.
    """
    scheme = QuantScheme(scheme)
    if scheme is QuantScheme.SYMMETRIC:
        quantized, scale = quantize_symmetric(tensor, bits=bits, dim=dim)
        return quantized, scale, None
    quantized, scale, zero_point = quantize_asymmetric(tensor, bits=bits, dim=dim)
    return quantized, scale, zero_point


def dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float16,
    dim: int | None = None,
    zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Dequantize INT8 tensor back to floating point.

    Symmetric: ``x = q * scale``.
    Asymmetric: ``x = (q - zero_point) * scale`` when ``zero_point`` is supplied.

    Args:
        quantized: INT8 quantized tensor.
        scale: Scale factor(s) from quantization.
        dtype: Output dtype (default: float16).
        dim: Dimension along which scale was computed (for broadcasting).
             For weight matrices with per-output-channel quantization, use dim=0.
        zero_point: Optional INT8 zero-point(s) from asymmetric quantization. If
             ``None`` (default), symmetric dequantization is used. Broadcasts the
             same way as ``scale``.

    Returns:
        Dequantized tensor in specified dtype.

    Example:
        >>> weight_int8, scale = quantize_symmetric(weight_fp16)
        >>> weight_restored = dequantize(weight_int8, scale)
    """

    # Broadcast a per-channel 1D parameter against a 2D quantized tensor.
    def _broadcast(param: torch.Tensor) -> torch.Tensor:
        if param.dim() == 1 and quantized.dim() == 2:
            if dim is None or dim == 0:
                # Shape (out_features,) -> (out_features, 1)
                return param.unsqueeze(1)
            # Shape (in_features,) -> (1, in_features)
            return param.unsqueeze(0)
        return param

    # Convert to float for arithmetic. Subtract the (integer) zero-point *before*
    # scaling for the affine scheme, matching quantize_asymmetric.
    quantized_float = quantized.float()
    if zero_point is not None:
        quantized_float = quantized_float - _broadcast(zero_point.float())

    dequantized = quantized_float * _broadcast(scale)

    return dequantized.to(dtype)


def quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight matrix with per-output-channel *symmetric* scales.

    For a weight matrix of shape (out_features, in_features), computes
    one scale per output channel (row).

    This is the common scheme for W8A16 inference, providing good accuracy
    while allowing efficient dequantization during matmul. For skewed weight
    distributions, see :func:`quantize_weight_per_channel_asymmetric`.

    Args:
        weight: Weight tensor of shape (out_features, in_features).

    Returns:
        Tuple of (weight_int8, scales):
        - weight_int8: INT8 weights of shape (out_features, in_features)
        - scales: FP32 scales of shape (out_features,)

    Example:
        >>> W = torch.randn(4096, 4096, dtype=torch.float16)
        >>> W_int8, scales = quantize_weight_per_channel(W)
    """
    assert weight.dim() == 2, f"Expected 2D weight matrix, got {weight.dim()}D"
    return quantize_symmetric(weight, bits=8, dim=1)


def quantize_weight_per_channel_asymmetric(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize weight matrix with per-output-channel *asymmetric* (affine) params.

    Same layout as :func:`quantize_weight_per_channel` but uses a learned integer
    zero-point per output channel, mapping each row's observed [min, max] onto the
    full INT8 range. This is the better choice for rows whose values are not
    centred at zero (e.g. gate/up projections after certain activations), where
    it recovers accuracy the symmetric scheme leaves on the table.

    Args:
        weight: Weight tensor of shape (out_features, in_features).

    Returns:
        Tuple of (weight_int8, scales, zero_points):
        - weight_int8: INT8 weights of shape (out_features, in_features)
        - scales: FP32 scales of shape (out_features,)
        - zero_points: INT8 zero-points of shape (out_features,)

    Example:
        >>> W = torch.rand(4096, 4096, dtype=torch.float16)  # skewed to [0, 1)
        >>> W_int8, scales, zp = quantize_weight_per_channel_asymmetric(W)
        >>> W_restored = dequantize(W_int8, scales, dim=0, zero_point=zp)
    """
    assert weight.dim() == 2, f"Expected 2D weight matrix, got {weight.dim()}D"
    return quantize_asymmetric(weight, bits=8, dim=1)


def calculate_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    dim: int | None = None,
    zero_point: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Calculate quantization error metrics.

    Args:
        original: Original FP tensor.
        quantized: INT8 quantized tensor.
        scale: Scale factor(s).
        dim: Dimension for scale broadcasting.
        zero_point: Optional INT8 zero-point(s) for the asymmetric scheme.

    Returns:
        Dictionary with error metrics:
        - max_abs_error: Maximum absolute error
        - mean_abs_error: Mean absolute error
        - relative_error: Max relative error (percentage)
        - snr_db: Signal-to-noise ratio in dB
    """
    # Dequantize
    restored = dequantize(quantized, scale, dtype=original.dtype, dim=dim, zero_point=zero_point)

    # Calculate errors
    diff = (original - restored).float()
    original_float = original.float()

    max_abs_error = diff.abs().max().item()
    mean_abs_error = diff.abs().mean().item()

    # Relative error: only consider significant values (> 10% of max) to avoid
    # misleading high errors for small values. For 8-bit quantization, the max
    # relative error is ~scale/(2*value), so small values inherently have high
    # relative error even with perfect quantization.
    threshold = original_float.abs().max() * 0.1
    significant_mask = original_float.abs() > threshold
    if significant_mask.any():
        rel_error = (
            diff[significant_mask].abs() / original_float[significant_mask].abs()
        ).max().item() * 100
    else:
        rel_error = 0.0

    # Signal-to-noise ratio
    signal_power = (original_float**2).mean()
    noise_power = (diff**2).mean()
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "relative_error_pct": rel_error,
        "snr_db": snr_db,
    }


class QuantizedLinear(torch.nn.Module):
    """
    Linear layer with INT8 quantized weights.

    Stores weights in INT8 format and dequantizes on-the-fly during forward pass.
    This is a reference implementation - the actual kernel-level optimization
    happens in the Triton INT8 GEMM kernel.

    For W8A16 inference:
    - Weights: INT8 (2x memory reduction)
    - Activations: FP16
    - Compute: FP16 with FP32 accumulation

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias (default: False for LLM weights).
        scheme: ``QuantScheme.SYMMETRIC`` (default) or ``QuantScheme.ASYMMETRIC``.
            The asymmetric scheme stores a per-channel integer zero-point and is
            more accurate on skewed weight distributions.

    Example:
        >>> linear = QuantizedLinear(4096, 4096)
        >>> linear.quantize_weights(pretrained_weight)
        >>> y = linear(x)  # x is FP16, y is FP16
        >>> # Asymmetric variant for skewed weights:
        >>> linear = QuantizedLinear(4096, 4096, scheme="asymmetric")
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scheme: QuantScheme | str = QuantScheme.SYMMETRIC,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scheme = QuantScheme(scheme)

        # Register buffers for quantized weights
        self.register_buffer(
            "weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32))
        # Zero-point buffer is always present (zeros = symmetric) so state_dict
        # shape is stable regardless of scheme; it is only *used* when asymmetric.
        self.register_buffer("weight_zero_point", torch.zeros(out_features, dtype=torch.int8))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._quantized = False

    def quantize_weights(self, weight: torch.Tensor) -> None:
        """
        Quantize and store weights using the configured scheme.

        Args:
            weight: FP16 or FP32 weight tensor of shape (out_features, in_features).
        """
        assert weight.shape == (self.out_features, self.in_features)
        if self.scheme is QuantScheme.ASYMMETRIC:
            weight_int8, scale, zero_point = quantize_weight_per_channel_asymmetric(weight)
            self.weight_zero_point.copy_(zero_point)
        else:
            weight_int8, scale = quantize_weight_per_channel(weight)
            self.weight_zero_point.zero_()
        self.weight_int8.copy_(weight_int8)
        self.weight_scale.copy_(scale)
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        if not self._quantized:
            raise RuntimeError("Weights not quantized. Call quantize_weights() first.")

        # Dequantize weights (subtract zero-point only for the asymmetric scheme).
        zero_point = self.weight_zero_point if self.scheme is QuantScheme.ASYMMETRIC else None
        weight = dequantize(
            self.weight_int8, self.weight_scale, dtype=x.dtype, dim=0, zero_point=zero_point
        )

        # Compute matmul
        y = torch.nn.functional.linear(x, weight, self.bias)

        return y

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        scheme: QuantScheme | str = QuantScheme.SYMMETRIC,
    ) -> "QuantizedLinear":
        """
        Create QuantizedLinear from existing nn.Linear.

        Args:
            linear: Pretrained linear layer.
            scheme: Quantization scheme to use (see the class docstring).

        Returns:
            QuantizedLinear with quantized weights.
        """
        has_bias = linear.bias is not None
        quantized = cls(linear.in_features, linear.out_features, bias=has_bias, scheme=scheme)
        quantized.quantize_weights(linear.weight.data)
        if has_bias:
            quantized.bias.data.copy_(linear.bias.data)
        return quantized

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scheme={self.scheme.value}, "
            f"quantized={self._quantized}"
        )
