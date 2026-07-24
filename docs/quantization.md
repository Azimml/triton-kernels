# INT8 Weight Quantization: Symmetric and Asymmetric

The W8A16 path stores weights in INT8 and keeps activations in FP16, halving the
weight memory traffic that dominates LLM inference. This note documents the two
quantization schemes in `triton_kernels/quantization.py` and when to reach for
each. Both are pure PyTorch reference math (they run on CPU); the Triton GEMM
consumes the resulting INT8 weights plus their scale (and, for the asymmetric
scheme, zero-point).

## Symmetric (default)

A single scale, zero-point fixed at 0:

```
scale = max(|w|) / 127
q     = round(w / scale).clamp(-128, 127)
w'    = q * scale
```

Fast and simple, and near-optimal when the weights are roughly centred at zero,
which most trained linear weights are. This is `quantize_symmetric` /
`quantize_weight_per_channel` and remains the default everywhere.

## Asymmetric / affine (zero-point)

A scale **and** an integer zero-point that map the observed `[min, max]` onto the
full INT8 range:

```
scale = (max - min) / 255
zp    = round(-128 - min / scale).clamp(-128, 127)   # integer zero-point
q     = round(w / scale + zp).clamp(-128, 127)
w'    = (q - zp) * scale
```

The range is forced to include 0 so a true zero stays exactly representable
(TFLite / Krishnamoorthi affine convention). This is `quantize_asymmetric` /
`quantize_weight_per_channel_asymmetric`.

### Why it helps

Symmetric quantization is symmetric about zero, so for a distribution that is not
centred at zero (all-positive weights, gate/up projections after certain
activations, anything skewed) roughly half the 256 INT8 codes fall in an unused
range. Asymmetric quantization slides and stretches the representable window to
cover exactly `[min, max]`, spending all 256 codes on the values that actually
occur. The more skewed the distribution, the larger the win.

Measured on synthetic all-positive data (`U(0,1) + offset`, 1024x1024,
per-tensor, CPU reference), SNR of asymmetric vs symmetric:

| Data                    | Symmetric SNR | Asymmetric SNR |
|-------------------------|--------------:|---------------:|
| offset = 1 (mild skew)  |      ~50 dB   |      ~57 dB    |
| offset = 5              |      ~52 dB   |      ~58 dB    |
| offset = 20             |      ~53 dB   |      ~59 dB    |
| zero-centred `N(0,1)`   |      ~40 dB   |      ~40 dB    |

Asymmetric is consistently ~6 dB better on the skewed (offset-from-zero) data
and on par (it uses 256 vs 255 codes) on zero-centred data. The cost is one
extra INT8 zero-point per channel plus one integer subtract per element at
dequant time. The relative ordering (asymmetric strictly wins on skewed data,
ties on centred data) is asserted by the tests in
`tests/test_quantization.py` (`TestQuantizeAsymmetric`).

## API

```python
from triton_kernels import (
    QuantScheme,
    quantize,                                   # unified dispatcher
    quantize_symmetric,                         # -> (q, scale)
    quantize_asymmetric,                        # -> (q, scale, zero_point)
    quantize_weight_per_channel,                # symmetric, -> (q, scale)
    quantize_weight_per_channel_asymmetric,     # -> (q, scale, zero_point)
    dequantize,                                 # accepts optional zero_point
    QuantizedLinear,                            # accepts scheme=
)

# Per-output-channel asymmetric weight quantization
w_int8, scale, zp = quantize_weight_per_channel_asymmetric(weight)   # weight: (out, in)
w_restored = dequantize(w_int8, scale, dim=0, zero_point=zp)

# Or via the reference module (CPU), selecting the scheme:
layer = QuantizedLinear.from_linear(nn_linear, scheme="asymmetric")
```

`quantize(tensor, scheme=...)` returns a uniform `(q, scale, zero_point)` tuple
where `zero_point` is `None` for the symmetric scheme, so callers can branch on a
single return shape.

## Status and verification

- Reference math, the full API, and `QuantizedLinear` integration are verified on
  CPU (`tests/test_quantization.py`, no GPU required).
- Kernel-side execution of the asymmetric weights (feeding the zero-point into
  the Triton W8A16 GEMM) is **GPU-gated**: the existing `int8_gemm` kernel path
  consumes symmetric per-channel scales today. The asymmetric reference and
  storage format are complete and tested; wiring the zero-point subtraction into
  the INT8 GEMM kernel is the natural next step and runs only on a GPU.

## See also

- [`INT8_GEMM_INVESTIGATION.md`](INT8_GEMM_INVESTIGATION.md) - why W8A16 is a
  memory-traffic win, not a tensor-core compute win, and the tensor-core /
  layout bugs found along the way.
- [`w4a16.md`](w4a16.md) - the 4-bit weight-only path, which already carries a
  per-group zero-point end to end through the kernel.
