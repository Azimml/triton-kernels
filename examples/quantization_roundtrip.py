"""
Example: INT8 weight quantization round-trip on CPU.

Quantizes a synthetic weight matrix with both the symmetric and asymmetric
(affine / zero-point) schemes and prints the reconstruction error metrics. This
uses only the pure-PyTorch quantization utilities, so it runs anywhere - no GPU
or Triton required.

Run (from the repo root):
    python examples/quantization_roundtrip.py
"""

import sys
from pathlib import Path

# Allow running from a plain checkout (no `pip install -e .`) by putting the repo
# root on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from triton_kernels.quantization import (  # noqa: E402
    calculate_quantization_error,
    quantize_weight_per_channel,
    quantize_weight_per_channel_asymmetric,
)


def main() -> None:
    torch.manual_seed(0)
    out_features, in_features = 4096, 4096

    # A skewed, all-positive weight (the case where asymmetric quantization wins):
    # e.g. a projection whose rows are not centred at zero.
    weight = torch.rand(out_features, in_features, dtype=torch.float32) + 1.0

    print(f"weight: {tuple(weight.shape)}  range [{weight.min():.3f}, {weight.max():.3f}]\n")

    # Symmetric per-output-channel quantization.
    w_int8, scale = quantize_weight_per_channel(weight)
    sym = calculate_quantization_error(weight, w_int8, scale, dim=0)

    # Asymmetric per-output-channel quantization (adds a per-channel zero-point).
    w_int8_a, scale_a, zp = quantize_weight_per_channel_asymmetric(weight)
    asym = calculate_quantization_error(weight, w_int8_a, scale_a, dim=0, zero_point=zp)

    header = f"{'scheme':<12}{'SNR (dB)':>10}{'max |err|':>12}{'mean |err|':>12}"
    print(header)
    print("-" * len(header))
    for name, err in (("symmetric", sym), ("asymmetric", asym)):
        print(
            f"{name:<12}{err['snr_db']:>10.2f}"
            f"{err['max_abs_error']:>12.5f}{err['mean_abs_error']:>12.5f}"
        )

    gain = asym["snr_db"] - sym["snr_db"]
    print(f"\nAsymmetric is {gain:.2f} dB better on this skewed weight.")
    print("Memory: INT8 weights use 2x less than FP16, at 1 extra scale/channel")
    print("(and, for asymmetric, 1 extra INT8 zero-point/channel).")


if __name__ == "__main__":
    main()
