"""
Example: W4A16 (4-bit weight-only) GEMM via the PyTorch reference, on CPU.

Quantizes a weight matrix to grouped 4-bit, packs it, runs the reference GEMM
``y = x @ dequant(W_q)``, and compares against the full-precision matmul. This
uses the pure-PyTorch reference (`reference/w4a16_reference.py`), so it runs on
CPU without a GPU or Triton - it demonstrates the exact numerics the Triton
`w4a16_gemm` kernel is validated against.

Run (from the repo root):
    python examples/w4a16_reference_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from reference.w4a16_reference import (  # noqa: E402
    quantize_weight_int4_grouped,
    w4a16_reference,
)


def main() -> None:
    torch.manual_seed(0)
    M, K, N = 4, 512, 256
    group_size = 128

    x = torch.randn(M, K, dtype=torch.float32)
    weight = torch.randn(K, N, dtype=torch.float32)

    # Offline weight prep: FP -> grouped 4-bit + pack (GPTQ/AWQ style).
    packed, scales, zeros = quantize_weight_int4_grouped(weight, group_size=group_size)

    fp16_bytes = weight.numel() * 2
    int4_bytes = packed.numel() * 4  # int32 words, each holding 8 nibbles
    print(
        f"weight {tuple(weight.shape)}: FP16 {fp16_bytes / 1024:.1f} KiB -> "
        f"packed int4 {int4_bytes / 1024:.1f} KiB ({fp16_bytes / int4_bytes:.1f}x smaller)\n"
    )

    # Reference 4-bit GEMM vs full-precision reference.
    y_w4a16 = w4a16_reference(x, packed, scales, zeros, group_size)
    y_full = x @ weight

    rel = (y_w4a16 - y_full).norm() / y_full.norm()
    print(f"output {tuple(y_w4a16.shape)}")
    print(f"relative error vs FP matmul: {rel:.4f}")
    print("(~10% is expected on i.i.d. random data at 4 bits; real trained weights,")
    print("being smoother within a group, quantize considerably more accurately.)")


if __name__ == "__main__":
    main()
