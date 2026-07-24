# Examples

Small, self-contained scripts that call the PyTorch **reference** implementations.
They are all **CPU-only** - no GPU or Triton required - so they run anywhere and
illustrate the exact numerics the Triton kernels are validated against.

Run any of them from the repo root:

```bash
python examples/quantization_roundtrip.py
python examples/w4a16_reference_demo.py
python examples/moe_reference_demo.py
```

| Script | What it shows |
|--------|---------------|
| [`quantization_roundtrip.py`](quantization_roundtrip.py) | INT8 weight quantization round-trip; symmetric vs asymmetric (affine) error metrics on a skewed weight |
| [`w4a16_reference_demo.py`](w4a16_reference_demo.py) | 4-bit grouped weight quantization + packing, the reference `y = x @ dequant(W_q)` GEMM, and the 4x weight-memory reduction |
| [`moe_reference_demo.py`](moe_reference_demo.py) | A Mixtral-style top-2/8-expert MoE forward pass and the resulting per-expert token load |

For the GPU kernels themselves (which require CUDA/ROCm), see the [Quick start](../README.md#quick-start)
section of the top-level README and the benchmark scripts under [`benchmarks/`](../benchmarks).
