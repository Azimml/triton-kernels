# Documentation index

Design notes and analyses for the kernels in this repository. Start with the
[top-level README](../README.md) for the overview and the kernels table.

## Per-kernel design notes

| Doc | Kernel(s) | Covers |
|-----|-----------|--------|
| [rmsnorm.md](rmsnorm.md) | RMSNorm, fused residual | Why it is memory-bound, FP32 reduction, the `(BLOCK_SIZE, num_warps)` launch heuristic, the residual-fusion traffic saving |
| [swiglu.md](swiglu.md) | SwiGLU activation | Activation fusion (eliminating the `silu` intermediate), the adaptive block-size heuristic, the biased variant |
| [quantization.md](quantization.md) | INT8 W8A16 utilities | Symmetric vs asymmetric (affine / zero-point) INT8 weight quantization and the accuracy comparison |
| [w4a16.md](w4a16.md) | W4A16 GEMM | 4-bit weight-only GEMM, load-once unpack, hoisted group scales, split-K decode dispatch, autotune config space |
| [moe_dispatch.md](moe_dispatch.md) | Fused MoE | Router / permute / grouped expert GEMM / unpermute pipeline and benchmarks |

## Analyses and postmortems

| Doc | Covers |
|-----|--------|
| [ROOFLINE_ANALYSIS.md](ROOFLINE_ANALYSIS.md) | The roofline methodology and measured bandwidth / speedups for every kernel |
| [INT8_GEMM_INVESTIGATION.md](INT8_GEMM_INVESTIGATION.md) | Why the first INT8 kernel was 3x slower than FP16, and what fixed it (tensor-core dtype and weight-layout bugs) |

## Figures

Roofline and benchmark plots live in [`figures/`](figures/) and are regenerated
by the scripts under [`../benchmarks/`](../benchmarks).
