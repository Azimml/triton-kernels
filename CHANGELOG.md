# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CPU reference-math test suites for the W4A16 int4 pack/unpack and grouped
  quantization reference, and for the MoE router / permute / expert-FFN
  reference, plus RMSNorm and SwiGLU reference numerics.
- `CONTRIBUTING.md` describing the CPU quality gate and how to add a kernel.
- Runnable examples under `examples/` that call the PyTorch references on CPU.
- `Makefile`, `.editorconfig`, `CITATION.cff`, and GitHub issue / PR templates.
- Design notes documenting the W4A16 autotune configuration space and the
  split-K decode dispatch heuristic.

## [0.1.0]

Initial public release.

### Added
- **RMSNorm** kernel (`rmsnorm`, `rmsnorm_residual_fused`) with FP32
  accumulation and a host-side `(BLOCK_SIZE, num_warps)` launch heuristic.
- **SwiGLU** fused activation kernel (`swiglu_fused`) with an adaptive
  block-size heuristic, plus an optional biased variant.
- **INT8 W8A16 GEMM** (`int8_gemm`, `Int8Linear`) with a Triton dequant path and
  a cuBLAS INT8 tensor-core fast path.
- **W4A16** 4-bit weight-only GEMM (`w4a16_gemm`) with fused group dequant in the
  K-loop and an autotuned split-K decode path.
- **Fused MoE dispatch** (`fused_moe_forward`): router, permute, block-scheduled
  grouped expert GEMM, and unpermute.
- **Quantization utilities**: symmetric and asymmetric (affine / zero-point)
  INT8 weight quantization with per-channel scales, error metrics, and a
  `QuantizedLinear` reference module.
- PyTorch reference implementations, correctness tests, benchmarks, and roofline
  analysis for every kernel.

[Unreleased]: https://github.com/Azimml/triton-kernels/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Azimml/triton-kernels/releases/tag/v0.1.0
