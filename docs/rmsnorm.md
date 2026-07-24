# RMSNorm in Triton: a memory-bound row reduction

RMSNorm (Zhang & Sennrich, [arXiv:1910.07467](https://arxiv.org/abs/1910.07467))
normalizes each row by its root-mean-square, without the mean-subtraction of
LayerNorm:

```
y = x * rsqrt(mean(x^2) + eps) * weight
```

Modern LLMs (LLaMA, Mistral, Qwen) use it in place of LayerNorm. This note
covers the two kernels in `triton_kernels/rmsnorm.py` and why they are shaped the
way they are.

## Why it is memory-bound

Per row of width `N` the kernel reads `x` and `weight` and writes `y` - about
`3N` element transfers - while doing on the order of `4N` FLOPs (square, reduce,
rsqrt, two multiplies). At FP16 that is an arithmetic intensity of roughly

```
AI ~= 4N / (3N * 2 bytes) ~= 0.67 FLOP/byte
```

which sits far to the left of the roofline ridge: the kernel is limited by HBM
bandwidth, not by arithmetic. The whole optimization strategy therefore targets
*memory traffic*, not compute.

## One program per row, FP32 accumulation

Each Triton program handles one full row. The sum of squares is accumulated in
FP32 even when the input is FP16, because the reduction over a wide hidden dim
(4096-8192) loses too much precision in FP16 and would shift the normalization
factor. The row is streamed in `BLOCK_SIZE`-wide chunks so an arbitrary hidden
dim is handled with a bounded register footprint.

## The launch heuristic

`_rmsnorm_launch_config(hidden_dim)` picks `(BLOCK_SIZE, num_warps)` on the host,
with no device probe, so it is portable across NVIDIA and AMD and needs no GPU to
compute (it is unit-tested on CPU in `tests/test_host_heuristics.py`):

- `BLOCK_SIZE = next_pow2(hidden_dim)`, capped at 8192 to bound register
  pressure. Rounding up to a power of two keeps the vectorized loads happy;
  capping means very wide rows are streamed in multiple chunks rather than
  blowing up per-program registers.
- `num_warps` scales with the block size (2 -> 4 -> 8 -> 16), so a wide row gets
  more warps to parallelize its reduction, following the reasoning of the Triton
  LayerNorm tutorial. Narrow rows stay cheap at 2 warps.

## The fused residual variant

`rmsnorm_residual_fused` computes `RMSNorm(x + residual) * weight` in a single
kernel. Doing the add inside the kernel avoids materializing the intermediate
`x + residual` in global memory:

- Naive (add, then norm): read `x`, read `residual`, write `tmp`; read `tmp`,
  write `y` -> ~`4N` transfers.
- Fused: read `x`, read `residual`, write `y` -> ~`3N` transfers.

For a memory-bound op that ~25% traffic reduction is close to a proportional
speedup, which is why the measured fused kernel lands around 6x over the PyTorch
add + norm (see the [roofline analysis](ROOFLINE_ANALYSIS.md)).

## Correctness

The kernels are validated against `rmsnorm_torch` / `rmsnorm_residual_torch`
(pure PyTorch, FP32 reduction) on GPU in `tests/test_rmsnorm.py`, and the
reference math itself is checked on CPU in `tests/test_reference_numerics.py`.
