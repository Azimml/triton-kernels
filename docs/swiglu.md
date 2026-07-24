# SwiGLU in Triton: a fused elementwise activation

SwiGLU (the Swish/SiLU-gated linear unit used in LLaMA and PaLM,
[arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) is the activation inside a
gated FFN:

```
y = silu(x @ W_gate) * (x @ W_up)     with   silu(z) = z * sigmoid(z)
```

The two projections are ordinary GEMMs (left to cuBLAS). The kernel in
`triton_kernels/swiglu.py` fuses just the activation stage that runs *after* them:

```
out = silu(gate) * up
```

## Why fuse it

Written out in PyTorch as `F.silu(gate) * up`, the `silu(gate)` step
materializes a full intermediate tensor in global memory: read `gate`, write
`silu`, then read `silu` and `up`, write `out`. Fusing collapses that to a single
pass - read `gate`, read `up`, write `out` (`3N` transfers) - eliminating one
full-size round-trip. The op is memory-bound (AI ~0.67 FLOP/byte at FP16), so the
traffic saved is the speedup; the measured kernel is ~1.6x over PyTorch (see the
[roofline analysis](ROOFLINE_ANALYSIS.md)).

The activation is computed in FP32 registers (inputs are upcast on load, the
result is cast back on store) so the SiLU is numerically clean regardless of the
FP16 storage dtype.

## The adaptive block-size heuristic

`_select_block_size(n_elements)` picks the elementwise `BLOCK_SIZE` on the host
(unit-tested on CPU in `tests/test_host_heuristics.py`):

| n_elements | BLOCK_SIZE |
|------------|-----------:|
| <= 4096    | 256        |
| <= 65536   | 512        |
| > 65536    | 1024       |

A single large fixed block (say 1024) launches too few programs for a small
tensor, leaving most SMs idle; shrinking the block for small inputs keeps enough
programs in the grid to fill the device. Large tensors cap at 1024, a good
balance of occupancy against per-program register pressure. All candidates are
powers of two, which the vectorized loads prefer. The heuristic is monotonic
non-decreasing in size, so the block never shrinks as the tensor grows.

## Biased variant

`swiglu_with_bias` adds optional gate/up biases inside the same fused pass
(`silu(gate + b_gate) * (up + b_up)`), indexing the bias by `offset % hidden_dim`
so it broadcasts across the flattened element grid. This is for linear layers
that carry a bias, without giving up the fusion.

## Correctness

Validated against `swiglu_torch` (`F.silu(gate) * up`) on GPU in
`tests/test_swiglu.py`, with the reference math checked on CPU in
`tests/test_reference_numerics.py`.
