# Contributing

Thanks for your interest in improving `triton-kernels`. This is an educational
kernel collection, so contributions that make the kernels clearer, better
documented, or better validated are just as welcome as new kernels.

## Development setup

```bash
git clone https://github.com/Azimml/triton-kernels.git
cd triton-kernels
pip install -e ".[all]"     # runtime + dev/test/bench/lint extras
```

A CUDA (or ROCm) GPU is required to *run* the Triton kernels, but not to work on
the pure-PyTorch reference math, the quantization utilities, the host-side launch
heuristics, or the docs. The CPU-runnable test subset is the same one CI runs.

## Quality gate

Every change must pass the CPU-side gate before it is merged. This mirrors the
`lint` GitHub Actions workflow and the always-on test subset:

```bash
ruff check .
ruff format --check .
CUDA_VISIBLE_DEVICES="" pytest -q      # GPU kernel tests skip automatically
mypy                                   # advisory, non-blocking
```

The GPU correctness tests (`tests/test_rmsnorm.py`, `test_swiglu.py`,
`test_quantized_matmul.py`, `test_w4a16.py`, `test_moe_dispatch.py`) run only on
a machine with a GPU and are skipped elsewhere via `conftest.py`.

## Guidelines

- **Keep CPU tests CPU-runnable.** New tests should either be pure reference math
  / host heuristics (no GPU) or be GPU-gated with `@pytest.mark.skipif(not
  torch.cuda.is_available(), ...)`. Add the module to `_GPU_ONLY_TEST_MODULES` in
  `conftest.py` if it imports an autotuned kernel at collection time.
- **Every kernel needs a PyTorch reference** in `reference/` (or a `*_torch`
  function next to it) and a correctness test that compares against it.
- **Document the "why".** A kernel PR should explain the memory-traffic or
  roofline reasoning behind a tiling / fusion choice, not just what changed.
- **One logical change per commit**, with a
  [Conventional Commits](https://www.conventionalcommits.org/) message
  (`feat:`, `fix:`, `test:`, `docs:`, `perf:`, `refactor:`, `chore:`).

## Adding a kernel

1. Implement the Triton kernel in `triton_kernels/` (or `triton_kernels/moe/`).
2. Add a PyTorch reference and a correctness test comparing the two.
3. Export public symbols from `triton_kernels/__init__.py` (add GPU-only symbols
   to `_GPU_EXPORTS` so CPU-only hosts get a clear error rather than a crash).
4. Add a benchmark under `benchmarks/` and, ideally, a roofline note in `docs/`.
5. Add a row to the kernels table in `README.md`.

## Reporting issues

Please use the issue templates. A minimal reproduction (shapes, dtypes, GPU
model, Triton/PyTorch versions) makes kernel bugs far faster to diagnose.
