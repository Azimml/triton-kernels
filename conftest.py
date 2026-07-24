"""
Pytest configuration for triton-kernels.

Some kernel modules apply ``@triton.autotune`` at import time, which makes
Triton eagerly initialize its GPU backend. On a host with no usable GPU driver
that raises during *collection*, before any ``skipif`` marker can take effect,
which would abort the whole run - including the pure-CPU numerics tests.

To keep the suite runnable everywhere, we skip collecting the GPU-only test
modules when no CUDA device (and therefore no Triton backend) is available. The
CPU-side tests (quantization numerics, host-side launch heuristics) do not import
any autotuned kernel module and always run.
"""

import torch

# Test modules that import Triton kernels which need a live GPU backend at import
# time. These are additionally guarded internally with `skipif(not cuda)`; the
# ignore here just prevents an import-time crash during collection on CPU hosts.
_GPU_ONLY_TEST_MODULES = [
    "tests/test_moe_dispatch.py",
    "tests/test_quantized_matmul.py",
    "tests/test_rmsnorm.py",
    "tests/test_swiglu.py",
    "tests/test_w4a16.py",
]

if not torch.cuda.is_available():
    # pytest reads this module-level list to skip collecting the named files.
    collect_ignore = list(_GPU_ONLY_TEST_MODULES)
