"""
triton-kernels: High-performance GPU kernels for LLM inference using OpenAI Triton.

This package provides fused and optimized kernels for common transformer operations.

Import behavior
---------------
The pure-PyTorch utilities (INT8 quantization reference, error metrics) import
unconditionally and work on CPU. The Triton-backed kernels are imported behind a
guard: on a host with no usable GPU driver, Triton raises at import time when it
tries to initialize its backend (e.g. from a module-level ``@triton.autotune``).
Rather than making the whole package unimportable, we degrade gracefully - the
kernel symbols are simply omitted from the namespace, so CPU-only workflows
(numerics tests, reference math, docs builds, linting) still work. Accessing a
kernel symbol that failed to load raises a clear error explaining a GPU is
required.
"""

__version__ = "0.1.0"

# Quantization utilities: pure PyTorch, always importable (no Triton/GPU).
from triton_kernels.quantization import (
    QuantizedLinear,
    QuantScheme,
    calculate_quantization_error,
    dequantize,
    quantize,
    quantize_asymmetric,
    quantize_symmetric,
    quantize_weight_per_channel,
    quantize_weight_per_channel_asymmetric,
)

# Symbols that require importing a Triton-backed module. Populated below when a
# GPU driver is available; left absent (and reported via __getattr__) otherwise.
_GPU_EXPORTS = [
    # RMSNorm
    "rmsnorm",
    "rmsnorm_torch",
    "rmsnorm_residual_fused",
    "rmsnorm_residual_torch",
    "TritonRMSNorm",
    # SwiGLU
    "swiglu_fused",
    "swiglu_with_bias",
    "swiglu_torch",
    "SwiGLU",
    # INT8 GEMM
    "int8_gemm",
    "int8_gemm_torch",
    "Int8Linear",
    # MoE Dispatch
    "moe_router",
    "moe_router_torch",
    "permute_tokens",
    "unpermute_tokens",
    "grouped_gemm",
    "expert_ffn_triton",
    "expert_ffn_torch",
    "fused_moe_forward",
    "fused_expert_ffn",
    # W4A16 GEMM
    "w4a16_gemm",
    "quantize_weight_int4_grouped",
    "pack_int4",
]

_QUANT_EXPORTS = [
    "QuantScheme",
    "quantize",
    "quantize_symmetric",
    "quantize_asymmetric",
    "dequantize",
    "quantize_weight_per_channel",
    "quantize_weight_per_channel_asymmetric",
    "calculate_quantization_error",
    "QuantizedLinear",
]

# Reason the Triton kernels could not be imported (None if they loaded fine).
_kernel_import_error: Exception | None = None

try:
    from triton_kernels.moe import (
        expert_ffn_torch,
        expert_ffn_triton,
        fused_expert_ffn,
        fused_moe_forward,
        grouped_gemm,
        moe_router,
        moe_router_torch,
        permute_tokens,
        unpermute_tokens,
    )
    from triton_kernels.quantized_matmul import (
        Int8Linear,
        int8_gemm,
        int8_gemm_torch,
    )
    from triton_kernels.rmsnorm import (
        TritonRMSNorm,
        rmsnorm,
        rmsnorm_residual_fused,
        rmsnorm_residual_torch,
        rmsnorm_torch,
    )
    from triton_kernels.swiglu import (
        SwiGLU,
        swiglu_fused,
        swiglu_torch,
        swiglu_with_bias,
    )
    from triton_kernels.w4a16 import (
        pack_int4,
        quantize_weight_int4_grouped,
        w4a16_gemm,
    )

    _KERNELS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - only hit on GPU-less hosts
    # Triton could not initialize its backend (typically: no GPU driver). Keep
    # the CPU-usable half of the package working and remember why.
    _KERNELS_AVAILABLE = False
    _kernel_import_error = exc


def __getattr__(name: str):
    """Give a clear error when a GPU-only symbol is used on a CPU-only host."""
    if name in _GPU_EXPORTS and not _KERNELS_AVAILABLE:
        raise AttributeError(
            f"'{name}' requires the Triton GPU kernels, which failed to import "
            f"on this host (no usable GPU driver). Original error: "
            f"{_kernel_import_error!r}"
        )
    raise AttributeError(f"module 'triton_kernels' has no attribute '{name}'")


# Export the quantization utilities always; export kernel symbols only when the
# Triton backend loaded, so `from triton_kernels import *` mirrors availability.
__all__ = _QUANT_EXPORTS + (_GPU_EXPORTS if _KERNELS_AVAILABLE else [])
