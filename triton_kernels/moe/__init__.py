"""
Fused MoE (Mixture-of-Experts) dispatch kernels implemented in Triton.

This submodule provides high-performance Triton kernels for the complete MoE
forward pass: router scoring, top-k gating, token permutation, expert GEMMs,
token unpermutation, and weighted combination.

The kernels use only Triton primitives for cross-platform portability
(NVIDIA + AMD via Triton backends).
"""

from triton_kernels.moe.expert_gemm import (
    expert_ffn_torch,
    expert_ffn_triton,
    grouped_gemm,
)
from triton_kernels.moe.fused_moe import (
    fused_expert_ffn,
    fused_moe_forward,
)
from triton_kernels.moe.permute import (
    compute_permutation_indices,
    permute_tokens,
    permute_tokens_torch,
    unpermute_tokens,
    unpermute_tokens_torch,
)
from triton_kernels.moe.router import moe_router, moe_router_torch

__all__ = [
    "moe_router",
    "moe_router_torch",
    "permute_tokens",
    "unpermute_tokens",
    "permute_tokens_torch",
    "unpermute_tokens_torch",
    "compute_permutation_indices",
    "grouped_gemm",
    "expert_ffn_triton",
    "expert_ffn_torch",
    "fused_moe_forward",
    "fused_expert_ffn",
]
