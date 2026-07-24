"""
Example: Mixture-of-Experts forward pass via the PyTorch reference, on CPU.

Builds a small Mixtral-style MoE layer (softmax gating, top-2 of 8 experts) with
the pure-PyTorch reference (`reference/moe_reference.py`), runs a forward pass,
and reports the per-expert token load. Runs on CPU - no GPU or Triton needed -
and shows the numerics the Triton `fused_moe_forward` kernel is validated against.

Run (from the repo root):
    python examples/moe_reference_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from reference.moe_reference import MoEReference  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    hidden_dim, ffn_dim = 256, 512
    num_experts, top_k = 8, 2
    num_tokens = 64

    moe = MoEReference(hidden_dim, ffn_dim, num_experts, top_k, gating="softmax")
    x = torch.randn(num_tokens, hidden_dim)

    output, routing = moe(x)

    print(f"MoE: {num_experts} experts, top-{top_k}, hidden={hidden_dim}, ffn={ffn_dim}")
    print(f"input  {tuple(x.shape)} -> output {tuple(output.shape)}\n")

    # Per-expert token load (each token activates top_k experts).
    load = torch.bincount(routing.top_k_indices.reshape(-1), minlength=num_experts)
    total = load.sum().item()
    print("expert load (tokens routed):")
    for e, count in enumerate(load.tolist()):
        bar = "#" * count
        print(f"  expert {e}: {count:3d}  {bar}")
    print(f"\ntotal routed slots = {total} = num_tokens * top_k = {num_tokens * top_k}")
    print(f"An ideal balanced router would place ~{total / num_experts:.0f} tokens per expert.")


if __name__ == "__main__":
    main()
