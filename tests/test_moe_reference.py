"""
CPU reference-math tests for the MoE reference module.

Exercises `reference/moe_reference.py`, the pure-PyTorch ground truth for the
Triton MoE dispatch kernels. All tests run on CPU (no GPU / Triton), covering
router gating and top-k selection numerics.
"""

import pytest
import torch

from reference.moe_reference import (
    moe_router_torch,
    permute_tokens,
    unpermute_tokens,
)


class TestMoERouter:
    """Router gating + top-k selection numerics (softmax and sigmoid)."""

    def test_softmax_scores_are_probabilities(self):
        """Selected softmax weights are a subset of a full simplex per token."""
        torch.manual_seed(0)
        num_tokens, hidden_dim, num_experts, top_k = 32, 64, 8, 2
        hidden = torch.randn(num_tokens, hidden_dim)
        router_w = torch.randn(num_experts, hidden_dim)

        result = moe_router_torch(hidden, router_w, top_k, gating="softmax")

        assert result.top_k_indices.shape == (num_tokens, top_k)
        assert result.top_k_weights.shape == (num_tokens, top_k)
        # Softmax weights are individually valid probabilities in [0, 1].
        assert torch.all(result.top_k_weights >= 0)
        assert torch.all(result.top_k_weights <= 1)

    def test_topk_selects_largest_logits(self):
        """Top-k indices are exactly the experts with the largest logits."""
        torch.manual_seed(1)
        num_tokens, hidden_dim, num_experts, top_k = 16, 32, 8, 3
        hidden = torch.randn(num_tokens, hidden_dim)
        router_w = torch.randn(num_experts, hidden_dim)

        result = moe_router_torch(hidden, router_w, top_k, gating="softmax")

        # softmax is monotonic, so top-k of scores == top-k of logits.
        expected = result.router_logits.topk(top_k, dim=-1).indices
        assert torch.equal(result.top_k_indices, expected)

    def test_sigmoid_weights_sum_to_one(self):
        """DeepSeek-style sigmoid gating renormalizes the selected weights."""
        torch.manual_seed(2)
        num_tokens, hidden_dim, num_experts, top_k = 24, 48, 16, 4
        hidden = torch.randn(num_tokens, hidden_dim)
        router_w = torch.randn(num_experts, hidden_dim)

        result = moe_router_torch(hidden, router_w, top_k, gating="sigmoid")
        sums = result.top_k_weights.float().sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3)

    def test_router_logits_computed_in_fp32(self):
        """Logits are the float32 projection regardless of input dtype."""
        hidden = torch.randn(8, 16, dtype=torch.float16)
        router_w = torch.randn(4, 16, dtype=torch.float16)
        result = moe_router_torch(hidden, router_w, top_k=2)
        expected = torch.nn.functional.linear(hidden.float(), router_w.float())
        torch.testing.assert_close(result.router_logits, expected)

    def test_top_k_equals_num_experts_covers_all(self):
        hidden = torch.randn(5, 16)
        router_w = torch.randn(4, 16)
        result = moe_router_torch(hidden, router_w, top_k=4, gating="softmax")
        # Every token selects all four experts (as a permutation).
        for row in result.top_k_indices:
            assert set(row.tolist()) == {0, 1, 2, 3}

    def test_unknown_gating_raises(self):
        with pytest.raises(ValueError, match="Unknown gating"):
            moe_router_torch(torch.randn(2, 8), torch.randn(4, 8), top_k=2, gating="relu")


class TestPermuteUnpermute:
    """Permutation to expert-contiguous layout and back is order-preserving."""

    def test_expert_offsets_partition_all_slots(self):
        """Offsets are a valid non-decreasing partition covering every slot."""
        torch.manual_seed(3)
        num_tokens, hidden_dim, num_experts, top_k = 20, 16, 8, 2
        hidden = torch.randn(num_tokens, hidden_dim)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k))

        permuted, expert_offsets, restore_idx = permute_tokens(hidden, top_k_indices, num_experts)

        assert permuted.shape == (num_tokens * top_k, hidden_dim)
        assert expert_offsets.shape == (num_experts + 1,)
        assert expert_offsets[0].item() == 0
        assert expert_offsets[-1].item() == num_tokens * top_k
        # Offsets are monotone non-decreasing.
        assert torch.all(expert_offsets[1:] >= expert_offsets[:-1])
        # Bucket sizes match the actual per-expert assignment counts.
        counts = torch.bincount(top_k_indices.reshape(-1), minlength=num_experts)
        assert torch.equal(expert_offsets[1:] - expert_offsets[:-1], counts)

    def test_permuted_rows_grouped_by_expert(self):
        """Every slot in expert e's range came from a token routed to expert e."""
        torch.manual_seed(4)
        num_tokens, hidden_dim, num_experts, top_k = 12, 8, 4, 2
        hidden = torch.arange(num_tokens * hidden_dim, dtype=torch.float32).reshape(
            num_tokens, hidden_dim
        )
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k))

        permuted, expert_offsets, _ = permute_tokens(hidden, top_k_indices, num_experts)
        expanded = hidden.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)

        for e in range(num_experts):
            start, end = expert_offsets[e].item(), expert_offsets[e + 1].item()
            for slot in range(start, end):
                # Find which original (token, k) this row equals and check its expert.
                match = (expanded == permuted[slot]).all(dim=1).nonzero()
                token_ids = match.flatten() // top_k
                assert e in top_k_indices[token_ids].reshape(-1).tolist()

    def test_unpermute_restores_weighted_combination(self):
        """Unpermute reproduces the direct top-k weighted sum of expert outputs."""
        torch.manual_seed(5)
        num_tokens, hidden_dim, num_experts, top_k = 10, 8, 4, 2
        hidden = torch.randn(num_tokens, hidden_dim)
        top_k_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        top_k_weights = torch.rand(num_tokens, top_k)

        permuted, _, restore_idx = permute_tokens(hidden, top_k_indices, num_experts)
        # Identity "expert": pass the permuted tokens straight through.
        combined = unpermute_tokens(permuted, restore_idx, top_k_weights)

        # Ground truth: each token is weight-summed over its own duplicated copies.
        expected = (hidden.unsqueeze(1) * top_k_weights.unsqueeze(-1)).sum(dim=1)
        torch.testing.assert_close(combined, expected, rtol=1e-4, atol=1e-4)
