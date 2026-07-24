"""
CPU reference-math tests for the MoE reference module.

Exercises `reference/moe_reference.py`, the pure-PyTorch ground truth for the
Triton MoE dispatch kernels. All tests run on CPU (no GPU / Triton), covering
router gating and top-k selection numerics.
"""

import pytest
import torch

from reference.moe_reference import moe_router_torch


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
