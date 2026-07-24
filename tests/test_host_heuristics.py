"""
CPU-side tests for pure host heuristics used by the kernels.

These cover launch-configuration logic that runs on the host (no GPU / Triton
compilation), so they execute anywhere - including CPU-only CI.
"""

import pytest

from triton_kernels.rmsnorm import _rmsnorm_launch_config
from triton_kernels.swiglu import _select_block_size


class TestSwigluBlockSize:
    """Tests for the adaptive SwiGLU block-size heuristic."""

    def test_returns_power_of_two(self):
        for n in [1, 100, 4096, 4097, 65536, 65537, 10_000_000]:
            bs = _select_block_size(n)
            assert bs & (bs - 1) == 0, f"{bs} is not a power of two"

    def test_small_tensors_use_smaller_blocks(self):
        """Small inputs get a smaller block so the grid has more programs."""
        small = _select_block_size(1024)
        large = _select_block_size(10_000_000)
        assert small < large
        assert small == 256
        assert large == 1024

    def test_monotonic_non_decreasing(self):
        """Block size never shrinks as the tensor grows."""
        sizes = [1, 4096, 4097, 65536, 65537, 1_000_000]
        blocks = [_select_block_size(n) for n in sizes]
        assert blocks == sorted(blocks)

    @pytest.mark.parametrize(
        "n,expected",
        [(4096, 256), (4097, 512), (65536, 512), (65537, 1024)],
    )
    def test_boundaries(self, n: int, expected: int):
        assert _select_block_size(n) == expected


class TestRMSNormLaunchConfig:
    """Tests for the RMSNorm (BLOCK_SIZE, num_warps) heuristic."""

    @pytest.mark.parametrize("hidden_dim", [64, 128, 512, 1024, 2048, 4096, 8192, 16384])
    def test_block_covers_hidden_dim_and_caps(self, hidden_dim: int):
        block_size, num_warps = _rmsnorm_launch_config(hidden_dim)
        # Block is a power of two, capped at 8192.
        assert block_size & (block_size - 1) == 0
        assert block_size <= 8192
        # It covers the row unless we hit the cap.
        assert block_size >= min(hidden_dim, 8192)
        # num_warps is a valid power-of-two warp count.
        assert num_warps in (1, 2, 4, 8, 16, 32)

    def test_num_warps_grows_with_width(self):
        """Wider rows should not use fewer warps than narrower rows."""
        dims = [128, 1024, 2048, 4096, 8192, 16384]
        warps = [_rmsnorm_launch_config(d)[1] for d in dims]
        assert warps == sorted(warps)
        # Narrow rows stay cheap, wide rows get the most warps.
        assert _rmsnorm_launch_config(128)[1] == 2
        assert _rmsnorm_launch_config(8192)[1] == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
