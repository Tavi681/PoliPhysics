"""Shard-aware sampler preserves length and shard locality."""
from __future__ import annotations

from ml.data.shard_sampler import ShardAwareSampler


def test_shard_aware_epoch_locality():
    index = [("r0", 5), ("r0", 6), ("r1", 5), ("r2", 5), ("r2", 6), ("r2", 7)]
    rid_to_shard = {"r0": "s0", "r1": "s0", "r2": "s1"}
    samp = ShardAwareSampler(index, rid_to_shard, seed=0)
    assert len(samp) == 6
    order = list(samp)
    assert sorted(order) == list(range(6))
    # all indices of s0 appear as a contiguous block (order within block may vary)
    pos = {i: k for k, i in enumerate(order)}
    s0 = [0, 1, 2]
    s1 = [3, 4, 5]
    s0_pos = sorted(pos[i] for i in s0)
    s1_pos = sorted(pos[i] for i in s1)
    assert s0_pos[-1] - s0_pos[0] == len(s0) - 1
    assert s1_pos[-1] - s1_pos[0] == len(s1) - 1
