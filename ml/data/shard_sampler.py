"""Shard-aware epoch sampler: shuffle shard order, then frames within each shard.

Avoids thrashing the LRU shard cache under uniform random access across ~164
× ~420 MB consolidated shards.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class ShardAwareSampler(Sampler[int]):
    """Yield dataset indices with shard locality.

    Parameters
    ----------
    index:
        Parallel to ``BallooningDataset.index``: list of ``(run_id, t)``.
    rid_to_shard:
        ``run_id → shard filename``.
    weights:
        Optional per-index weights (e.g. per-run). When set, within each shard
        we draw ``len(shard_indices)`` samples with replacement proportional to
        weight (keeps expected per-run mass while preserving locality).
    """

    def __init__(
        self,
        index: Sequence[tuple[str, int]],
        rid_to_shard: dict[str, str],
        *,
        weights: torch.Tensor | None = None,
        seed: int = 0,
    ):
        self.seed = int(seed)
        self._epoch = 0
        by_shard: dict[str, list[int]] = defaultdict(list)
        for i, (rid, _) in enumerate(index):
            by_shard[rid_to_shard[rid]].append(i)
        self._shard_order = sorted(by_shard.keys())
        self._by_shard = {s: np.asarray(ix, dtype=np.int64) for s, ix in by_shard.items()}
        self._weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.num_samples = len(index)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)
        shards = list(self._shard_order)
        rng.shuffle(shards)
        out: list[int] = []
        for sh in shards:
            ix = self._by_shard[sh]
            if self._weights is None:
                order = ix.copy()
                rng.shuffle(order)
                out.extend(int(i) for i in order)
            else:
                w = self._weights[ix]
                w = w / max(w.sum(), 1e-12)
                chosen = rng.choice(ix, size=len(ix), replace=True, p=w)
                out.extend(int(i) for i in chosen)
        # epoch length matches dataset length
        return iter(out)
