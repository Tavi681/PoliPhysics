"""Oracle GT-acceleration through full_rollout_surrogate matches stored positions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.data.frames import DT
from ml.evaluate import full_rollout_surrogate
from ml.models.oracle import OracleGroundTruthAccel

DATA = Path("results/ml/processed/5c60fa669a719404")
N1 = "N1_sw0.15_x+4_i0099"
N4 = "N4_sw0.15_x+1_i0029"
C = 5
N_STEPS = 100
# Float32 Euler + spider float64 path: ~1e-6 m over 100 steps is the expected floor.
ATOL_M = 5e-6


@pytest.fixture(scope="module")
def store():
    if not (DATA / "norm_stats.json").exists():
        pytest.skip(f"missing data dir {DATA}")
    from ml.data.dataset import BallooningRunStore

    return BallooningRunStore(DATA, cache_shards=2, mmap=True)


def _oracle_rollout(store, rid: str, n_steps: int = N_STEPS):
    rec = store.get(rid)
    assert rec["v"].shape[0] >= C + n_steps + 1
    model = OracleGroundTruthAccel(
        rec, C=C, a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"], dt=DT,
    )
    model.reset()
    # full_rollout_surrogate compares absolute sim time to t_end (not time-since-release).
    t_end = float(rec["t"][C]) + n_steps * DT + 0.5 * DT
    pred = full_rollout_surrogate(
        model, rec, store.norm, torch.device("cpu"),
        C=C, t_end=t_end, return_trajectory=True,
        node_features=["v_hist", "u", "charge"],
        globals_fields=["qE", "m", "N"],
        edge_sets=["material"],
        radius=0.05,
        normalize_features=True,
    )
    T = min(pred["x"].shape[0], n_steps + 1)
    x_gt = (
        rec["spider_abs"][C : C + T, None, :]
        + rec["x_rel"][C : C + T].astype(np.float64)
    )
    return pred["x"][:T], x_gt[:T], rec


@pytest.mark.parametrize("rid", [N1, N4])
def test_oracle_rollout_matches_hdf5_to_roundoff(store, rid):
    if rid not in store:
        pytest.skip(f"run {rid} not in local store")
    X, G, rec = _oracle_rollout(store, rid, N_STEPS)
    assert X.shape[0] == N_STEPS + 1, f"got T={X.shape[0]} status may have exited early"
    err = np.abs(X - G)
    # Spider (float64 path) should be near machine eps; thread nodes float32 Euler floor.
    assert float(np.max(err[:, 0])) < 1e-6
    assert float(np.max(err)) < ATOL_M, (
        f"{rid}: max |dx|={float(np.max(err)):.3e} m "
        f"(spider={float(np.max(err[:, 0])):.3e})"
    )
    # Node indexing: 1-based thread j starts at node (j-1)*N_t+1; stored thread_id is 0-based.
    N = int(rec["globals"]["N"])
    N_t = int(rec["globals"]["N_t"])
    assert rec["x_rel"].shape[1] == 1 + N * N_t
    assert int(rec["thread_id"][0]) == -1
    for j in range(1, N + 1):
        assert int(rec["thread_id"][(j - 1) * N_t + 1]) == j - 1
