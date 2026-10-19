"""Batched torch KS must match simulator KinematicSimulation to float32 round-off."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.data.air import make_ks, u_at
from ml.data.dataset import BallooningRunStore
from ml.data.ks_torch import (
    TurbModeCache,
    ks_u_torch,
    turb_modes_from_globals,
    turb_modes_from_h5,
    turb_modes_from_ks,
)
from ml.utils.git_info import repo_root


def _f32_eps_scale(ref: np.ndarray) -> float:
    """Typical float32 round-off scale for comparing two f32 evaluations."""
    return float(np.finfo(np.float32).eps * max(1.0, np.max(np.abs(ref))))


def test_torch_ks_vs_simulator_three_runs_per_N():
    """(a) 3 runs per N, 50 random (x,t): max |Δ| ≤ float32 round-off."""
    raw = repo_root() / "results" / "ml" / "raw_smoke"
    if not raw.is_dir():
        pytest.skip("raw_smoke HDF5 not present")
    h5s = sorted(raw.glob("N*_sw*.h5"))
    by_n: dict[int, list[Path]] = {}
    for p in h5s:
        # N1_sw0.15_x+0_i0000.h5
        n = int(p.name.split("_")[0][1:])
        by_n.setdefault(n, []).append(p)
    assert by_n, "no HDF5 files"
    rng = np.random.default_rng(0)
    max_abs = 0.0
    max_rel = 0.0
    n_checked = 0
    for n, paths in sorted(by_n.items()):
        for p in paths[:3]:
            modes_h5 = turb_modes_from_h5(p)
            # reconstruct via make_ks from attrs
            import h5py
            with h5py.File(p, "r") as f:
                attrs = dict(f["turb"].attrs)
                params = dict(f["params"].attrs)
                g = {
                    "sigma_w": float(attrs.get("sigma_w", attrs.get("sigma"))),
                    "ell": float(attrs["ell"]),
                    "U_h": float(attrs.get("U_h", params.get("U_h", 1.0))),
                    "seed": int(attrs["seed"]),
                }
            flow = make_ks(g)
            modes_ks = turb_modes_from_ks(flow)
            assert np.allclose(modes_h5.k, modes_ks.k)
            assert np.allclose(modes_h5.a, modes_ks.a)
            assert np.allclose(modes_h5.b, modes_ks.b)
            assert np.allclose(modes_h5.omega, modes_ks.omega)

            for _ in range(50):
                x = rng.normal(0.0, 2.0, size=(8, 3))
                t = float(rng.uniform(0.0, 30.0))
                ref = np.asarray(flow.u(x, t), dtype=np.float64)
                got = ks_u_torch(
                    torch.from_numpy(x.astype(np.float64)),
                    t,
                    modes_h5,
                    dtype=torch.float32,
                ).numpy().astype(np.float64)
                diff = np.abs(got - ref)
                max_abs = max(max_abs, float(np.max(diff)))
                denom = np.maximum(np.abs(ref), 1e-30)
                max_rel = max(max_rel, float(np.max(diff / denom)))
                n_checked += 1
    assert n_checked >= 50
    print(
        f"test_torch_ks max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e} n={n_checked}",
        flush=True,
    )
    # Amplitude sum is float32 after f64 phase; typical max_abs ~1.2e-5 vs f64 ref.
    assert max_abs <= 1.5e-5, f"max_abs={max_abs}"


def test_torch_u_next_matches_stored_local_subset():
    """(b) u_next via torch KS == stored u_next to float32 round-off."""
    root = repo_root() / "results" / "ml" / "processed" / "local_subset"
    if not (root / "manifest.json").exists():
        pytest.skip("local_subset missing")
    store = BallooningRunStore(root, cache_shards=1, mmap=True)
    cache = TurbModeCache()
    max_abs = 0.0
    n = 0
    for rid in list(store.runs.keys())[:8]:
        rec = store.get(rid)
        if "u_next" not in rec:
            pytest.skip("local_subset has no stored u_next")
        modes = cache.get(rec["globals"])
        T = min(int(rec["v"].shape[0]), 40)
        dt = float(rec["globals"].get("dt", 0.01))
        for t in range(0, T, 3):
            x_abs = np.asarray(rec["spider_abs"][t], dtype=np.float64) + np.asarray(
                rec["x_rel"][t], dtype=np.float64
            )
            t_val = float(np.asarray(rec["t"][t])) + dt
            got = ks_u_torch(
                torch.from_numpy(x_abs),
                t_val,
                modes,
                dtype=torch.float64,
            ).numpy().astype(np.float32)
            ref = np.asarray(rec["u_next"][t], dtype=np.float32)
            max_abs = max(max_abs, float(np.max(np.abs(got - ref))))
            n += 1
    print(f"test_stored_u_next max_abs_diff={max_abs:.6e} n={n}")
    assert n > 0
    assert max_abs <= 1e-5, f"max_abs={max_abs}"


def test_batched_matches_single():
    raw = repo_root() / "results" / "ml" / "raw_smoke"
    paths = sorted(raw.glob("N*_sw*.h5"))[:3] if raw.is_dir() else []
    if len(paths) < 2:
        pytest.skip("need >=2 raw_smoke files")
    modes = [turb_modes_from_h5(p) for p in paths]
    rng = np.random.default_rng(1)
    M = 16
    xs = np.stack([rng.normal(0, 1, (M, 3)) for _ in modes])
    ts = rng.uniform(0, 10, size=len(modes))
    batched = ks_u_torch(
        torch.from_numpy(xs),
        torch.from_numpy(ts),
        modes,
        dtype=torch.float64,
    ).numpy()
    for i, m in enumerate(modes):
        single = ks_u_torch(
            torch.from_numpy(xs[i]),
            float(ts[i]),
            m,
            dtype=torch.float64,
        ).numpy()
        assert np.max(np.abs(batched[i] - single)) < 1e-12
