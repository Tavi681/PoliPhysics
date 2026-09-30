"""Smoke tests for the equal-time invariance pipeline (no multi-hour run).

Covers the failure that burned the first full run (float time matching) plus
dt selection, co-moving IC, parallel integrate, cache, meta write, from-cache.

Usage (from ``Tema 1``):
    python scripts/smoke_invariance_equal_t.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning.studies import (
    INVARIANCE_EQUAL_T_DT_CANDIDATES,
    _select_fixed_dt,
    _uniform_vertical_velocity,
    invariance_equal_t_params,
    max_shape_dev_equal_t,
)


def _ok(msg: str) -> None:
    print(f"PASS: {msg}", flush=True)


def test_float_time_matching() -> None:
    """Reproduce the exact failure mode from the 3.3 h run (0.10002 vs 0.1)."""
    t0 = SimpleNamespace()
    t1 = SimpleNamespace()
    # Same float-accumulation pattern as fixed dt=3e-5 after many steps.
    t0.t = np.array([0.0, 0.10002, 0.20004, 2.00004])
    t1.t = t0.t.copy()
    n = 5
    t0.x = np.zeros((len(t0.t), n, 3))
    t1.x = np.zeros_like(t0.x)
    t0.x[1, 2, 0] = 1e-14
    # Old tolerance 1e-9 would raise; new default must accept this.
    r = max_shape_dev_equal_t(
        t0, t1, L=0.5,
        t_compare=np.array([0.1, 0.2, 2.0]),
        output_dt=0.1,
    )
    assert r["max_shape_dev_equal_t"] < 1e-12
    # Tight tolerance still fails (guards that we didn't remove the check).
    try:
        max_shape_dev_equal_t(
            t0, t1, L=0.5,
            t_compare=np.array([0.1]),
            t_tol=1e-9,
            output_dt=0.1,
        )
    except RuntimeError:
        _ok("float time matching (0.10002 vs 0.1) + tight-tol still raises")
    else:
        raise AssertionError("expected RuntimeError for t_tol=1e-9")


def test_dt_selection() -> None:
    dt = _select_fixed_dt()
    assert dt in INVARIANCE_EQUAL_T_DT_CANDIDATES
    assert dt <= 5e-5  # 1e-4 and usually 5e-5 fail at eps=1e-10
    assert abs(dt - 3e-5) < 1e-15
    _ok(f"fixed dt selection -> {dt:.3e}")


def test_comoving_ic() -> None:
    P = invariance_equal_t_params(0.5, 3e-5, t_end=3e-5, output_dt=3e-5)
    xd = _uniform_vertical_velocity(P, 0.5)
    assert xd.shape == (P.n_dof,)
    assert np.allclose(xd[2:3 * P.n_nodes:3], 0.5)
    assert np.allclose(xd[0:3 * P.n_nodes:3], 0.0)
    assert np.allclose(xd[1:3 * P.n_nodes:3], 0.0)
    assert np.allclose(xd[3 * P.n_nodes:], 0.0)
    _ok("comoving IC sets all node vz=w, other DOFs 0")


def test_end_to_end_smoke() -> None:
    cache = ROOT / "results" / "equal_t_cache_smoke"
    meta = ROOT / "results" / "meta_smoke.json"
    for p in cache.glob("*.npz"):
        p.unlink()
    if meta.exists():
        meta.unlink()
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "run_invariance_equal_t.py"),
        "--smoke", "--dt", "3e-5",
        "--cache-dir", str(cache),
        "--meta-out", str(meta),
    ]
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(var, "1")
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"smoke pipeline exited {proc.returncode}")
    assert (cache / "w0.npz").exists()
    assert (cache / "w_comoving.npz").exists()
    assert (cache / "w_transient.npz").exists()
    meta_d = json.loads(meta.read_text())
    eq = meta_d["studies"]["fig_invariance"]["equal_t"]
    assert "comoving" in eq and "transient" in eq
    assert eq["dt"] == 3e-5
    assert eq["eps"] == 1e-10
    assert eq["K"] == 20
    assert eq["adaptive_dt"] is False
    assert len(eq["t_compare"]) >= 1
    # from-cache round trip via CLI
    cmd2 = [
        sys.executable, "-u", str(ROOT / "scripts" / "run_invariance_equal_t.py"),
        "--from-cache", str(cache), "--dt", "3e-5",
        "--t-end", str(eq["t_end"]), "--output-dt", str(eq["output_dt"]),
        "--meta-out", str(ROOT / "results" / "meta_smoke_from_cache.json"),
    ]
    proc2 = subprocess.run(cmd2, cwd=str(ROOT), env=env, capture_output=True, text=True)
    sys.stdout.write(proc2.stdout)
    sys.stderr.write(proc2.stderr)
    if proc2.returncode != 0:
        raise RuntimeError(f"from-cache exited {proc2.returncode}")
    _ok("end-to-end smoke + cache + meta + from-cache")


def main() -> int:
    test_float_time_matching()
    test_dt_selection()
    test_comoving_ic()
    test_end_to_end_smoke()
    print("ALL SMOKE TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
