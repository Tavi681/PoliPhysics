"""Frame regularity + training-window masks."""
from __future__ import annotations

import numpy as np

DT = 0.01
DT_TOL = 1e-6


def frame_indices(t: np.ndarray, dt: float = DT) -> np.ndarray:
    """Integer frame indices on the nominal grid (float32-aware)."""
    return np.round(np.asarray(t, dtype=np.float64) / dt).astype(np.int64)


def irregular_intervals(
    t: np.ndarray,
    dt: float = DT,
    tol: float = DT_TOL,
    mode: str = "index",
) -> np.ndarray:
    """Return interval indices i where (t[i], t[i+1]) is irregular.

    mode='raw':   |dt_i - dt| > tol on float64(cast of stored t)
    mode='index': round(t/dt) fails to advance by exactly 1 (true skips/dups;
                  float32 ULP jitter on a regular grid yields no flags)
    """
    t64 = np.asarray(t, dtype=np.float64)
    if len(t64) < 2:
        return np.array([], dtype=np.int64)
    if mode == "raw":
        dti = np.diff(t64)
        return np.where(np.abs(dti - dt) > tol)[0].astype(np.int64)
    if mode == "index":
        k = frame_indices(t64, dt=dt)
        return np.where(np.diff(k) != 1)[0].astype(np.int64)
    raise ValueError(f"unknown mode {mode}")


def window_ok_mask(
    t: np.ndarray,
    C: int,
    dt: float = DT,
    mode: str = "index",
) -> np.ndarray:
    """Boolean mask over frame index t for usable training samples.

    A sample at frame t (predicting a_t) needs history of C velocities and
    v_{t+1}, so frames [t-C, ..., t+1] and intervals [t-C, ..., t].
    Valid t ∈ [C, T-2].
    """
    T = len(t)
    ok = np.zeros(T, dtype=bool)
    if T < C + 2:
        return ok
    bad = set(irregular_intervals(t, dt=dt, mode=mode).tolist())
    for ti in range(C, T - 1):
        if any(i in bad for i in range(ti - C, ti + 1)):
            continue
        ok[ti] = True
    return ok
