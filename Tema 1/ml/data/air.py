"""KinematicSimulation helpers for air velocity at (x_t, t) / (x_t, t+dt)."""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ml.data.frames import DT
from ml.utils.git_info import repo_root


def make_ks(globals_: dict[str, Any]):
    tema1 = str(repo_root())
    if tema1 not in sys.path:
        sys.path.insert(0, tema1)
    from ballooning.fields import KinematicSimulation
    return KinematicSimulation(
        sigma=float(globals_["sigma_w"]),
        ell=float(globals_["ell"]),
        U_h=float(globals_["U_h"]),
        N_k=200,
        seed=int(globals_["seed"]),
        lambda_=0.5,
        renormalize=True,
    )


def u_at(flow, x_abs: np.ndarray, t: float) -> np.ndarray:
    """Air velocity float32 at absolute positions x_abs (n,3) and time t."""
    return np.asarray(flow.u(np.asarray(x_abs, dtype=np.float64), float(t)), dtype=np.float32)


def u_next_frame(rec: dict[str, Any], t: int, *, flow=None) -> np.ndarray:
    """u_next[t]: KS at current x_t and time t+dt (not /u_air[t+1])."""
    if "u_next" in rec:
        return np.asarray(rec["u_next"][t], dtype=np.float32)
    g = rec["globals"]
    dt = float(g.get("dt", DT))
    if flow is None:
        flow = make_ks(g)
    x_rel = np.asarray(rec["x_rel"][t], dtype=np.float64)
    spider = np.asarray(rec["spider_abs"][t], dtype=np.float64)
    x_abs = spider + x_rel
    t_val = float(np.asarray(rec["t"][t], dtype=np.float64))
    return u_at(flow, x_abs, t_val + dt)


def compute_u_next_run(rec: dict[str, Any]) -> np.ndarray:
    """Full (T,n,3) u_next array for a run."""
    g = rec["globals"]
    dt = float(g.get("dt", DT))
    flow = make_ks(g)
    t = np.asarray(rec["t"], dtype=np.float64)
    x_rel = np.asarray(rec["x_rel"], dtype=np.float64)
    spider = np.asarray(rec["spider_abs"], dtype=np.float64)
    T, n, _ = x_rel.shape
    out = np.empty((T, n, 3), dtype=np.float32)
    for i in range(T):
        out[i] = u_at(flow, spider[i] + x_rel[i], float(t[i] + dt))
    return out
