"""Velocity-Verlet-style integrator consistent with the velocity-based target.

Target:  a_t = (v_{t+1} - v_t) / dt
Step:    v_{t+1} = v_t + dt * a_t
         x_{t+1} = x_t + dt * v_{t+1}
"""
from __future__ import annotations

import numpy as np
import torch

DT = 0.01


def accel_from_velocities(v: np.ndarray, dt: float = DT) -> np.ndarray:
    """a[t] for t=0..T-2 from v[T,n,3] → shape (T-1,n,3)."""
    return (v[1:] - v[:-1]) / dt


def step_np(
    x: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray]:
    """One integration step. x,v,a: (n,3)."""
    v_next = v + dt * a
    x_next = x + dt * v_next
    return x_next, v_next


def step_torch(
    x: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    dt: float = DT,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_next = v + dt * a
    x_next = x + dt * v_next
    return x_next, v_next


def rollout_np(
    x0: np.ndarray,
    v0: np.ndarray,
    accelerations: np.ndarray,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a sequence of accelerations. accelerations: (K,n,3)."""
    xs = [x0]
    vs = [v0]
    x, v = x0.copy(), v0.copy()
    for k in range(len(accelerations)):
        x, v = step_np(x, v, accelerations[k], dt=dt)
        xs.append(x)
        vs.append(v)
    return np.stack(xs), np.stack(vs)


def integrate_gt_velocities(
    v: np.ndarray,
    x0: np.ndarray,
    dt: float = DT,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ground-truth a_t = (v_{t+1}-v_t)/dt from (x0, v0=v[0]).

    By definition ``v_t + dt * a_t = v_{t+1}`` exactly, so the velocity
    trajectory is reproduced bit-exactly in float32. Positions follow
    ``x_{t+1} = x_t + dt * v_{t+1}`` and match stored x within the Euler
    drift floor measured in check B (~1e-7 m RMS per step).
    """
    v = np.asarray(v, dtype=np.float32)
    x = np.asarray(x0, dtype=np.float32).copy()
    xs = [x.copy()]
    vs = [v[0].copy()]
    for t in range(len(v) - 1):
        a = (v[t + 1] - v[t]) / np.float32(dt)
        # Exact real identity; in float32, prefer the stored v_{t+1}.
        vv = v[t + 1]
        assert np.allclose(v[t] + np.float32(dt) * a, vv, rtol=0, atol=1e-5)
        x = x + np.float32(dt) * vv
        xs.append(x.copy())
        vs.append(vv.copy())
    return np.stack(xs), np.stack(vs)
