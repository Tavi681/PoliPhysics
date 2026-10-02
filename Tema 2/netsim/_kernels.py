"""Low-level segment kernels (strains, tensions, internal forces).

Shared by :mod:`netsim.integrator` and :mod:`netsim.energy`. Provided as a
vectorized NumPy implementation plus an optional Numba-jitted force loop with an
identical signature. This private module is an implementation detail (not part of
the file list in the objective); it exists to avoid duplicating the force/strain
math and to keep the hot loop in one place.
"""

from __future__ import annotations

import numpy as np

try:  # optional acceleration
    import numba

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is optional
    numba = None
    _HAVE_NUMBA = False

__all__ = [
    "segment_strains",
    "segment_forces",
    "have_numba",
]


def have_numba() -> bool:
    return _HAVE_NUMBA


def segment_strains(x: np.ndarray, edges: np.ndarray, rest: np.ndarray):
    """Return (eps, length) per segment (unmasked; caller applies intact mask)."""
    d = x[edges[:, 1]] - x[edges[:, 0]]
    length = np.sqrt(np.sum(d * d, axis=1))
    eps = length / rest - 1.0
    return eps, length


def segment_forces_numpy(x, edges, rest, A, intact, E0, b):
    """Vectorized internal (tension) forces on nodes.

    Tension T = A*(E0 eps + b eps^3) for intact, non-slack segments (eps > 0),
    zero otherwise. Force on node i from segment ij is T*(x_j - x_i)/|x_j - x_i|.
    """
    i = edges[:, 0]
    j = edges[:, 1]
    d = x[j] - x[i]
    length = np.sqrt(np.sum(d * d, axis=1))
    eps = length / rest - 1.0
    active = intact & (eps > 0.0)
    T = np.where(active, A * (E0 * eps + b * eps ** 3), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fdir = np.where(length[:, None] > 0.0, d / length[:, None], 0.0)
    fseg = T[:, None] * fdir
    f = np.zeros_like(x)
    np.add.at(f, i, fseg)
    np.add.at(f, j, -fseg)
    return f


if _HAVE_NUMBA:  # pragma: no cover - exercised only when numba present

    @numba.njit(cache=True, fastmath=True)
    def _segment_forces_numba(x, edges, rest, A, intact, E0, b, f):
        n_seg = edges.shape[0]
        for s in range(n_seg):
            if not intact[s]:
                continue
            i = edges[s, 0]
            j = edges[s, 1]
            dx = x[j, 0] - x[i, 0]
            dy = x[j, 1] - x[i, 1]
            dz = x[j, 2] - x[i, 2]
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            if length <= 0.0:
                continue
            eps = length / rest[s] - 1.0
            if eps <= 0.0:
                continue
            T = A[s] * (E0 * eps + b * eps * eps * eps)
            inv = T / length
            fx = inv * dx
            fy = inv * dy
            fz = inv * dz
            f[i, 0] += fx
            f[i, 1] += fy
            f[i, 2] += fz
            f[j, 0] -= fx
            f[j, 1] -= fy
            f[j, 2] -= fz
        return f

    def segment_forces_numba(x, edges, rest, A, intact, E0, b):
        f = np.zeros_like(x)
        return _segment_forces_numba(x, edges, rest, A,
                                     intact.astype(np.bool_), E0, b, f)


def segment_forces(x, edges, rest, A, intact, E0, b, *, use_numba=False):
    """Internal forces; dispatches to the numba kernel when requested/available."""
    if use_numba and _HAVE_NUMBA:
        return segment_forces_numba(x, edges, rest, A, intact, E0, b)
    return segment_forces_numpy(x, edges, rest, A, intact, E0, b)
