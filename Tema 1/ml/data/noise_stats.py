"""Training noise scale from acceleration normalizer (GNS-style).

Legacy ``noise_stats.json`` (std of dv with t<5 rest→air frames) is removed:
it was ~6× too large vs ``a_std * dt`` on the same train frames. Noise is
always derived from ``norm_stats`` (global or per-type ``a_std``).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ml.data.frames import DT

# node_type: 0=spider, 1=interior, 2=tip
TYPE_NAMES = ("spider", "interior", "tip")


def dv_std_from_a_std(a_std: Sequence[float] | np.ndarray, dt: float = DT) -> float:
    """Scalar RMS std of dv = a*dt from per-component acceleration std."""
    a = np.asarray(a_std, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((a * dt) ** 2)))


def a_std_by_type_matrix(norm: dict[str, Any]) -> np.ndarray | None:
    """Return (3, 3) a_std[type, component] from norm_stats, or None if missing."""
    by = norm.get("a_by_type")
    if not isinstance(by, dict):
        return None
    rows = []
    for name in TYPE_NAMES:
        entry = by.get(name)
        if not isinstance(entry, dict) or "std" not in entry:
            return None
        rows.append(np.asarray(entry["std"], dtype=np.float64).reshape(-1)[:3])
    return np.stack(rows, axis=0).astype(np.float32)


def resolve_noise(
    *,
    noise_k: float | None,
    noise_std: float | None,
    noise_mode: str = "per_type",
    a_std: Sequence[float] | None,
    a_std_by_type: np.ndarray | Sequence[Sequence[float]] | None = None,
    dt: float = DT,
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """Return (scalar_vel_rms, per_type_comp[3,3] or None, global_comp[3] or None).

    ``noise_mode``:
      - ``per_type`` (default): ``sigma[type,c] = k * a_std_type[c] * dt``
      - ``global``: ``sigma[c] = k * a_std[c] * dt`` for all nodes

    Target normalization stays global ``(a - a_mean) / a_std``. With the
    corrected target ``a = (v_{t+1}^{gt} - v_t^{noisy}) / dt``, ``k=1`` adds
    variance ``(a_std_type[c] / a_std[c])^2`` (per_type) or ``1`` (global) to
    each normalized component within that node type.
    """
    mode = str(noise_mode or "per_type").strip().lower()
    if mode not in ("per_type", "global"):
        raise ValueError(f"noise_mode must be 'per_type' or 'global', got {noise_mode!r}")

    if noise_k is not None:
        k = float(noise_k)
        if mode == "per_type" and a_std_by_type is not None:
            mat = (np.asarray(a_std_by_type, dtype=np.float64).reshape(3, 3) * float(dt) * k).astype(
                np.float32
            )
            return float(np.sqrt(np.mean(mat.astype(np.float64) ** 2))), mat, None
        if a_std is not None:
            comp = (np.asarray(a_std, dtype=np.float64).reshape(-1)[:3] * float(dt) * k).astype(
                np.float32
            )
            if mode == "per_type":
                # Fallback: no a_by_type → broadcast global to all types.
                mat = np.stack([comp, comp, comp], axis=0)
                return float(np.sqrt(np.mean(comp.astype(np.float64) ** 2))), mat, None
            return float(np.sqrt(np.mean(comp.astype(np.float64) ** 2))), None, comp
        return 0.0, None, None
    if noise_std is not None:
        return float(noise_std), None, None
    return 0.0, None, None
