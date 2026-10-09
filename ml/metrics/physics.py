"""Physical diagnostics and paper metrics."""
from __future__ import annotations

import math

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return centre - half, centre + half


def edge_length_drift(x: np.ndarray, edges: np.ndarray, L_seg: float) -> float:
    """RMS relative drift of edge lengths vs rest length L_seg = L/N_t."""
    # x: (T,n,3) or (n,3)
    if x.ndim == 2:
        x = x[None]
    i, j = edges[:, 0], edges[:, 1]
    lengths = np.linalg.norm(x[:, j] - x[:, i], axis=-1)
    return float(np.sqrt(np.mean(((lengths - L_seg) / L_seg) ** 2)))


def min_interthread_distance(x: np.ndarray, thread_id: np.ndarray) -> float:
    """Minimum distance between nodes on different threads (exclude spider)."""
    if x.ndim == 3:
        x = x[-1]
    tid = thread_id
    mask = tid >= 0
    pts = x[mask]
    tids = tid[mask]
    if len(pts) < 2:
        return float("inf")
    dmin = float("inf")
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            if tids[a] == tids[b]:
                continue
            d = float(np.linalg.norm(pts[a] - pts[b]))
            if d < dmin:
                dmin = d
    return dmin


def opening_R_over_L(x: np.ndarray, tip_ids: np.ndarray, spider_idx: int = 0, L: float = 0.5) -> float:
    """Mean tip radial distance from spider / L."""
    if x.ndim == 3:
        # time average
        rs = []
        for t in range(len(x)):
            sp = x[t, spider_idx]
            tips = x[t, tip_ids]
            R = np.mean(np.linalg.norm(tips[:, :2] - sp[:2], axis=-1))
            rs.append(R / L)
        return float(np.mean(rs))
    sp = x[spider_idx]
    tips = x[tip_ids]
    R = np.mean(np.linalg.norm(tips[:, :2] - sp[:2], axis=-1))
    return float(R / L)


def mean_vertical_spider_velocity(x: np.ndarray, t: np.ndarray, spider_idx: int = 0) -> float:
    z = x[:, spider_idx, 2]
    if len(t) < 2:
        return float("nan")
    return float((z[-1] - z[0]) / (t[-1] - t[0]))
