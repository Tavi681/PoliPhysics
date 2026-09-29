"""Derived observables (observables.py).

- V       : spider vertical velocity zdot_0
- R       : mean horizontal distance of tips from the bundle axis
- theta_L : tip angle from the vertical
- d_min   : min distance between nodes of different threads (excluding the two
            nodes nearest the spider)
- T       : per-edge tension  T = Y A (|e|/l0 - 1)
- invariant T^beta sin(theta) : first integral along each thread
"""

from __future__ import annotations

import numpy as np

from .geometry import Topology
from .params import Params

Z_HAT = np.array([0.0, 0.0, 1.0])


def spider_vertical_velocity(V: np.ndarray) -> float:
    """V = zdot_0 (vertical velocity of the spider node)."""
    return float(V[0, 2])


def tip_radius(topo: Topology, X: np.ndarray) -> float:
    """R = mean horizontal distance of the thread tips from the spider axis."""
    axis_xy = X[0, :2]
    d = [np.linalg.norm(X[topo.tip_nodes[j], :2] - axis_xy) for j in range(topo.N)]
    return float(np.mean(d))


def tip_angle(topo: Topology, X: np.ndarray) -> float:
    """theta_L = mean angle of each thread's last edge from the vertical [rad]."""
    et = topo.edge_tangents(X)
    angs = []
    for j in range(topo.N):
        last_edge = topo.thread_edges[j][-1]
        c = np.clip(abs(np.dot(et[last_edge], Z_HAT)), -1.0, 1.0)
        angs.append(np.arccos(c))
    return float(np.mean(angs))


def min_interthread_distance(topo: Topology, X: np.ndarray) -> float:
    """d_min between nodes of different threads, excluding the two nodes nearest
    the spider on each thread."""
    if topo.N < 2:
        return float("inf")
    dmin = np.inf
    for ja in range(topo.N):
        for jb in range(ja + 1, topo.N):
            for na in topo.thread_nodes[ja][2:]:
                for nb in topo.thread_nodes[jb][2:]:
                    dmin = min(dmin, float(np.linalg.norm(X[na] - X[nb])))
    return dmin


def edge_tensions(P: Params, topo: Topology, X: np.ndarray) -> np.ndarray:
    """Per-edge tension T = Y A (|e|/l0 - 1)."""
    lengths = topo.edge_lengths(X)
    return P.Y * P.A * (lengths / P.l0 - 1.0)


def edge_angles_from_vertical(topo: Topology, X: np.ndarray) -> np.ndarray:
    """Per-edge angle from the vertical [rad]."""
    et = topo.edge_tangents(X)
    c = np.clip(np.abs(et @ Z_HAT), -1.0, 1.0)
    return np.arccos(c)


def first_integral(P: Params, topo: Topology, X: np.ndarray) -> np.ndarray:
    """Per-edge invariant  T^beta * sin(theta)  (theta = edge angle from vertical)."""
    T = edge_tensions(P, topo, X)
    ang = edge_angles_from_vertical(topo, X)
    beta = P.beta_first_integral
    return np.sign(T) * np.abs(T) ** beta * np.sin(ang)


def summary(P: Params, topo: Topology, X: np.ndarray, V: np.ndarray) -> dict:
    """Convenience bundle of scalar observables."""
    return {
        "V": spider_vertical_velocity(V),
        "R": tip_radius(topo, X),
        "theta_L": tip_angle(topo, X),
        "d_min": min_interthread_distance(topo, X),
    }
