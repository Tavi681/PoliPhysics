"""Contact models between the rigid drone (sphere) and the net.

Two mandatory modes:

* ``frictionless`` - a Hertz penalty force ``f_c = k_c delta^{3/2} n_hat``
  between the sphere and each net node, with ``delta = r_d - |x_i - x_d| > 0``
  and ``n_hat`` the sphere's outward radial normal. The equal and opposite force
  acts on the drone. No friction.
* ``gripped`` - at first contact the net node closest to the impact point is
  rigidly attached to the drone (it follows the drone and its mass is added to
  M). All other nodes interact through the penalty force as in ``frictionless``.

An optional sphere-segment penalty (point-segment distance) is provided behind a
flag; it is not the default.

The contact potential energy of the Hertz penalty is
``U_c = (2/5) k_c delta^{5/2}`` per contacting pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GrippedState",
    "hertz_penalty_nodes",
    "sphere_segment_penalty",
    "nearest_node",
]

_EPS = 1e-30


@dataclass
class GrippedState:
    """State of the gripped attachment (``gripped`` contact mode)."""

    active: bool = False
    node: int = -1
    offset: np.ndarray = None  # x_grip - x_drone at attachment (constant)
    mass: float = 0.0          # mass of the gripped node, added to drone


def nearest_node(x: np.ndarray, p3: np.ndarray, mask: np.ndarray) -> int:
    """Index of the node (within ``mask``) closest to point ``p3``."""
    d2 = np.sum((x - p3) ** 2, axis=1)
    d2 = np.where(mask, d2, np.inf)
    return int(np.argmin(d2))


def hertz_penalty_nodes(x: np.ndarray, xd: np.ndarray, r_d: float, k_c: float,
                        active: np.ndarray):
    """Hertz penalty between the sphere and net nodes.

    Parameters
    ----------
    x:
        (n, 3) node positions.
    xd:
        (3,) drone centre.
    r_d, k_c:
        sphere radius and penalty stiffness.
    active:
        (n,) boolean mask of nodes eligible for penalty contact (excludes the
        gripped node).

    Returns
    -------
    f_nodes:
        (n, 3) penalty force on each node.
    f_drone:
        (3,) total reaction on the drone.
    energy:
        scalar contact potential energy (2/5) k_c sum delta^{5/2}.
    """
    d = x - xd  # from drone centre to node
    dist = np.sqrt(np.sum(d * d, axis=1))
    delta = r_d - dist
    contact = active & (delta > 0.0) & (dist > _EPS)
    f_nodes = np.zeros_like(x)
    if not np.any(contact):
        return f_nodes, np.zeros(3), 0.0
    nhat = d[contact] / dist[contact, None]
    fmag = k_c * delta[contact] ** 1.5
    f = fmag[:, None] * nhat
    f_nodes[contact] = f
    f_drone = -f.sum(axis=0)
    energy = float(0.4 * k_c * np.sum(delta[contact] ** 2.5))
    return f_nodes, f_drone, energy


def sphere_segment_penalty(x: np.ndarray, seg_edges: np.ndarray,
                           intact: np.ndarray, xd: np.ndarray, r_d: float,
                           k_c: float, active_node: np.ndarray):
    """Optional sphere-segment Hertz penalty (point-segment distance).

    For each intact segment the closest point to the sphere centre is found; if
    it penetrates the sphere, a penalty force ``k_c delta^{3/2} n_hat`` is
    applied at that point and distributed to the two endpoints by linear
    weights. The reaction acts on the drone. Endpoints that are not ``active``
    (e.g. the gripped node) do not receive the node share, but the drone still
    gets the reaction.
    """
    f_nodes = np.zeros_like(x)
    f_drone = np.zeros(3)
    energy = 0.0
    a = x[seg_edges[:, 0]]
    b = x[seg_edges[:, 1]]
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    t = np.where(ab2 > _EPS,
                 np.sum((xd - a) * ab, axis=1) / np.maximum(ab2, _EPS), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[:, None] * ab
    d = closest - xd
    dist = np.sqrt(np.sum(d * d, axis=1))
    delta = r_d - dist
    hit = intact & (delta > 0.0) & (dist > _EPS)
    for s in np.nonzero(hit)[0]:
        nhat = d[s] / dist[s]
        f = k_c * delta[s] ** 1.5 * nhat
        i, j = seg_edges[s]
        wi, wj = 1.0 - t[s], t[s]
        if active_node[i]:
            f_nodes[i] += wi * f
        if active_node[j]:
            f_nodes[j] += wj * f
        f_drone -= f
        energy += 0.4 * k_c * delta[s] ** 2.5
    return f_nodes, f_drone, float(energy)
