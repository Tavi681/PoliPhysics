"""Subdivide the coarse net into segments and lump masses at the nodes.

Each coarse thread ``e`` is split into ``n_s`` equal segments along the straight
line between its end nodes, with rest length ``ell_e / n_s``. Because the initial
geometry is the (pre)stressed equilibrium, this discrete state is exactly in
equilibrium. Masses are lumped at the nodes:

    m_i = sum_{e ni i} (1/2) rho A_e ell_e / n_s.

Coarse (graph) nodes keep their indices; interior segment nodes are appended.
Anchored coarse nodes stay fixed. Maps are kept so that the fine graph can be
reconstructed and each segment traced back to its parent thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger("netsim.discretize")

__all__ = ["Discretization", "discretize"]


@dataclass
class Discretization:
    """Fine (segment) discretization of a net.

    Attributes
    ----------
    x0:
        (n_node, 3) initial segment-node positions (z = 0 for coarse nodes).
    seg_edges:
        (n_seg, 2) segment connectivity into segment nodes.
    seg_parent:
        (n_seg,) parent coarse-thread index for each segment.
    seg_rest_length:
        (n_seg,) segment rest lengths ell_e / n_s.
    seg_A:
        (n_seg,) segment cross-sections (already area-scaled).
    mass:
        (n_node,) lumped nodal masses.
    anchored:
        (n_node,) boolean, True for fixed nodes.
    node_parent_thread:
        (n_node,) parent coarse-thread index for interior nodes, -1 for coarse
        nodes.
    node_coarse:
        (n_node,) coarse-node index for coarse nodes, -1 for interior nodes.
    node_thread_pos:
        (n_node,) fractional position along the parent thread for interior
        nodes (in (0, 1)), NaN for coarse nodes.
    n_s:
        Number of segments per thread.
    """

    x0: np.ndarray
    seg_edges: np.ndarray
    seg_parent: np.ndarray
    seg_rest_length: np.ndarray
    seg_A: np.ndarray
    mass: np.ndarray
    anchored: np.ndarray
    node_parent_thread: np.ndarray
    node_coarse: np.ndarray
    node_thread_pos: np.ndarray
    n_s: int

    @property
    def n_node(self) -> int:
        return self.x0.shape[0]

    @property
    def n_seg(self) -> int:
        return self.seg_edges.shape[0]


def discretize(net, material, n_s: int, *, area_scale: float = 1.0,
               r_d: float | None = None, warn_ratio: float = 0.25) -> Discretization:
    """Build the segment discretization of ``net``.

    Parameters
    ----------
    net:
        The coarse :class:`~netsim.topology.Net`.
    material:
        Material (for density -> lumped masses).
    n_s:
        Segments per thread (>= 1).
    area_scale:
        Cross-section scale factor s (A_e = s * A_hat_e).
    r_d:
        Drone radius; if given, warn when a segment spacing exceeds
        ``warn_ratio * r_d`` (default r_d / 4).
    """
    if n_s < 1:
        raise ValueError("n_s must be >= 1")
    nodes3 = net.nodes3()
    edges = net.edges
    A = net.A(area_scale)
    rho = material.rho

    positions = [nodes3[i] for i in range(net.n_v)]
    node_parent_thread = [-1] * net.n_v
    node_coarse = list(range(net.n_v))
    node_thread_pos = [np.nan] * net.n_v

    seg_edges: list[tuple[int, int]] = []
    seg_parent: list[int] = []
    seg_rest: list[float] = []
    seg_A: list[float] = []

    for e in range(net.n_e):
        i, j = int(edges[e, 0]), int(edges[e, 1])
        xi, xj = nodes3[i], nodes3[j]
        A_e = float(A[e])
        seg_rest_e = float(net.rest_length[e]) / n_s
        prev = i
        for s in range(1, n_s):
            t = s / n_s
            pos = xi + t * (xj - xi)
            idx = len(positions)
            positions.append(pos)
            node_parent_thread.append(e)
            node_coarse.append(-1)
            node_thread_pos.append(t)
            seg_edges.append((prev, idx))
            seg_parent.append(e)
            seg_rest.append(seg_rest_e)
            seg_A.append(A_e)
            prev = idx
        seg_edges.append((prev, j))
        seg_parent.append(e)
        seg_rest.append(seg_rest_e)
        seg_A.append(A_e)

    x0 = np.asarray(positions, dtype=float)
    seg_edges_a = np.asarray(seg_edges, dtype=np.int64)
    seg_parent_a = np.asarray(seg_parent, dtype=np.int64)
    seg_rest_a = np.asarray(seg_rest, dtype=float)
    seg_A_a = np.asarray(seg_A, dtype=float)

    # Lumped nodal masses: half of each segment's mass to each endpoint.
    n_node = x0.shape[0]
    mass = np.zeros(n_node)
    seg_mass = rho * seg_A_a * seg_rest_a
    np.add.at(mass, seg_edges_a[:, 0], 0.5 * seg_mass)
    np.add.at(mass, seg_edges_a[:, 1], 0.5 * seg_mass)

    anchored = np.zeros(n_node, dtype=bool)
    anchored[: net.n_v] = net.anchored

    # Discretization check.
    if r_d is not None:
        max_spacing = float(seg_rest_a.max())
        if max_spacing > warn_ratio * r_d:
            logger.warning(
                "segment spacing %.4g m exceeds %.2f*r_d = %.4g m; "
                "consider a larger n_s or r_d", max_spacing, warn_ratio,
                warn_ratio * r_d)
        else:
            logger.info("segment spacing %.4g m <= %.2f*r_d = %.4g m (ok)",
                        max_spacing, warn_ratio, warn_ratio * r_d)

    return Discretization(
        x0=x0,
        seg_edges=seg_edges_a,
        seg_parent=seg_parent_a,
        seg_rest_length=seg_rest_a,
        seg_A=seg_A_a,
        mass=mass,
        anchored=anchored,
        node_parent_thread=np.asarray(node_parent_thread, dtype=np.int64),
        node_coarse=np.asarray(node_coarse, dtype=np.int64),
        node_thread_pos=np.asarray(node_thread_pos, dtype=float),
        n_s=n_s,
    )
