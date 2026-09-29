"""Geometry, DOF packing and topology (geometry.py).

Degrees of freedom (as in the paper / objective):

    Node 0 = spider. Thread j (j = 1..N) has N_t edges and nodes
    x_{(j-1)N_t+1} ... x_{jN_t}; its first edge connects node 0 to
    x_{(j-1)N_t+1}. Total nodes n = N N_t + 1, edges N N_t.

    xi = [x_0, x_1, ..., x_{N N_t}, theta^0, ..., theta^{N N_t - 1}]
    n_dof = 3(N N_t + 1) + N N_t.
"""

from __future__ import annotations

import numpy as np

from .params import Params


class Topology:
    """Precomputed connectivity for a given (N, N_t)."""

    def __init__(self, P: Params):
        self.P = P
        self.N = P.N
        self.N_t = P.N_t
        self.n_nodes = P.n_nodes
        self.n_edges = P.n_edges
        self.n_dof = P.n_dof

        # edges[k] = (node_start, node_end) for global edge k
        edges = np.empty((self.n_edges, 2), dtype=np.int64)
        # per-thread list of edge indices and node indices
        self.thread_edges: list[np.ndarray] = []
        self.thread_nodes: list[np.ndarray] = []
        # thread id per node (-1 for spider)
        self.thread_id = np.full(self.n_nodes, -1, dtype=np.int64)
        self.tip_nodes = np.empty(self.N, dtype=np.int64)

        for j in range(self.N):
            base = j * self.N_t + 1  # first thread node index
            e0 = j * self.N_t  # first edge global index
            # first edge: node 0 -> base
            edges[e0] = (0, base)
            for i in range(1, self.N_t):
                edges[e0 + i] = (base + i - 1, base + i)
            node_ids = np.arange(base, base + self.N_t, dtype=np.int64)
            self.thread_nodes.append(node_ids)
            self.thread_edges.append(np.arange(e0, e0 + self.N_t, dtype=np.int64))
            self.thread_id[node_ids] = j
            self.tip_nodes[j] = base + self.N_t - 1

        self.edges = edges

        # interior nodes of a thread: nodes with two adjacent edges.
        # Per thread, the base node and all but the tip are "interior" for
        # bending/twist (they connect two edges). The tip has a single edge.
        # Bending/twist interior nodes: local index 1..N_t-1 correspond to the
        # node shared by edge (i-1) and edge (i) inside the thread.
        # We store, per thread, the list of (node, edge_prev, edge_next).
        self.interior: list[list[tuple[int, int, int]]] = []
        for j in range(self.N):
            eidx = self.thread_edges[j]
            entries = []
            for i in range(1, self.N_t):
                # node shared by edge eidx[i-1] and eidx[i]
                node = int(self.edges[eidx[i]][0])
                entries.append((node, int(eidx[i - 1]), int(eidx[i])))
            self.interior.append(entries)

    # --- DOF slicing -----------------------------------------------------
    def positions(self, xi: np.ndarray) -> np.ndarray:
        """Return the (n_nodes, 3) node positions view from a DOF vector."""
        return xi[: 3 * self.n_nodes].reshape(self.n_nodes, 3)

    def thetas(self, xi: np.ndarray) -> np.ndarray:
        """Return the (n_edges,) twist angles view from a DOF vector."""
        return xi[3 * self.n_nodes:]

    def pos_slice(self, node: int) -> slice:
        """DOF slice for a node's 3 translational coordinates."""
        return slice(3 * node, 3 * node + 3)

    def theta_index(self, edge: int) -> int:
        """DOF index of a twist angle."""
        return 3 * self.n_nodes + edge

    # --- edge geometry ---------------------------------------------------
    def edge_vectors(self, x: np.ndarray) -> np.ndarray:
        """Edge vectors e_k = x_end - x_start, shape (n_edges, 3)."""
        return x[self.edges[:, 1]] - x[self.edges[:, 0]]

    def edge_lengths(self, x: np.ndarray) -> np.ndarray:
        """Edge lengths |e_k|, shape (n_edges,)."""
        return np.linalg.norm(self.edge_vectors(x), axis=1)

    def edge_tangents(self, x: np.ndarray) -> np.ndarray:
        """Unit edge tangents t_k = e_k / |e_k|, shape (n_edges, 3)."""
        e = self.edge_vectors(x)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def node_voronoi_lengths(self) -> np.ndarray:
        """Undeformed Voronoi length dl_k per node (0 for the spider node).

        Interior thread node -> l0, tip node -> l0/2. The spider node uses
        Stokes drag (not RFT), so its Voronoi length is 0.
        """
        l0 = self.P.l0
        dl = np.zeros(self.n_nodes)
        for j in range(self.N):
            for node in self.thread_nodes[j]:
                dl[node] = l0
            dl[self.tip_nodes[j]] = 0.5 * l0
        return dl


def initial_state(P: Params) -> tuple[np.ndarray, np.ndarray, Topology]:
    """Build the initial DOF vector xi^0 and velocity xi_dot^0 = 0.

    Threads are vertical, their bases spaced on a circle of radius d0 around
    the spider (radius 0 for a single thread), pointing up; theta = 0.
    """
    P.validate()
    topo = Topology(P)
    xi = np.zeros(P.n_dof)
    x = topo.positions(xi)

    # spider at (0, 0, z0)
    x[0] = (0.0, 0.0, P.z0)

    radius = P.d0 if P.N > 1 else 0.0
    l0 = P.l0
    for j in range(P.N):
        phi = 2.0 * np.pi * j / P.N
        cx = radius * np.cos(phi)
        cy = radius * np.sin(phi)
        for i in range(1, P.N_t + 1):
            node = j * P.N_t + i
            x[node] = (cx, cy, P.z0 + i * l0)

    xi_dot = np.zeros(P.n_dof)
    return xi, xi_dot, topo
