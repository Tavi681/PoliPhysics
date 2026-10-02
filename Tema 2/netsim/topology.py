"""Coarse net topology: the planar graph G = (V, E).

A :class:`Net` stores the prestressed planar net as node positions in the plane
z = 0, thread connectivity, anchored flags, force densities ``q`` and *nominal*
cross-sections ``A_hat``. Cross-sections used by the simulation are ``A = s *
A_hat`` where ``s`` is an area-scale factor, so that Algorithm 2 (minimum-mass
bisection, not implemented here) can rescale all cross-sections without
rebuilding the topology.

Constructors provided:

* :func:`star` - N radial threads from a central hub, uniform prestress eps_p.
* :func:`star_with_rings` - a star plus concentric circumferential threads.
* :func:`from_arrays` - build directly from user arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Net", "star", "star_with_rings", "from_arrays"]


@dataclass
class Net:
    """A prestressed planar net.

    Attributes
    ----------
    nodes:
        (n_v, 2) node positions in the plane z = 0.
    edges:
        (n_e, 2) integer connectivity (thread end-node indices).
    anchored:
        (n_v,) boolean, True for nodes fixed to the rigid frame.
    q:
        (n_e,) force densities [N/m].
    A_hat:
        (n_e,) nominal cross-sections [m^2].
    rest_length:
        (n_e,) thread rest lengths ell_e [m].
    R:
        Frame radius [m] (metadata, used by outputs / cascade criteria).
    """

    nodes: np.ndarray
    edges: np.ndarray
    anchored: np.ndarray
    q: np.ndarray
    A_hat: np.ndarray
    rest_length: np.ndarray
    R: float = 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 2)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.anchored = np.asarray(self.anchored, dtype=bool).reshape(-1)
        self.q = np.asarray(self.q, dtype=float).reshape(-1)
        self.A_hat = np.asarray(self.A_hat, dtype=float).reshape(-1)
        self.rest_length = np.asarray(self.rest_length, dtype=float).reshape(-1)
        n_v = self.nodes.shape[0]
        n_e = self.edges.shape[0]
        if self.anchored.shape[0] != n_v:
            raise ValueError("anchored must have length n_v")
        for name, arr in (("q", self.q), ("A_hat", self.A_hat),
                          ("rest_length", self.rest_length)):
            if arr.shape[0] != n_e:
                raise ValueError(f"{name} must have length n_e = {n_e}")
        if self.edges.min(initial=0) < 0 or (n_e and self.edges.max() >= n_v):
            raise ValueError("edge indices out of range")

    @property
    def n_v(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_e(self) -> int:
        return self.edges.shape[0]

    def A(self, area_scale: float = 1.0) -> np.ndarray:
        """Return cross-sections A = area_scale * A_hat."""
        return area_scale * self.A_hat

    def net_mass(self, rho: float, area_scale: float = 1.0) -> float:
        """Net mass m_net = sum_e rho * A_e * ell_e."""
        return float(np.sum(rho * self.A(area_scale) * self.rest_length))

    def nodes3(self) -> np.ndarray:
        """Node positions as (n_v, 3), with z = 0."""
        p = np.zeros((self.n_v, 3))
        p[:, :2] = self.nodes
        return p


def _q_from_prestress(material, eps_p: float, length: np.ndarray,
                      A_hat: np.ndarray) -> np.ndarray:
    """Force density q = T / l for uniform prestress strain eps_p."""
    sigma = material.sigma(eps_p)
    T = A_hat * sigma  # nominal tension at the reference (unscaled) area
    return T / length


def star(N: int, R: float, eps_p: float, *, material=None,
         A_hat: float = 1e-6) -> Net:
    """Star with ``N`` radial threads and uniform prestress ``eps_p``.

    Hub at the origin; ``N`` anchors evenly spaced on the circle of radius ``R``
    in the plane z = 0; rest length ell = R / (1 + eps_p) for every radial.

    ``material`` is only needed to fill the force-density array ``q`` (for HDF5
    provenance); if omitted, ``q`` is set to zero.
    """
    if N < 2:
        raise ValueError("star requires N >= 2")
    phi = 2.0 * np.pi * np.arange(N) / N
    anchors = np.c_[R * np.cos(phi), R * np.sin(phi)]
    nodes = np.vstack([[0.0, 0.0], anchors])  # node 0 = hub
    edges = np.c_[np.zeros(N, dtype=np.int64), np.arange(1, N + 1)]
    anchored = np.zeros(N + 1, dtype=bool)
    anchored[1:] = True
    length = np.full(N, R)
    rest = length / (1.0 + eps_p)
    A_arr = np.full(N, A_hat)
    if material is not None:
        q = _q_from_prestress(material, eps_p, length, A_arr)
    else:
        q = np.zeros(N)
    return Net(nodes=nodes, edges=edges, anchored=anchored, q=q,
               A_hat=A_arr, rest_length=rest, R=R,
               meta={"kind": "star", "N": N, "eps_p": eps_p})


def star_with_rings(N: int, R: float, radii, eps_p: float, *,
                    material=None, A_hat: float = 1e-6,
                    q_ratio: float = 1.0) -> Net:
    """Star with ``N`` radials plus concentric circumferential rings.

    ``radii`` is a sequence of ring radii in (0, R]; the outermost radius must
    equal ``R`` and its nodes are anchored. Ring nodes are *shared* with the
    radials (one node per radial/ring crossing, no overlapping threads).

    The ring prestress is set by the force-density method (FDM): the radial
    threads are given a force density ``q_radial`` (chosen so the radials sit at
    about the prestress strain ``eps_p``) and the rings a force density
    ``q_ring = q_ratio * q_radial``. The equilibrium geometry is solved from the
    force densities with the outer ring anchored, and the rest lengths are
    recovered from the solved geometry (:func:`netsim.fdm.rest_lengths_from_q`).
    With nonzero ``q_ratio`` the inner ring radii therefore come out of the FDM
    solve, not from the supplied ``radii`` (which set the number of rings and the
    initial layout). When ``material`` is ``None`` the net falls back to the
    uniform geometric prestress (``q = 0``).
    """
    if N < 2:
        raise ValueError("star_with_rings requires N >= 2")
    radii = np.asarray(sorted(float(r) for r in radii), dtype=float)
    if radii.size == 0 or radii[0] <= 0.0:
        raise ValueError("radii must be positive")
    if not np.isclose(radii[-1], R):
        raise ValueError("outermost ring radius must equal R")
    n_rings = radii.size
    phi = 2.0 * np.pi * np.arange(N) / N
    cos, sin = np.cos(phi), np.sin(phi)

    # Node 0 = hub, then ring j node for radial k at index 1 + j*N + k.
    node_list = [[0.0, 0.0]]
    for j in range(n_rings):
        for k in range(N):
            node_list.append([radii[j] * cos[k], radii[j] * sin[k]])
    nodes = np.asarray(node_list, dtype=float)

    def ring_idx(j, k):
        return 1 + j * N + (k % N)

    edges = []
    lengths = []
    is_ring = []
    # Radial threads (hub outward through successive ring nodes).
    for k in range(N):
        prev = 0
        prev_r = 0.0
        for j in range(n_rings):
            cur = ring_idx(j, k)
            edges.append((prev, cur))
            lengths.append(radii[j] - prev_r)
            is_ring.append(False)
            prev, prev_r = cur, radii[j]
    # Circumferential (ring) threads between neighbouring radials.
    chord = 2.0 * np.sin(np.pi / N)
    for j in range(n_rings):
        for k in range(N):
            edges.append((ring_idx(j, k), ring_idx(j, k + 1)))
            lengths.append(radii[j] * chord)
            is_ring.append(True)
    edges = np.asarray(edges, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=float)
    is_ring = np.asarray(is_ring, dtype=bool)

    anchored = np.zeros(nodes.shape[0], dtype=bool)
    for k in range(N):
        anchored[ring_idx(n_rings - 1, k)] = True

    A_arr = np.full(edges.shape[0], A_hat)

    if material is None:
        q = np.zeros(edges.shape[0])
        rest = lengths / (1.0 + eps_p)
        meta_q = {"q_ratio": q_ratio}
        return Net(nodes=nodes, edges=edges, anchored=anchored, q=q,
                   A_hat=A_arr, rest_length=rest, R=R,
                   meta={"kind": "star_with_rings", "N": N, "eps_p": eps_p,
                         "radii": radii.tolist(), **meta_q})

    # --- Force-density prestress -----------------------------------------------
    from .fdm import fdm_equilibrium, rest_lengths_from_q

    sigma_p = float(material.sigma(eps_p))
    L_rad_mean = float(np.mean(lengths[~is_ring]))
    q_radial = A_hat * sigma_p / L_rad_mean  # radials ~ eps_p prestress
    q = np.where(is_ring, q_ratio * q_radial, q_radial)

    nodes_eq = fdm_equilibrium(nodes, edges, q, anchored)
    rest = rest_lengths_from_q(nodes_eq, edges, q, A_arr, material)

    return Net(nodes=nodes_eq, edges=edges, anchored=anchored, q=q,
               A_hat=A_arr, rest_length=rest, R=R,
               meta={"kind": "star_with_rings", "N": N, "eps_p": eps_p,
                     "radii": radii.tolist(), "q_ratio": q_ratio,
                     "is_ring": is_ring.tolist()})


def from_arrays(nodes, edges, anchored, q, A, *, rest_length=None,
                R: float = 1.0, material=None) -> Net:
    """Build a :class:`Net` directly from arrays.

    If ``rest_length`` is None it is obtained from the force-density prestress
    (requires ``material``): see :func:`netsim.fdm.rest_lengths_from_q`.
    """
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 2)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    q = np.asarray(q, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float).reshape(-1)
    if rest_length is None:
        from .fdm import rest_lengths_from_q

        if material is None:
            raise ValueError("rest_length omitted requires a material")
        rest_length = rest_lengths_from_q(nodes, edges, q, A, material)
    return Net(nodes=nodes, edges=edges, anchored=anchored, q=q,
               A_hat=A, rest_length=rest_length, R=R,
               meta={"kind": "from_arrays"})
