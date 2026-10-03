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
                    q_ratio: float = 1.0, fix_radii: bool = True) -> Net:
    """Star with ``N`` radials plus concentric circumferential rings.

    ``radii`` is a sequence of ring radii in (0, R]; the outermost radius must
    equal ``R`` and its nodes are anchored. Ring nodes are *shared* with the
    radials (one node per radial/ring crossing, no overlapping threads).

    Prestress is set by the force-density method (FDM):

    * ``fix_radii=True`` (default, paper nets): keep ring nodes at the
      prescribed radii. Inner and outer radial segments of each ring get
      different force densities from radial equilibrium at the ring nodes,
      ``q_inner * r = q_outer * (R_next - r)`` (and analogously for multiple
      rings), with ``q_ring = q_ratio * q_radial_ref``. Rest lengths are then
      recovered from ``(q, geometry)``.
    * ``fix_radii=False``: free FDM solve with uniform ``q_radial`` and
      ``q_ring = q_ratio * q_radial``. The solved ring radii generally differ
      from the prescribed ones (rings are pulled inward).

    When ``material`` is ``None`` the net falls back to the uniform geometric
    prestress (``q = 0``).
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
    # Radial family tags for rest-length / eps reporting:
    # radial segment of ring-crossing j (0 = hub->first ring, ...).
    radial_family = []  # parallel to edges; -1 for ring edges, else family id
    # Radial threads (hub outward through successive ring nodes).
    for k in range(N):
        prev = 0
        prev_r = 0.0
        for j in range(n_rings):
            cur = ring_idx(j, k)
            edges.append((prev, cur))
            lengths.append(radii[j] - prev_r)
            is_ring.append(False)
            radial_family.append(j)  # family = which radial span
            prev, prev_r = cur, radii[j]
    # Circumferential (ring) threads between neighbouring radials.
    chord = 2.0 * np.sin(np.pi / N)
    for j in range(n_rings):
        for k in range(N):
            edges.append((ring_idx(j, k), ring_idx(j, k + 1)))
            lengths.append(radii[j] * chord)
            is_ring.append(True)
            radial_family.append(-1)
    edges = np.asarray(edges, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=float)
    is_ring = np.asarray(is_ring, dtype=bool)
    radial_family = np.asarray(radial_family, dtype=np.int64)

    anchored = np.zeros(nodes.shape[0], dtype=bool)
    for k in range(N):
        anchored[ring_idx(n_rings - 1, k)] = True

    A_arr = np.full(edges.shape[0], A_hat)

    if material is None:
        q = np.zeros(edges.shape[0])
        rest = lengths / (1.0 + eps_p)
        return Net(nodes=nodes, edges=edges, anchored=anchored, q=q,
                   A_hat=A_arr, rest_length=rest, R=R,
                   meta={"kind": "star_with_rings", "N": N, "eps_p": eps_p,
                         "radii": radii.tolist(), "q_ratio": q_ratio,
                         "fix_radii": fix_radii,
                         "is_ring": is_ring.tolist(),
                         "radial_family": radial_family.tolist()})

    from .fdm import fdm_equilibrium, rest_lengths_from_q

    sigma_p = float(material.sigma(eps_p))
    L_rad_mean = float(np.mean(lengths[~is_ring]))
    q_ref = A_hat * sigma_p / L_rad_mean  # reference radial prestress ~ eps_p

    if fix_radii:
        # Prescribed geometry: set radial q from ring-node radial equilibrium.
        # At a free ring of radius r_j between r_prev and r_next, with uniform
        # circumferential q (forces cancel by symmetry):
        #   q_in * (r_j - r_prev) = q_out * (r_next - r_j)
        # (the FDM force on the node along the radial is q * Delta_r).
        # We set the outermost radial span (to the anchored ring at R) as the
        # reference q_ref and propagate inward.
        r_levels = np.concatenate([[0.0], radii])  # hub, rings...
        # q_radial[j] = force density on the span between r_levels[j] and
        # r_levels[j+1], j = 0..n_rings-1.
        q_rad = np.zeros(n_rings)
        q_rad[-1] = q_ref
        for j in range(n_rings - 2, -1, -1):
            # At free ring j (radius radii[j] = r_levels[j+1]):
            # q_rad[j] * (r_levels[j+1] - r_levels[j])
            #   = q_rad[j+1] * (r_levels[j+2] - r_levels[j+1])
            dr_in = r_levels[j + 1] - r_levels[j]
            dr_out = r_levels[j + 2] - r_levels[j + 1]
            if dr_in <= 0:
                raise ValueError("ring radii must be strictly increasing")
            q_rad[j] = q_rad[j + 1] * dr_out / dr_in

        q = np.empty(edges.shape[0])
        for e in range(edges.shape[0]):
            if is_ring[e]:
                q[e] = q_ratio * q_ref
            else:
                q[e] = q_rad[radial_family[e]]
        nodes_eq = nodes.copy()  # keep prescribed radii
    else:
        q = np.where(is_ring, q_ratio * q_ref, q_ref)
        nodes_eq = fdm_equilibrium(nodes, edges, q, anchored)

    rest = rest_lengths_from_q(nodes_eq, edges, q, A_arr, material)

    # Per-family prestress strain for reporting.
    d = nodes_eq[edges[:, 1]] - nodes_eq[edges[:, 0]]
    l_cur = np.linalg.norm(d, axis=1)
    eps_e = l_cur / rest - 1.0
    family_eps = {}
    for j in range(n_rings):
        mask = (~is_ring) & (radial_family == j)
        family_eps[f"radial_{j}"] = float(np.mean(eps_e[mask])) if np.any(mask) \
            else float("nan")
    for j in range(n_rings - 1):  # free rings only (outer is frame)
        # ring edges for ring j occupy a contiguous block after all radials
        pass
    ring_eps = []
    n_rad_edges = N * n_rings
    for j in range(n_rings):
        sl = slice(n_rad_edges + j * N, n_rad_edges + (j + 1) * N)
        ring_eps.append(float(np.mean(eps_e[sl])))
        family_eps[f"ring_{j}"] = ring_eps[-1]

    return Net(nodes=nodes_eq, edges=edges, anchored=anchored, q=q,
               A_hat=A_arr, rest_length=rest, R=R,
               meta={"kind": "star_with_rings", "N": N, "eps_p": eps_p,
                     "radii": radii.tolist(), "q_ratio": q_ratio,
                     "fix_radii": fix_radii,
                     "is_ring": is_ring.tolist(),
                     "radial_family": radial_family.tolist(),
                     "family_eps": family_eps})


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
