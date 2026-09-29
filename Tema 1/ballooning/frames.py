"""Reference and material frames (frames.py).

Adapted-frame machinery following Bergou et al. (2008, 2010) and the pedagogical
implementation in Jawed, Novelia & O'Reilly, "A Primer on the Kinematics of
Discrete Elastic Rods" (2018).

- Reference directors (d1, d2) per edge, updated each Newton iteration by
  *time* parallel transport from the previous converged step (Alg.1 line 6).
- Reference twist at each interior (bending) node from the transported frames.
- Material frame: m1 = cos(theta) d1 + sin(theta) d2,
                  m2 = -sin(theta) d1 + cos(theta) d2.

Bending/twist stencils never cross node 0 between threads: the first edge of each
thread (spider -> base) couples to node 0 only through stretching, so the
junction node (base) is a *free* joint and is excluded from bending/twist.
"""

from __future__ import annotations

import numpy as np

from .vec3 import cross

from .geometry import Topology


# ---------------------------------------------------------------------------
# Low-level frame utilities (ported from the DER primer MATLAB helpers)
# ---------------------------------------------------------------------------
def parallel_transport(u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Parallel transport vector ``u`` from tangent ``t1`` to tangent ``t2``."""
    b = cross(t1, t2)
    nb = np.linalg.norm(b)
    if nb == 0.0:
        return u.copy()
    b = b / nb
    b = b - np.dot(b, t1) * t1
    b = b / np.linalg.norm(b)
    b = b - np.dot(b, t2) * t2
    b = b / np.linalg.norm(b)
    n1 = cross(t1, b)
    n2 = cross(t2, b)
    return np.dot(u, t1) * t2 + np.dot(u, n1) * n2 + np.dot(u, b) * b


def rotate_axis_angle(v: np.ndarray, z: np.ndarray, theta: float) -> np.ndarray:
    """Rotate vector ``v`` about unit axis ``z`` by angle ``theta`` (Rodrigues)."""
    if theta == 0.0:
        return v.copy()
    c = np.cos(theta)
    s = np.sin(theta)
    return c * v + s * cross(z, v) + np.dot(z, v) * (1.0 - c) * z


def signed_angle(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> float:
    """Signed angle from ``u`` to ``v`` measured about axis ``n``."""
    w = cross(u, v)
    angle = np.arctan2(np.linalg.norm(w), np.dot(u, v))
    if np.dot(n, w) < 0.0:
        angle = -angle
    return angle


def _orthonormal_perp(t: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to ``t``."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, t)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    d1 = ref - np.dot(ref, t) * t
    return d1 / np.linalg.norm(d1)


def init_reference_directors(topo: Topology, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Initial adapted reference directors (space-parallel along each thread)."""
    t = topo.edge_tangents(x)
    d1 = np.zeros((topo.n_edges, 3))
    d2 = np.zeros((topo.n_edges, 3))
    for j in range(topo.N):
        eidx = topo.thread_edges[j]
        seed = _orthonormal_perp(t[eidx[0]])
        d1[eidx[0]] = seed
        d2[eidx[0]] = cross(t[eidx[0]], seed)
        for k in range(1, len(eidx)):
            e_prev, e_cur = eidx[k - 1], eidx[k]
            d = parallel_transport(d1[e_prev], t[e_prev], t[e_cur])
            d = d - np.dot(d, t[e_cur]) * t[e_cur]
            d = d / np.linalg.norm(d)
            d1[e_cur] = d
            d2[e_cur] = cross(t[e_cur], d)
    return d1, d2


def time_parallel_transport(
    topo: Topology, d1_conv: np.ndarray, t_conv: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transport converged reference directors onto the current tangents."""
    t_cur = topo.edge_tangents(x)
    d1 = np.zeros_like(d1_conv)
    d2 = np.zeros_like(d1_conv)
    for e in range(topo.n_edges):
        d = parallel_transport(d1_conv[e], t_conv[e], t_cur[e])
        d = d - np.dot(d, t_cur[e]) * t_cur[e]
        d = d / np.linalg.norm(d)
        d1[e] = d
        d2[e] = cross(t_cur[e], d)
    return d1, d2


def material_directors(
    topo: Topology, d1: np.ndarray, d2: np.ndarray, thetas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Material directors m1, m2 from reference directors and twist angles."""
    cs = np.cos(thetas)[:, None]
    ss = np.sin(thetas)[:, None]
    m1 = cs * d1 + ss * d2
    m2 = -ss * d1 + cs * d2
    return m1, m2


def compute_reference_twist(
    topo: Topology,
    d1: np.ndarray,
    x: np.ndarray,
    ref_twist_prev: np.ndarray,
) -> np.ndarray:
    """Reference twist per interior vertex (indexed 0..N_t-2 within each thread).

    ``ref_twist_prev`` and the return value are stored per thread as arrays of
    length ``N_t - 1`` (one entry per interior vertex ``i = 1..N_t-1``; the entry
    at local index 0 is the free spider junction and is unused for energy).
    """
    t = topo.edge_tangents(x)
    ref_twist = np.zeros_like(ref_twist_prev)
    for j in range(topo.N):
        for local_i, (node, e_prev, e_next) in enumerate(topo.interior[j]):
            u0 = d1[e_prev]
            u1 = d1[e_next]
            t0 = t[e_prev]
            t1 = t[e_next]
            ut = parallel_transport(u0, t0, t1)
            ut = rotate_axis_angle(ut, t1, ref_twist_prev[j][local_i])
            ref_twist[j][local_i] = ref_twist_prev[j][local_i] + signed_angle(ut, u1, t1)
    return ref_twist


def zero_reference_twist(topo: Topology) -> np.ndarray:
    """Allocate zeroed per-thread reference twist arrays."""
    return np.array([np.zeros(topo.N_t - 1) for _ in range(topo.N)], dtype=object)


class FrameState:
    """Holds converged reference-frame data and produces per-iterate frames.

    Usage in the integrator:
      - build once from the initial positions,
      - each Newton iteration call :meth:`evaluate` with the current DOFs,
      - on step acceptance call :meth:`commit` with the accepted DOFs.
    """

    def __init__(self, topo: Topology, x0: np.ndarray):
        self.topo = topo
        d1, d2 = init_reference_directors(topo, x0)
        self.d1_conv = d1
        self.d2_conv = d2
        self.t_conv = topo.edge_tangents(x0)
        self.ref_twist_conv = zero_reference_twist(topo)

    def evaluate(self, x: np.ndarray, thetas: np.ndarray):
        """Return (d1, d2, m1, m2, ref_twist) for the current Newton iterate."""
        d1, d2 = time_parallel_transport(self.topo, self.d1_conv, self.t_conv, x)
        ref_twist = compute_reference_twist(self.topo, d1, x, self.ref_twist_conv)
        m1, m2 = material_directors(self.topo, d1, d2, thetas)
        return d1, d2, m1, m2, ref_twist

    def commit(self, x: np.ndarray, thetas: np.ndarray) -> None:
        """Freeze the reference frame at the accepted configuration."""
        d1, d2 = time_parallel_transport(self.topo, self.d1_conv, self.t_conv, x)
        ref_twist = compute_reference_twist(self.topo, d1, x, self.ref_twist_conv)
        self.d1_conv = d1
        self.d2_conv = d2
        self.t_conv = self.topo.edge_tangents(x)
        self.ref_twist_conv = ref_twist
