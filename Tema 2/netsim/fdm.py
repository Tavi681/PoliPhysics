"""Force-density prestress solve and rest-length recovery.

The force-density method (FDM) finds the equilibrium geometry of a net whose
threads carry given force densities ``q_e = T_e / l_e``. With the graph Laplacian
``L_q`` weighted by ``q_e``, free-node positions ``x_f`` satisfy

    L_q^{ff} x_f = - L_q^{fa} x_a,

where ``x_a`` are the fixed anchor positions. Once the geometry is known:

    l_e   = current thread length,
    T_e   = q_e * l_e,
    eps_e = inverse of T(eps_e) = T_e  (Newton on the cubic, 0 <= eps < eps_b),
    ell_e = l_e / (1 + eps_e)          (rest length).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["weighted_laplacian", "fdm_equilibrium", "rest_lengths_from_q"]


def incidence_matrix(edges: np.ndarray, n_v: int) -> sp.csr_matrix:
    """Signed edge-node incidence matrix C (n_e, n_v)."""
    n_e = edges.shape[0]
    rows = np.repeat(np.arange(n_e), 2)
    cols = edges.reshape(-1)
    data = np.tile(np.array([1.0, -1.0]), n_e)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_e, n_v))


def weighted_laplacian(edges: np.ndarray, q: np.ndarray, n_v: int) -> sp.csr_matrix:
    """Graph Laplacian L_q = C^T diag(q) C weighted by force densities q."""
    C = incidence_matrix(edges, n_v)
    Q = sp.diags(np.asarray(q, dtype=float))
    return (C.T @ Q @ C).tocsr()


def fdm_equilibrium(nodes: np.ndarray, edges: np.ndarray, q: np.ndarray,
                    anchored: np.ndarray) -> np.ndarray:
    """Solve for free-node positions given anchor positions and force densities.

    ``nodes`` supplies the anchor positions (rows where ``anchored`` is True);
    free-node rows are used only as an initial layout and are overwritten by the
    solution. Returns the full node array with equilibrium free positions.
    """
    nodes = np.asarray(nodes, dtype=float)
    anchored = np.asarray(anchored, dtype=bool)
    n_v = nodes.shape[0]
    free = ~anchored
    L = weighted_laplacian(edges, q, n_v)
    Lff = L[free][:, free]
    Lfa = L[free][:, anchored]
    xa = nodes[anchored]
    rhs = -(Lfa @ xa)
    xf = spla.spsolve(Lff.tocsc(), rhs)
    xf = np.asarray(xf).reshape(free.sum(), -1)
    out = nodes.copy()
    out[free] = xf
    return out


def rest_lengths_from_q(nodes: np.ndarray, edges: np.ndarray, q: np.ndarray,
                        A: np.ndarray, material) -> np.ndarray:
    """Recover thread rest lengths from the prestressed geometry.

    For each edge: l_e = |x_j - x_i|, T_e = q_e * l_e, eps_e solves
    A_e*(E0 eps + b eps^3) = T_e (checked in [0, eps_b)), ell_e = l_e/(1+eps_e).
    """
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=np.int64)
    q = np.asarray(q, dtype=float)
    A = np.asarray(A, dtype=float)
    d = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    l = np.linalg.norm(d, axis=1)
    T = q * l
    eps = np.empty_like(l)
    for e in range(edges.shape[0]):
        eps[e] = material.strain_from_tension(T[e], A[e])
    return l / (1.0 + eps)
