"""External forces f_ext = W + F_v + F_r + F_l (forces.py), Eq. (eq:fext).

SI units throughout. Twist DOFs carry no external force.

Jacobian bookkeeping (fed to the integrator, Eq. (eq:jacobian)):
  - The viscous (RFT/Stokes) term contributes a resistance matrix ``C`` with
    d F_v / d xi = -C / dt (node tangents are *lagged*, i.e. taken from the
    current iterate but not differentiated, when ``lag_tangent`` is True).
  - The lift term F_l and Coulomb term F_r contribute analytic position
    derivatives assembled in :func:`external_position_jacobian`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .geometry import Topology
from .params import Params

Z_HAT = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------
def node_charges(P: Params, topo: Topology) -> np.ndarray:
    """Charge per node q_k (n_nodes,). q_0 = Q_s; thread charge per model."""
    q = np.zeros(topo.n_nodes)
    q[0] = P.Q_s
    if P.charge_model == "tip":
        for j in range(topo.N):
            q[topo.tip_nodes[j]] = P.Q_t
    elif P.charge_model == "uniform":
        dl = topo.node_voronoi_lengths()
        for j in range(topo.N):
            for node in topo.thread_nodes[j]:
                q[node] = P.Q_t * dl[node] / P.L
    else:
        raise ValueError(f"unknown charge_model {P.charge_model!r}")
    return q


# ---------------------------------------------------------------------------
# Node tangents for RFT
# ---------------------------------------------------------------------------
def node_tangents(topo: Topology, X: np.ndarray) -> np.ndarray:
    """Node-based tangents (normalized average of adjacent edge tangents).

    ``X`` is the (n_nodes, 3) node position array.
    """
    et = topo.edge_tangents(X)
    tvec = np.zeros((topo.n_nodes, 3))
    counts = np.zeros(topo.n_nodes)
    for k in range(topo.n_edges):
        na, nb = topo.edges[k]
        tvec[na] += et[k]
        tvec[nb] += et[k]
        counts[na] += 1
        counts[nb] += 1
    for node in range(topo.n_nodes):
        n = np.linalg.norm(tvec[node])
        if n > 0:
            tvec[node] /= n
    return tvec


# ---------------------------------------------------------------------------
# External force assembly
# ---------------------------------------------------------------------------
def assemble_fext(P: Params, topo: Topology, x: np.ndarray, v: np.ndarray,
                  field, flow, t: float, q: np.ndarray) -> np.ndarray:
    """Full external force vector f_ext (n_dof,)  # Eq. (eq:fext)."""
    f = np.zeros(topo.n_dof)
    dl = topo.node_voronoi_lengths()

    # positions/velocities as (n_nodes, 3) views
    X = x[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3)
    V = v[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3)
    tang = node_tangents(topo, X)

    for node in range(topo.n_nodes):
        fi = np.zeros(3)
        z = X[node, 2]

        # --- W: weight ---
        if node == 0:
            fi += -P.m * P.g * Z_HAT
        else:
            fi += -P.rho_t * P.A * dl[node] * P.g * Z_HAT

        # --- F_l: electrostatic lift ---
        fi += q[node] * field.E(z) * Z_HAT

        # --- F_v: viscous drag (Eq. (eq:Fvk); Stokes on the spider) ---
        u = flow.u(X[node], t)
        rel = V[node] - u
        if node == 0:
            fi += -P.zeta_s * rel
        else:
            tk = tang[node]
            proj = np.outer(tk, tk)
            resist = P.eta_par * proj + P.eta_perp * (np.eye(3) - proj)
            fi += -dl[node] * resist @ rel

        f[3 * node: 3 * node + 3] = fi

    # --- F_r: Coulomb repulsion (Eq. (eq:coulomb)), added to position DOFs ---
    _add_coulomb_force(P, topo, X, q, f)
    return f


def _charged_nodes(q: np.ndarray) -> np.ndarray:
    return np.where(q != 0.0)[0]


def _coulomb_source(P: Params, i: int, k: int) -> bool:
    """Whether node ``i`` contributes to the Coulomb force on node ``k``.

    Default (``coulomb_include_spider=False``): Habchi & Jawed sum over
    ``i != 0`` and ``i != k`` — the spider is never a source. With the flag
    set, every other node ``i != k`` is a source.
    """
    if i == k:
        return False
    if not P.coulomb_include_spider and i == 0:
        return False
    return True


def _add_coulomb_force(P: Params, topo: Topology, X: np.ndarray,
                       q: np.ndarray, f: np.ndarray) -> None:
    idx = _charged_nodes(q)
    soft = P.coulomb_softening
    for a in range(len(idx)):
        k = int(idx[a])
        acc = np.zeros(3)
        for b in range(len(idx)):
            i = int(idx[b])
            if not _coulomb_source(P, i, k):
                continue
            r = X[k] - X[i]
            dist = np.sqrt(r @ r + soft * soft)
            acc += q[i] * r / dist ** 3
        f[3 * k: 3 * k + 3] += P.k_e * q[k] * acc


# ---------------------------------------------------------------------------
# Jacobian pieces
# ---------------------------------------------------------------------------
def resistance_matrix(P: Params, topo: Topology, x: np.ndarray) -> sp.csr_matrix:
    """Block-diagonal RFT/Stokes resistance C (n_dof x n_dof), position DOFs only.

    Defined so that d F_v / d xi = -C / dt (lagged node tangents).
    """
    dl = topo.node_voronoi_lengths()
    X = x[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3)
    tang = node_tangents(topo, X)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for node in range(topo.n_nodes):
        if node == 0:
            block = P.zeta_s * np.eye(3)
        else:
            tk = tang[node]
            proj = np.outer(tk, tk)
            block = dl[node] * (P.eta_par * proj + P.eta_perp * (np.eye(3) - proj))
        base = 3 * node
        for a in range(3):
            for b in range(3):
                rows.append(base + a)
                cols.append(base + b)
                vals.append(block[a, b])
    return sp.coo_matrix((vals, (rows, cols)), shape=(topo.n_dof, topo.n_dof)).tocsr()


def viscous_flow_jacobian(P: Params, topo: Topology, x: np.ndarray,
                          flow, t: float):
    """Return d F_v / d xi from the spatial dependence of u(x, t).

    ``F_{v,k} = -D_k (v_k - u(x_k, t))`` with ``D_k`` the RFT/Stokes resistance
    tensor (times ``dl_k`` on the threads, ``zeta_s I`` on the spider). The tangent
    inside ``D_k`` stays *lagged* (as in :func:`resistance_matrix`), so the only
    position dependence differentiated here is ``u`` itself:

        d F_{v,k} / d x_k = -D_k * d(v_k - u)/d x_k = + D_k grad_u(x_k, t).

    For ``ZeroFlow`` and ``UniformFlow`` ``grad_u == 0``, so the computation is
    skipped and an all-zero matrix is returned (no change to existing behaviour).
    """
    from .fields import ZeroFlow, UniformFlow

    if isinstance(flow, (ZeroFlow, UniformFlow)):
        return sp.csr_matrix((topo.n_dof, topo.n_dof))

    dl = topo.node_voronoi_lengths()
    X = x[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3)
    tang = node_tangents(topo, X)
    G = flow.grad_u(X, t)  # (n_nodes, 3, 3)
    Id3 = np.eye(3)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for node in range(topo.n_nodes):
        if node == 0:
            D = P.zeta_s * Id3
        else:
            tk = tang[node]
            proj = np.outer(tk, tk)
            D = dl[node] * (P.eta_par * proj + P.eta_perp * (Id3 - proj))
        block = D @ G[node]  # + D_k grad_u
        base = 3 * node
        for a in range(3):
            for b in range(3):
                rows.append(base + a)
                cols.append(base + b)
                vals.append(block[a, b])
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(topo.n_dof, topo.n_dof)).tocsr()


def external_position_jacobian(P: Params, topo: Topology, x: np.ndarray,
                               field, q: np.ndarray):
    """Return d(F_l + F_r)/d xi as a sparse matrix (n_dof x n_dof).

    - F_l: d/dz_k of q_k E(z_k) contributes on the (z_k, z_k) entry.
    - F_r: analytic Coulomb Jacobian. Pairs follow ``_coulomb_source``
      (not symmetric when the spider is excluded as a source).
    """
    X = x[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # --- F_l position derivative ---
    for node in range(topo.n_nodes):
        if q[node] != 0.0:
            zdof = 3 * node + 2
            rows.append(zdof)
            cols.append(zdof)
            vals.append(q[node] * field.dE(X[node, 2]))

    # --- F_r Coulomb Jacobian ---
    idx = _charged_nodes(q)
    soft = P.coulomb_softening
    Id3 = np.eye(3)
    for a in range(len(idx)):
        k = int(idx[a])
        kk = np.zeros((3, 3))
        for b in range(len(idx)):
            i = int(idx[b])
            if not _coulomb_source(P, i, k):
                continue
            r = X[k] - X[i]
            dist = np.sqrt(r @ r + soft * soft)
            G = Id3 / dist ** 3 - 3.0 * np.outer(r, r) / dist ** 5
            block = P.k_e * q[k] * q[i] * G  # d F_{r,k} / d x_k contribution from i
            kk += block
            # off-diagonal d F_{r,k} / d x_i = -block
            bk = 3 * k
            bi = 3 * i
            for p in range(3):
                for r_ in range(3):
                    rows.append(bk + p)
                    cols.append(bi + r_)
                    vals.append(-block[p, r_])
        bk = 3 * k
        for p in range(3):
            for r_ in range(3):
                rows.append(bk + p)
                cols.append(bk + r_)
                vals.append(kk[p, r_])
    return sp.coo_matrix((vals, (rows, cols)), shape=(topo.n_dof, topo.n_dof)).tocsr()
