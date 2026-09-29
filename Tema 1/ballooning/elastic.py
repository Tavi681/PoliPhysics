"""Elastic energy E_el = E_s + E_b + E_t with analytic gradients and Hessians.

Discrete Elastic Rods (Bergou et al. 2008). The per-stencil gradient and Hessian
expressions are ported from the DER primer reference implementation
(Jawed, Novelia & O'Reilly 2018, ``getFs`` / ``getFb`` / ``getFt``) and adapted
to this project's DOF layout and multi-thread topology.

Energy conventions (from ``tema1_objective.md``):
    E_s = 1/2 * Y A * sum_k (|e_k|/l0 - 1)^2 * l0          # Eq. (eq:Es)
    E_b = sum_i (Y I / l_i) |kappa_i - kappa0_i|^2         # Eq. (eq:Eb), kappa0 = 0
    E_t = sum_i (G J / l_i) m_i^2                          # Eq. (eq:Et)

To reuse the primer's half-quadratic form (E = sum 1/(2 l) q^T B q) the effective
stiffnesses are EA = Y A, EI = 2 Y I, GJ = 2 G J so the coefficients match the
objective's per-node energies exactly.

Each thread's first edge (spider -> base) couples through *stretching only*; the
base junction is a free joint, hence excluded from bending and twist.
"""

from __future__ import annotations

import numpy as np

import scipy.sparse as sp

from .vec3 import cross

from .geometry import Topology
from .params import Params

# Local stencil DOF layout (11 entries):
#   [0:3] x_prev, [3] theta_eprev, [4:7] x_curr, [7] theta_enext, [8:11] x_next
_A = slice(0, 3)
_TE = 3
_B = slice(4, 7)
_TF = 7
_C = slice(8, 11)


def _cross_mat(a: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


# ---------------------------------------------------------------------------
# Stretching, Eq. (eq:Es)
# ---------------------------------------------------------------------------
def stretch_energy_grad_hess(P: Params, topo: Topology, x: np.ndarray):
    """Return (E_s, grad (n_dof,), Hess COO lists rows, cols, vals)."""
    EA = P.Y * P.A
    l0 = P.l0
    n_dof = topo.n_dof
    grad = np.zeros(n_dof)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    E = 0.0
    Id3 = np.eye(3)

    e = topo.edge_vectors(x)
    lengths = np.linalg.norm(e, axis=1)
    for k in range(topo.n_edges):
        na, nb = topo.edges[k]
        length = lengths[k]
        t = e[k] / length
        eps = length / l0 - 1.0
        E += 0.5 * EA * eps ** 2 * l0

        gvec = EA * eps * t  # d E_s / d x_end
        ga = slice(3 * na, 3 * na + 3)
        gb = slice(3 * nb, 3 * nb + 3)
        grad[ga] -= gvec
        grad[gb] += gvec

        M = EA * ((1.0 / l0 - 1.0 / length) * Id3 + (1.0 / length) * np.outer(t, t))
        gi = [3 * na, 3 * na + 1, 3 * na + 2, 3 * nb, 3 * nb + 1, 3 * nb + 2]
        block = np.block([[M, -M], [-M, M]])
        for a in range(6):
            for b in range(6):
                rows.append(gi[a])
                cols.append(gi[b])
                vals.append(block[a, b])
    return E, grad, (rows, cols, vals)


# ---------------------------------------------------------------------------
# Per-vertex kappa gradient / Hessian (bending)
# ---------------------------------------------------------------------------
def _kappa_grad_hess(te, tf, ne, nf, m1e, m2e, m1f, m2f, kb, kappa1, kappa2):
    Id3 = np.eye(3)
    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    tilde_d1 = (m1e + m1f) / chi
    tilde_d2 = (m2e + m2f) / chi

    gradK = np.zeros((11, 2))
    Dk1De = (1.0 / ne) * (-kappa1 * tilde_t + cross(tf, tilde_d2))
    Dk1Df = (1.0 / nf) * (-kappa1 * tilde_t - cross(te, tilde_d2))
    Dk2De = (1.0 / ne) * (-kappa2 * tilde_t - cross(tf, tilde_d1))
    Dk2Df = (1.0 / nf) * (-kappa2 * tilde_t + cross(te, tilde_d1))
    gradK[_A, 0] = -Dk1De
    gradK[_B, 0] = Dk1De - Dk1Df
    gradK[_C, 0] = Dk1Df
    gradK[_A, 1] = -Dk2De
    gradK[_B, 1] = Dk2De - Dk2Df
    gradK[_C, 1] = Dk2Df
    gradK[_TE, 0] = -0.5 * np.dot(kb, m1e)
    gradK[_TF, 0] = -0.5 * np.dot(kb, m1f)
    gradK[_TE, 1] = -0.5 * np.dot(kb, m2e)
    gradK[_TF, 1] = -0.5 * np.dot(kb, m2f)

    norm2_e = ne * ne
    norm2_f = nf * nf
    tt_o_tt = np.outer(tilde_t, tilde_t)

    # --- kappa1 Hessian ---
    tmp = cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_tf_c_d2t = tf_c_d2t_o_tt.T
    kb_o_d2e = np.outer(kb, m2e)
    d2e_o_kb = kb_o_d2e.T
    D2k1De2 = (
        (1.0 / norm2_e) * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t)
        - kappa1 / (chi * norm2_e) * (Id3 - np.outer(te, te))
        + (1.0 / (4.0 * norm2_e)) * (kb_o_d2e + d2e_o_kb)
    )
    tmp = cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d2t = te_c_d2t_o_tt.T
    kb_o_d2f = np.outer(kb, m2f)
    d2f_o_kb = kb_o_d2f.T
    D2k1Df2 = (
        (1.0 / norm2_f) * (2 * kappa1 * tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t)
        - kappa1 / (chi * norm2_f) * (Id3 - np.outer(tf, tf))
        + (1.0 / (4.0 * norm2_f)) * (kb_o_d2f + d2f_o_kb)
    )
    D2k1DeDf = (
        -kappa1 / (chi * ne * nf) * (Id3 + np.outer(te, tf))
        + (1.0 / (ne * nf))
        * (2 * kappa1 * tt_o_tt - tf_c_d2t_o_tt + tt_o_te_c_d2t - _cross_mat(tilde_d2))
    )
    D2k1DfDe = D2k1DeDf.T

    # --- kappa2 Hessian ---
    tmp = cross(tf, tilde_d1)
    tf_c_d1t_o_tt = np.outer(tmp, tilde_t)
    tt_o_tf_c_d1t = tf_c_d1t_o_tt.T
    kb_o_d1e = np.outer(kb, m1e)
    d1e_o_kb = kb_o_d1e.T
    D2k2De2 = (
        (1.0 / norm2_e) * (2 * kappa2 * tt_o_tt + tf_c_d1t_o_tt + tt_o_tf_c_d1t)
        - kappa2 / (chi * norm2_e) * (Id3 - np.outer(te, te))
        - (1.0 / (4.0 * norm2_e)) * (kb_o_d1e + d1e_o_kb)
    )
    tmp = cross(te, tilde_d1)
    te_c_d1t_o_tt = np.outer(tmp, tilde_t)
    tt_o_te_c_d1t = te_c_d1t_o_tt.T
    kb_o_d1f = np.outer(kb, m1f)
    d1f_o_kb = kb_o_d1f.T
    D2k2Df2 = (
        (1.0 / norm2_f) * (2 * kappa2 * tt_o_tt - te_c_d1t_o_tt - tt_o_te_c_d1t)
        - kappa2 / (chi * norm2_f) * (Id3 - np.outer(tf, tf))
        - (1.0 / (4.0 * norm2_f)) * (kb_o_d1f + d1f_o_kb)
    )
    D2k2DeDf = (
        -kappa2 / (chi * ne * nf) * (Id3 + np.outer(te, tf))
        + (1.0 / (ne * nf))
        * (2 * kappa2 * tt_o_tt + tf_c_d1t_o_tt - tt_o_te_c_d1t + _cross_mat(tilde_d1))
    )
    D2k2DfDe = D2k2DeDf.T

    # theta second derivatives
    D2k1Dthetae2 = -0.5 * np.dot(kb, m2e)
    D2k1Dthetaf2 = -0.5 * np.dot(kb, m2f)
    D2k2Dthetae2 = 0.5 * np.dot(kb, m1e)
    D2k2Dthetaf2 = 0.5 * np.dot(kb, m1f)

    D2k1DeDthetae = (1.0 / ne) * (0.5 * np.dot(kb, m1e) * tilde_t - (1.0 / chi) * cross(tf, m1e))
    D2k1DeDthetaf = (1.0 / ne) * (0.5 * np.dot(kb, m1f) * tilde_t - (1.0 / chi) * cross(tf, m1f))
    D2k1DfDthetae = (1.0 / nf) * (0.5 * np.dot(kb, m1e) * tilde_t + (1.0 / chi) * cross(te, m1e))
    D2k1DfDthetaf = (1.0 / nf) * (0.5 * np.dot(kb, m1f) * tilde_t + (1.0 / chi) * cross(te, m1f))
    D2k2DeDthetae = (1.0 / ne) * (0.5 * np.dot(kb, m2e) * tilde_t - (1.0 / chi) * cross(tf, m2e))
    D2k2DeDthetaf = (1.0 / ne) * (0.5 * np.dot(kb, m2f) * tilde_t - (1.0 / chi) * cross(tf, m2f))
    D2k2DfDthetae = (1.0 / nf) * (0.5 * np.dot(kb, m2e) * tilde_t + (1.0 / chi) * cross(te, m2e))
    D2k2DfDthetaf = (1.0 / nf) * (0.5 * np.dot(kb, m2f) * tilde_t + (1.0 / chi) * cross(te, m2f))

    DDk1 = np.zeros((11, 11))
    DDk2 = np.zeros((11, 11))

    def _assemble(DD, De2, Df2, DeDf, DfDe, Dthetae2, Dthetaf2,
                  DeDthetae, DeDthetaf, DfDthetae, DfDthetaf):
        DD[_A, _A] = De2
        DD[_A, _B] = -De2 + DeDf
        DD[_A, _C] = -DeDf
        DD[_B, _A] = -De2 + DfDe
        DD[_B, _B] = De2 - DeDf - DfDe + Df2
        DD[_B, _C] = DeDf - Df2
        DD[_C, _A] = -DfDe
        DD[_C, _B] = DfDe - Df2
        DD[_C, _C] = Df2
        DD[_TE, _TE] = Dthetae2
        DD[_TF, _TF] = Dthetaf2
        DD[_A, _TE] = -DeDthetae
        DD[_B, _TE] = DeDthetae - DfDthetae
        DD[_C, _TE] = DfDthetae
        DD[_TE, _A] = DD[_A, _TE]
        DD[_TE, _B] = DD[_B, _TE]
        DD[_TE, _C] = DD[_C, _TE]
        DD[_A, _TF] = -DeDthetaf
        DD[_B, _TF] = DeDthetaf - DfDthetaf
        DD[_C, _TF] = DfDthetaf
        DD[_TF, _A] = DD[_A, _TF]
        DD[_TF, _B] = DD[_B, _TF]
        DD[_TF, _C] = DD[_C, _TF]

    _assemble(DDk1, D2k1De2, D2k1Df2, D2k1DeDf, D2k1DfDe, D2k1Dthetae2, D2k1Dthetaf2,
              D2k1DeDthetae, D2k1DeDthetaf, D2k1DfDthetae, D2k1DfDthetaf)
    _assemble(DDk2, D2k2De2, D2k2Df2, D2k2DeDf, D2k2DfDe, D2k2Dthetae2, D2k2Dthetaf2,
              D2k2DeDthetae, D2k2DeDthetaf, D2k2DfDthetae, D2k2DfDthetaf)
    return gradK, DDk1, DDk2


def _twist_grad_hess(te, tf, ne, nf, kb):
    chi = 1.0 + np.dot(te, tf)
    tilde_t = (te + tf) / chi
    gradT = np.zeros(11)
    gradT[_A] = -0.5 / ne * kb
    gradT[_C] = 0.5 / nf * kb
    gradT[_B] = -(gradT[_A] + gradT[_C])
    gradT[_TE] = -1.0
    gradT[_TF] = 1.0

    norm2_e = ne * ne
    norm2_f = nf * nf
    D2mDe2 = -0.25 / norm2_e * (np.outer(kb, te + tilde_t) + np.outer(te + tilde_t, kb))
    D2mDf2 = -0.25 / norm2_f * (np.outer(kb, tf + tilde_t) + np.outer(tf + tilde_t, kb))
    D2mDeDf = 0.5 / (ne * nf) * (2.0 / chi * _cross_mat(te) - np.outer(kb, tilde_t))
    D2mDfDe = D2mDeDf.T

    DDt = np.zeros((11, 11))
    DDt[_A, _A] = D2mDe2
    DDt[_A, _B] = -D2mDe2 + D2mDeDf
    DDt[_B, _A] = -D2mDe2 + D2mDfDe
    DDt[_B, _B] = D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2
    DDt[_A, _C] = -D2mDeDf
    DDt[_C, _A] = -D2mDfDe
    DDt[_C, _B] = D2mDfDe - D2mDf2
    DDt[_B, _C] = D2mDeDf - D2mDf2
    DDt[_C, _C] = D2mDf2
    return gradT, DDt


def _global_indices(topo: Topology, node_prev, e_prev, node_curr, e_next, node_next):
    return [
        3 * node_prev, 3 * node_prev + 1, 3 * node_prev + 2,
        topo.theta_index(e_prev),
        3 * node_curr, 3 * node_curr + 1, 3 * node_curr + 2,
        topo.theta_index(e_next),
        3 * node_next, 3 * node_next + 1, 3 * node_next + 2,
    ]


def bending_twist_grad_hess(P: Params, topo: Topology, x: np.ndarray,
                            m1: np.ndarray, m2: np.ndarray, ref_twist, thetas: np.ndarray):
    """Return (E_b + E_t, grad (n_dof,), Hess COO lists) for bending and twist."""
    EI = 2.0 * P.Y * P.I
    GJ = 2.0 * P.G * P.J
    l_i = P.l0  # undeformed Voronoi length at interior nodes
    n_dof = topo.n_dof
    grad = np.zeros(n_dof)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    E = 0.0

    t = topo.edge_tangents(x)
    lengths = topo.edge_lengths(x)

    for j in range(topo.N):
        interior = topo.interior[j]
        # local index 0 is the free spider junction -> skipped (bending/twist)
        for local_i in range(1, len(interior)):
            node, e_prev, e_next = interior[local_i]
            node_prev = int(topo.edges[e_prev][0])
            node_next = int(topo.edges[e_next][1])
            te = t[e_prev]
            tf = t[e_next]
            ne_ = lengths[e_prev]
            nf_ = lengths[e_next]
            kb = 2.0 * cross(te, tf) / (1.0 + np.dot(te, tf))

            m1e, m2e = m1[e_prev], m2[e_prev]
            m1f, m2f = m1[e_next], m2[e_next]
            kappa1 = 0.5 * np.dot(kb, m2e + m2f)
            kappa2 = -0.5 * np.dot(kb, m1e + m1f)

            gidx = _global_indices(topo, node_prev, e_prev, node, e_next, node_next)

            # ---- Bending, Eq. (eq:Eb) ----
            E += (EI / l_i) * 0.5 * (kappa1 ** 2 + kappa2 ** 2)  # EI = 2 Y I -> Y I |kappa|^2
            gradK, DDk1, DDk2 = _kappa_grad_hess(te, tf, ne_, nf_, m1e, m2e, m1f, m2f,
                                                 kb, kappa1, kappa2)
            gb = (EI / l_i) * (gradK[:, 0] * kappa1 + gradK[:, 1] * kappa2)
            Hb = (EI / l_i) * (
                np.outer(gradK[:, 0], gradK[:, 0]) + np.outer(gradK[:, 1], gradK[:, 1])
                + kappa1 * DDk1 + kappa2 * DDk2
            )

            # ---- Twist, Eq. (eq:Et) ----
            m_twist = thetas[e_next] - thetas[e_prev] + ref_twist[j][local_i]
            E += (GJ / l_i) * 0.5 * m_twist ** 2  # GJ = 2 G J -> G J m^2
            gradT, DDt = _twist_grad_hess(te, tf, ne_, nf_, kb)
            gt = (GJ / l_i) * m_twist * gradT
            Ht = (GJ / l_i) * (m_twist * DDt + np.outer(gradT, gradT))

            gloc = gb + gt
            Hloc = Hb + Ht
            for a in range(11):
                grad[gidx[a]] += gloc[a]
                for b in range(11):
                    if Hloc[a, b] != 0.0:
                        rows.append(gidx[a])
                        cols.append(gidx[b])
                        vals.append(Hloc[a, b])
    return E, grad, (rows, cols, vals)


def elastic_energy_grad_hess(P: Params, topo: Topology, x: np.ndarray,
                             m1: np.ndarray, m2: np.ndarray, ref_twist, thetas: np.ndarray):
    """Total elastic energy, gradient (n_dof,) and sparse Hessian (n_dof x n_dof)."""
    Es, gs, (rs, cs, vs) = stretch_energy_grad_hess(P, topo, x)
    Ebt, gbt, (rb, cb, vb) = bending_twist_grad_hess(P, topo, x, m1, m2, ref_twist, thetas)
    grad = gs + gbt
    rows = rs + rb
    cols = cs + cb
    vals = vs + vb
    H = sp.coo_matrix((vals, (rows, cols)), shape=(topo.n_dof, topo.n_dof)).tocsr()
    return Es + Ebt, grad, H
