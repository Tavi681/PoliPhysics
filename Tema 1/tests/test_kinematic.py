"""Kinematic Simulation turbulence field tests (Stage A), paper Eq. (eq:ks)/(eq:vk).

Covers the four checks required by the objective:
  a) divergence-free field,
  b) single-point statistics (std ~ sigma, mean ~ sigma/10) over many seeds,
  c) reproducibility from a fixed seed,
  d) finite-difference check of grad_u and of the new F_v Jacobian term.
"""

import numpy as np
import pytest

from ballooning.params import Params
from ballooning.geometry import initial_state
from ballooning import forces
from ballooning.fields import (
    KinematicSimulation,
    UniformFlow,
    ZeroFlow,
    make_flow,
    von_karman_E,
)


def _ks(**kw):
    base = dict(sigma=0.25, ell=1.0, U_h=0.0, N_k=100, seed=0, L=0.5, N_t=100)
    base.update(kw)
    return KinematicSimulation(**base)


# --- a) divergence-free ----------------------------------------------------
def test_divergence_free():
    ks = _ks(N_k=200, seed=7)
    rng = np.random.default_rng(1)
    X = rng.uniform(-5.0, 5.0, (5000, 3))
    t = 0.42
    G = ks.grad_u(X, t)                     # (M, 3, 3), analytic
    div = np.einsum("mii->m", G)           # trace = div u
    U = ks.u(X, t)
    max_k = float(np.max(np.abs(ks.k)))
    max_u = float(np.max(np.abs(U)))
    assert np.max(np.abs(div)) < 1e-10 * max_k * max_u


# --- b) statistics ---------------------------------------------------------
@pytest.mark.slow
def test_statistics_full():
    """1e5 points x 200 seeds: std within 3% of sigma, mean within 3% of sigma/10."""
    sigma = 0.25
    n_pts, n_seed = 100_000, 200
    stds = np.zeros((n_seed, 3))
    means = np.zeros((n_seed, 3))
    for s in range(n_seed):
        ks = _ks(sigma=sigma, N_k=100, seed=s)
        rng = np.random.default_rng(10_000 + s)
        X = rng.uniform(-10.0, 10.0, (n_pts, 3))
        U = ks.u(X, 0.0)
        stds[s] = U.std(axis=0)
        means[s] = U.mean(axis=0)
    std_mean = stds.mean(axis=0)
    mean_abs = np.abs(means).mean(axis=0)
    assert np.all(np.abs(std_mean - sigma) < 0.03 * sigma)
    assert np.all(mean_abs < 0.03 * sigma + sigma / 10.0)


def test_statistics_quick():
    """Fast variant of (b): fewer seeds/points, looser tolerance."""
    sigma = 0.25
    stds = []
    for s in range(30):
        ks = _ks(sigma=sigma, N_k=100, seed=s)
        rng = np.random.default_rng(100 + s)
        X = rng.uniform(-10.0, 10.0, (20_000, 3))
        stds.append(ks.u(X, 0.0).std(axis=0))
    std_mean = np.mean(stds, axis=0)
    assert np.all(np.abs(std_mean - sigma) < 0.05 * sigma)


# --- c) reproducibility ----------------------------------------------------
def test_reproducibility():
    ks1 = _ks(seed=123, N_k=150)
    ks2 = _ks(seed=123, N_k=150)
    assert np.array_equal(ks1.k, ks2.k)
    assert np.array_equal(ks1.a, ks2.a)
    assert np.array_equal(ks1.b, ks2.b)
    assert np.array_equal(ks1.omega, ks2.omega)
    X = np.array([[0.1, 0.2, 0.3], [1.0, -2.0, 0.5]])
    assert np.array_equal(ks1.u(X, 0.7), ks2.u(X, 0.7))
    # different seed -> different field
    ks3 = _ks(seed=124, N_k=150)
    assert not np.array_equal(ks1.k, ks3.k)


# --- d) finite-difference checks ------------------------------------------
def test_grad_u_fd():
    ks = _ks(N_k=120, seed=5)
    rng = np.random.default_rng(3)
    t = 0.31
    h = 1e-6
    for _ in range(20):
        x0 = rng.uniform(-3.0, 3.0, 3)
        Ga = ks.grad_u(x0, t)
        Gfd = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            Gfd[:, j] = (ks.u(x0 + e, t) - ks.u(x0 - e, t)) / (2 * h)
        rel = np.linalg.norm(Ga - Gfd) / (np.linalg.norm(Gfd) + 1e-30)
        assert rel < 1e-5


def test_viscous_flow_jacobian_fd():
    """d F_v/d xi from u(x,t): analytic + D_k grad_u vs FD (tangent lagged)."""
    P = Params(N=2, N_t=6, L=0.3, flow_model="kinematic",
               sigma_w=0.3, ell=0.2, turb_seed=3, turb_N_k=40)
    xi, xidot, topo = initial_state(P)
    X = xi[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3).copy()
    X[1:] += 0.01 * np.random.default_rng(0).standard_normal(X[1:].shape)
    xi[: 3 * topo.n_nodes] = X.reshape(-1)
    flow = make_flow(P)
    t = 0.13

    J = forces.viscous_flow_jacobian(P, topo, xi, flow, t).toarray()

    dl = topo.node_voronoi_lengths()
    tang = forces.node_tangents(topo, xi[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3))
    Xn = xi[: 3 * topo.n_nodes].reshape(topo.n_nodes, 3)
    h = 1e-6
    err = 0.0
    nrm = 0.0
    for k in range(topo.n_nodes):
        if k == 0:
            D = P.zeta_s * np.eye(3)
        else:
            tk = tang[k]
            proj = np.outer(tk, tk)
            D = dl[k] * (P.eta_par * proj + P.eta_perp * (np.eye(3) - proj))
        blk = J[3 * k:3 * k + 3, 3 * k:3 * k + 3]
        fd = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            # F_v flow part = -D (v - u); v lagged (0 here), tangent (D) lagged.
            fp = -D @ (0.0 - flow.u(Xn[k] + e, t))
            fm = -D @ (0.0 - flow.u(Xn[k] - e, t))
            fd[:, j] = (fp - fm) / (2 * h)
        err += np.linalg.norm(blk - fd) ** 2
        nrm += np.linalg.norm(fd) ** 2
    assert (err ** 0.5) / (nrm ** 0.5 + 1e-30) < 1e-5


# --- additive-only guarantee ----------------------------------------------
def test_zero_uniform_flow_jacobian_is_zero():
    P = Params(N=2, N_t=5, L=0.3)
    xi, xidot, topo = initial_state(P)
    for flow in (ZeroFlow(), UniformFlow([0.0, 0.0, 0.4])):
        J = forces.viscous_flow_jacobian(P, topo, xi, flow, 0.0)
        assert J.nnz == 0


def test_energy_spectrum_normalisation():
    """int_0^inf E dk = (3/2) sigma^2 (numeric check of the C normalisation)."""
    from scipy import integrate
    sigma, ell = 0.25, 1.0
    val, _ = integrate.quad(lambda k: von_karman_E(k, sigma, ell), 0, np.inf)
    assert abs(val - 1.5 * sigma ** 2) / (1.5 * sigma ** 2) < 1e-6
