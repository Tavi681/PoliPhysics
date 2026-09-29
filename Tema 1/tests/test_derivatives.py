"""Test 1 & 2 (partial): finite-difference verification of analytic derivatives
and zero elastic force for a straight, untwisted, unstretched thread."""

import numpy as np
import pytest

from ballooning.params import Params
from ballooning.geometry import initial_state
from ballooning.frames import FrameState
from ballooning import elastic, forces
from ballooning.fields import make_field, ZeroFlow


def _random_twisted_state(P, seed=0, scale=0.02):
    rng = np.random.default_rng(seed)
    xi, xidot, topo = initial_state(P)
    X = topo.positions(xi).copy()
    X[1:] += scale * rng.standard_normal(X[1:].shape)
    xi[: 3 * topo.n_nodes] = X.reshape(-1)
    th = topo.thetas(xi)
    th[:] = 0.1 * rng.standard_normal(len(th))
    return xi, topo


def _elastic_energy_grad_hess(P, topo, fs, xi):
    X = topo.positions(xi)
    th = topo.thetas(xi)
    _d1, _d2, m1, m2, rt = fs.evaluate(X, th)
    return elastic.elastic_energy_grad_hess(P, topo, X, m1, m2, rt, th)


@pytest.mark.parametrize("N,N_t", [(1, 8), (2, 6), (3, 5)])
def test_elastic_gradient_and_hessian_fd(N, N_t):
    """Analytic elastic gradient/Hessian match central finite differences."""
    P = Params(N=N, N_t=N_t, L=0.3)
    xi, topo = _random_twisted_state(P, seed=N * 10 + N_t)
    fs = FrameState(topo, topo.positions(initial_state(P)[0]))

    E0, g, H = _elastic_energy_grad_hess(P, topo, fs, xi)
    H = H.toarray()

    eps = 1e-7
    gfd = np.zeros_like(g)
    Hfd = np.zeros_like(H)
    for i in range(len(xi)):
        d = np.zeros_like(xi)
        d[i] = eps
        Ep, gp, _ = _elastic_energy_grad_hess(P, topo, fs, xi + d)
        Em, gm, _ = _elastic_energy_grad_hess(P, topo, fs, xi - d)
        gfd[i] = (Ep - Em) / (2 * eps)
        Hfd[:, i] = (gp - gm) / (2 * eps)

    assert np.linalg.norm(g - gfd) / (np.linalg.norm(gfd) + 1e-30) < 1e-5
    assert np.linalg.norm(H - Hfd) / (np.linalg.norm(Hfd) + 1e-30) < 1e-5


@pytest.mark.parametrize("charge_model", ["tip", "uniform"])
def test_external_force_jacobian_fd(charge_model):
    """Analytic d(F_l + F_r)/dxi matches finite differences (v = 0 -> F_v = 0)."""
    P = Params(N=3, N_t=5, L=0.3, charge_model=charge_model, field_model="chamber")
    xi, topo = _random_twisted_state(P, seed=7, scale=0.01)
    q = forces.node_charges(P, topo)
    field = make_field(P)
    flow = ZeroFlow()
    v = np.zeros_like(xi)

    J = forces.external_position_jacobian(P, topo, xi, field, q).toarray()

    def f(vec):
        return forces.assemble_fext(P, topo, vec, v, field, flow, 0.0, q)

    eps = 1e-7
    n = 3 * topo.n_nodes
    Jfd = np.zeros((topo.n_dof, topo.n_dof))
    for i in range(n):
        d = np.zeros_like(xi)
        d[i] = eps
        Jfd[:, i] = (f(xi + d) - f(xi - d)) / (2 * eps)

    err = np.linalg.norm(J[:n, :n] - Jfd[:n, :n]) / (np.linalg.norm(Jfd[:n, :n]) + 1e-30)
    assert err < 1e-5


def test_zero_elastic_force_straight_thread():
    """A straight, untwisted, unstretched thread has ~zero elastic gradient."""
    P = Params(N=1, N_t=20, L=0.5)
    xi, xidot, topo = initial_state(P)
    fs = FrameState(topo, topo.positions(xi))
    E, g, H = _elastic_energy_grad_hess(P, topo, fs, xi)
    assert E < 1e-18
    assert np.max(np.abs(g)) < 1e-9
