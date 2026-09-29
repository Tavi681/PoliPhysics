"""Test 2 (frames): parallel transport of a straight thread leaves the reference
frames unchanged."""

import numpy as np

from ballooning.params import Params
from ballooning.geometry import initial_state
from ballooning import frames


def test_parallel_transport_straight_thread_unchanged():
    P = Params(N=2, N_t=15, L=0.5)
    xi, xidot, topo = initial_state(P)
    X = topo.positions(xi)

    d1, d2 = frames.init_reference_directors(topo, X)
    t_conv = topo.edge_tangents(X)

    # Time-parallel transport onto the *same* tangents must be the identity.
    d1_new, d2_new = frames.time_parallel_transport(topo, d1, t_conv, X)
    assert np.max(np.abs(d1_new - d1)) < 1e-12
    assert np.max(np.abs(d2_new - d2)) < 1e-12

    # Reference twist of a straight untwisted thread is zero.
    rt_prev = frames.zero_reference_twist(topo)
    rt = frames.compute_reference_twist(topo, d1, X, rt_prev)
    for j in range(topo.N):
        assert np.max(np.abs(rt[j])) < 1e-12


def test_reference_directors_are_adapted():
    """Reference directors are orthonormal and perpendicular to the tangent."""
    P = Params(N=1, N_t=10, L=0.5)
    xi, xidot, topo = initial_state(P)
    X = topo.positions(xi)
    d1, d2 = frames.init_reference_directors(topo, X)
    t = topo.edge_tangents(X)
    for e in range(topo.n_edges):
        assert abs(np.dot(d1[e], t[e])) < 1e-12
        assert abs(np.dot(d2[e], t[e])) < 1e-12
        assert abs(np.dot(d1[e], d2[e])) < 1e-12
        assert abs(np.linalg.norm(d1[e]) - 1.0) < 1e-12
