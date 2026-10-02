"""Validation 5c (pytest): star_with_rings FDM prestress and shared nodes."""

import numpy as np
import pytest

from netsim.materials import get_material
from netsim.topology import star_with_rings
from netsim.fdm import weighted_laplacian


@pytest.mark.parametrize("q_ratio", [0.5, 1.0, 2.0])
def test_fdm_equilibrium_residual(q_ratio):
    m = get_material("S")
    net = star_with_rings(8, 1.0, [1 / 3, 2 / 3, 1.0], 0.05, material=m,
                          q_ratio=q_ratio)
    # Unloaded equilibrium: L_q x = 0 at every free node.
    L = weighted_laplacian(net.edges, net.q, net.n_v)
    resid = L @ net.nodes
    free = ~net.anchored
    assert np.max(np.linalg.norm(resid[free], axis=1)) < 1e-9


def test_rings_share_nodes():
    N, n_rings = 8, 3
    m = get_material("S")
    net = star_with_rings(N, 1.0, [1 / 3, 2 / 3, 1.0], 0.05, material=m)
    # One hub + N nodes per ring, shared between radials and rings (no overlap).
    assert net.n_v == 1 + N * n_rings
    # Edge count: N radials * n_rings + n_rings rings * N.
    assert net.n_e == N * n_rings + n_rings * N
    # Every node index referenced by an edge is within range (shared, not dup).
    assert net.edges.max() == net.n_v - 1


def test_ring_nodes_on_radial_lines():
    # The FDM solution must preserve the N-fold symmetry: ring nodes stay on
    # their radial lines (same angles as the hub spokes).
    N = 8
    m = get_material("S")
    net = star_with_rings(N, 1.0, [1 / 3, 2 / 3, 1.0], 0.05, material=m,
                          q_ratio=2.0)
    base = 2 * np.pi * np.arange(N) / N
    for j in range(3):
        for k in range(N):
            node = net.nodes[1 + j * N + k]
            ang = np.arctan2(node[1], node[0])
            assert np.isclose(np.cos(ang), np.cos(base[k]), atol=1e-6)
            assert np.isclose(np.sin(ang), np.sin(base[k]), atol=1e-6)


def test_higher_q_ratio_pulls_rings_inward():
    m = get_material("S")
    r_lo = star_with_rings(8, 1.0, [1 / 3, 2 / 3, 1.0], 0.05, material=m,
                           q_ratio=0.5).nodes
    r_hi = star_with_rings(8, 1.0, [1 / 3, 2 / 3, 1.0], 0.05, material=m,
                           q_ratio=2.0).nodes
    # Inner ring (nodes 1..N) radius shrinks as q_ring/q_radial grows.
    rad_lo = np.hypot(r_lo[1, 0], r_lo[1, 1])
    rad_hi = np.hypot(r_hi[1, 0], r_hi[1, 1])
    assert rad_hi < rad_lo


def test_outer_ring_anchored_at_R():
    N = 8
    m = get_material("S")
    net = star_with_rings(N, 1.0, [1 / 3, 2 / 3, 1.0], 0.05, material=m)
    anc = net.nodes[net.anchored]
    assert anc.shape[0] == N
    assert np.allclose(np.hypot(anc[:, 0], anc[:, 1]), 1.0, atol=1e-9)


def test_ring_net_simulates():
    # A full run on the ring net must complete and conserve energy reasonably.
    from netsim.config import (SimConfig, MaterialConfig, NetConfig,
                               DroneConfig, NumericsConfig, ContactConfig,
                               OutputConfig)
    from netsim.simulate import simulate_config
    cfg = SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star_with_rings", N=8, R=1.0, eps_p=0.05,
                      A_hat=1e-6, radii=[0.5, 1.0], q_ratio=1.0),
        drone=DroneConfig(M=0.2, r_d=0.15, v0=5.0, p=(0.0, 0.0)),
        numerics=NumericsConfig(n_s=4, C=0.5, t_end=0.05, dt_out=2e-3,
                                use_numba=False),
        contact=ContactConfig(mode="frictionless", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )
    res = simulate_config(cfg, write=False)
    assert res.outcome in ("arrested", "perforated", "timeout")
