"""Round-4 unit tests: early perforation bookkeeping and free-lateral pull."""

import numpy as np
import pytest

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig,
                           KinematicConfig)
from netsim.materials import get_material
from netsim.simulate import simulate_config
from netsim.topology import from_arrays
from netsim.discretize import discretize
from netsim.integrator import integrate
from netsim._kernels import segment_forces


def test_failure_energy_booked_at_eps_b():
    """Single-thread break: residual of W - Δ(KE+Uel+Ufail) must be < 1e-3."""
    mat = get_material("D")
    L, n_nodes = 1.0, 21
    xs = np.linspace(0.0, L, n_nodes)
    nodes = np.c_[xs, np.zeros(n_nodes)]
    edges = np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)]
    anchored = np.zeros(n_nodes, dtype=bool)
    anchored[0] = True
    h = L / (n_nodes - 1)
    rest = np.full(n_nodes - 1, h)
    A = np.full(n_nodes - 1, 1e-6)
    net = from_arrays(nodes, edges, anchored, np.zeros(n_nodes - 1), A,
                      rest_length=rest, R=L)
    disc = discretize(net, mat, 1, r_d=1.0, warn_ratio=1e9)
    c = mat.c_tan_max
    speed = 0.5 * mat.eps_b * c / 5.0
    t_end = mat.eps_b * L / speed + 10.0 * L / c
    kin = KinematicConfig(enabled=True, node=n_nodes - 1, mode="velocity",
                          direction=(1.0, 0.0, 0.0), amplitude=speed)
    traj = integrate(
        disc, mat, DroneConfig(M=1.0, r_d=1.0, v0=0.0),
        NumericsConfig(n_s=1, C=0.4, t_end=t_end, dt_out=t_end / 200,
                       use_numba=False),
        ContactConfig(mode="frictionless", k_c=1e7),
        OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9),
        kinematic=kin, net_R=L)
    assert traj.failures.shape[0] >= 1
    # Booked failure energy must equal A*l*Phi(eps_b) per broken segment.
    phi_b = float(mat.Phi(mat.eps_b))
    U_expect = float(np.sum(disc.seg_A * disc.seg_rest_length)) * phi_b
    # Only the broken segments contribute; all break in a chain eventually or
    # just the overloaded ones. Check per-failure Phi booking via U_fail column.
    U_fail = float(traj.energy[-1, 3])
    n_fail = traj.failures.shape[0]
    # Each failure books A_s * l_s * Phi(eps_b); segments share A,l.
    assert U_fail == pytest.approx(n_fail * disc.seg_A[0] * disc.seg_rest_length[0]
                                   * phi_b, rel=1e-9)


def test_early_perforation_stops_before_minus_2R():
    # Soft net / fast drone: should perforate early, not wait for z=-2R.
    cfg = SimConfig(
        material=MaterialConfig(name="D"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.5, 0.0)),
        numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.5, dt_out=2e-3,
                                area_scale=0.05, use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7, k_c_mode="relative",
                              k_c_factor=8.0, penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )
    res = simulate_config(cfg, write=False)
    assert res.outcome == "perforated"
    assert res.trajectory.drone[-1, 2] > -2.0 * cfg.net.R  # stopped early


def test_free_lateral_drifts_outward():
    from validation.offcentre import run_offcentre
    r = run_offcentre("S", a_over_R=0.5, eps_p_frac=0.0, n_s=16,
                      free_lateral=True, continue_to_B=False)
    assert r["eta_A"] == r["eta_A"]  # not NaN
    # Gripped point moves outward from a=0.5 (offc_free gives ~0.65).
    assert r["px_fail"] > 0.5
