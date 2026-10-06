"""Round 8: lock_xy default is bit-identical; extras are opt-in."""

import numpy as np
import pytest

from netsim.config import (
    SimConfig, MaterialConfig, NetConfig, DroneConfig, NumericsConfig,
    ContactConfig, OutputConfig, KinematicConfig,
)
from netsim.simulate import simulate_config, simulate, build_net
from netsim.materials import get_material
from netsim.topology import star
from netsim.discretize import discretize
from netsim.io_hdf5 import select_fail_dense_frames


def _dyn_cfg(lock_xy=False, n_s=6, t_end=0.04):
    return SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.5, 0.0),
                          lock_xy=lock_xy),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=t_end, dt_out=5e-3,
                                use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def test_lock_xy_default_bit_identical_to_explicit_false():
    """Default (lock_xy off) matches an explicit lock_xy=False run."""
    a = simulate_config(_dyn_cfg(), write=False)
    b = simulate_config(_dyn_cfg(lock_xy=False), write=False)
    np.testing.assert_array_equal(a.trajectory.energy, b.trajectory.energy)
    np.testing.assert_array_equal(a.trajectory.drone, b.trajectory.drone)
    np.testing.assert_array_equal(a.trajectory.t, b.trajectory.t)
    assert a.outcome == b.outcome
    assert a.energy_error == b.energy_error


def test_lock_xy_zeros_inplane_drone_velocity():
    res = simulate_config(_dyn_cfg(lock_xy=True), write=False)
    vd = res.trajectory.drone[:, 3:6]
    assert np.all(np.isfinite(vd))
    np.testing.assert_allclose(vd[:, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(vd[:, 1], 0.0, atol=1e-15)
    # Off-centre impact: free run must be allowed to develop vx.
    free = simulate_config(_dyn_cfg(lock_xy=False), write=False)
    assert np.max(np.abs(free.trajectory.drone[:, 3])) > np.max(
        np.abs(res.trajectory.drone[:, 3]))


def test_energy_groups_sum_to_uel():
    cfg = _dyn_cfg()
    cfg.numerics.energy_groups = True
    cfg.numerics.store_mesh = True
    res = simulate_config(cfg, write=False)
    traj = res.trajectory
    disc = discretize(build_net(cfg), cfg.material.resolve(), cfg.numerics.n_s,
                      r_d=cfg.drone.r_d)
    assert traj.uel_r0.shape[0] == traj.t.size
    assert traj.uel_other.shape[0] == traj.t.size
    tot = traj.uel_r0.sum(axis=1) + traj.uel_other
    np.testing.assert_allclose(tot, traj.energy[:, 2], rtol=1e-12, atol=1e-14)


def test_hub_force_keeps_prestretched_thread0_in_equilibrium():
    mat = get_material("S")
    ep = 0.1 * mat.eps_b
    e0 = 0.5 * mat.eps_b
    A = 1e-6
    R = 1.0
    force = (-(mat.sigma(e0) - mat.sigma(ep)) * A, 0.0, 0.0)
    net = star(8, R, ep, material=mat, A_hat=A, eps_p_thread0=e0)
    cfg = SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=R, eps_p=ep, A_hat=A,
                      eps_p_thread0=e0),
        drone=DroneConfig(M=1.0, r_d=1.0, v0=0.0),
        numerics=NumericsConfig(n_s=4, C=0.4, t_end=0.002, dt_out=2e-4,
                                damping=400.0, use_numba=False),
        contact=ContactConfig(mode="frictionless", k_c=1e7),
        output=OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9),
        kinematic=KinematicConfig(
            enabled=True, node=1, mode="velocity",
            direction=(1.0, 0.0, 0.0), amplitude=0.0,
            extra_force_node=0, extra_force=force,
        ),
    )
    res = simulate(net, mat, cfg.drone, cfg.numerics, cfg.contact, cfg.output,
                   kinematic=cfg.kinematic)
    hub = res.trajectory.x[:, 0]
    assert np.max(np.abs(hub[:, 0])) < 1e-5
    assert np.max(np.abs(hub[:, 1])) < 1e-5


def test_fail_dense_frame_selection():
    t = np.linspace(0.0, 0.05, 501)
    idx = select_fail_dense_frames(t, [0.02], dt_fine=1e-4, dt_coarse=5e-4)
    assert 0 in idx and (len(t) - 1) in idx
    # Near the failure the local spacing should be ~0.1 ms.
    near = t[idx]
    near = near[(near > 0.015) & (near < 0.025)]
    if near.size > 2:
        assert np.median(np.diff(near)) < 0.0002


def test_round8_resume_json(tmp_path, monkeypatch):
    import validation.round8 as r8
    monkeypatch.setattr(r8, "OUT", tmp_path)
    r8._save_run("B", "demo", {"run_id": "demo", "ok": 1})
    got = r8._load_run("B", "demo")
    assert got["ok"] == 1
    jobs = [dict(run_id="demo", shard="B")]
    # _pool_jobs with a worker that would fail if called.
    def boom(job):
        raise AssertionError("should have been skipped")
    rows = r8._pool_jobs("B", jobs, boom, "demo")
    assert rows[0]["ok"] == 1
