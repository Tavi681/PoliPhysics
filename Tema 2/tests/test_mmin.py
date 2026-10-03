"""Validation 5b (pytest): Algorithm 2 minimum-mass bisection."""

import os

import numpy as np
import pytest

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.mmin import (MminConfig, analytical_s0, minimum_mass,
                         monotonicity_scan, evaluate, _criterion_pass,
                         _write_cache, _read_cache, pass_from_info)


def _cfg(n_s=6, t_end=0.2):
    return SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.0, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=0.5, t_end=t_end, dt_out=5e-3,
                                use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def test_analytical_s0_formula():
    cfg = _cfg()
    from netsim.simulate import build_net
    mat = cfg.material.resolve()
    net = build_net(cfg)
    expect = (0.5 * cfg.drone.M * cfg.drone.v0 ** 2 / mat.e_mat) \
        / net.net_mass(mat.rho, 1.0)
    assert analytical_s0(cfg) == pytest.approx(expect, rel=1e-12)


class _FakeTraj:
    def __init__(self, mids, drone_xy=None):
        self.failure_midpoints = np.asarray(mids, dtype=float).reshape(-1, 3)
        n = self.failure_midpoints.shape[0]
        self.failures = np.zeros((n, 3))
        if drone_xy is None:
            self.failure_drone_xy = np.zeros((n, 2))
        else:
            self.failure_drone_xy = np.asarray(drone_xy, dtype=float).reshape(-1, 2)


class _FakeRes:
    def __init__(self, arrested, n_failures, mids=(), R_d=None,
                 outcome="arrested", drone_xy=None):
        self.arrested = arrested
        self.n_failures = n_failures
        self.trajectory = _FakeTraj(mids, drone_xy)
        self.outcome = outcome
        if R_d is not None:
            self.R_d = R_d
        elif len(mids):
            self.R_d = float(max(np.hypot(m[0], m[1]) for m in mids))
        else:
            self.R_d = 0.0
        self.energy_error = 0.0


def test_criterion_logic():
    # A: arrested and no failures.
    assert _criterion_pass(_FakeRes(True, 0), (0, 0), "A", 0.5, 10)
    assert not _criterion_pass(_FakeRes(True, 1, [(0, 0, 0)]), (0, 0), "A", 0.5, 10)
    assert not _criterion_pass(_FakeRes(False, 0, outcome="timeout"), (0, 0),
                               "A", 0.5, 10)
    # B_any: arrested, not perforated, n_failed < k_max (no R_max cut).
    assert _criterion_pass(_FakeRes(True, 0), (0, 0), "B_any", 0.5, 10)
    assert _criterion_pass(
        _FakeRes(True, 2, [(0.9, 0, 0), (0.8, 0, 0)], R_d=0.9),
        (0, 0), "B_any", 0.5, 10)
    assert not _criterion_pass(
        _FakeRes(True, 2, [(0.1, 0, 0)], outcome="perforated"),
        (0, 0), "B_any", 0.5, 10)
    assert not _criterion_pass(
        _FakeRes(True, 10, [(0, 0, 0)] * 10), (0, 0), "B_any", 0.5, 10)
    # B_loc: failures within R_max of drone at break (not of p).
    # Failures far from p=0 but near drifting drone at x=0.6 → pass at R_max=0.5.
    assert _criterion_pass(
        _FakeRes(True, 1, mids=[(0.7, 0, 0)], drone_xy=[(0.6, 0)]),
        (0, 0), "B_loc", 0.5, 10)
    # Far from drone → fail.
    assert not _criterion_pass(
        _FakeRes(True, 1, mids=[(0.9, 0, 0)], drone_xy=[(0.0, 0)]),
        (0, 0), "B_loc", 0.5, 10)
    # Legacy B == B_loc.
    assert _criterion_pass(
        _FakeRes(True, 1, mids=[(0.7, 0, 0)], drone_xy=[(0.6, 0)]),
        (0, 0), "B", 0.5, 10)


def test_a_implies_b_any():
    info = {"arrested": True, "n_failures": 0, "outcome": "arrested",
            "fail_mids": np.zeros((0, 3)), "fail_drone_xy": np.zeros((0, 2)),
            "R_d": 0.0}
    ok_a, _ = pass_from_info(info, "A", 0.5, 10)
    ok_b, _ = pass_from_info(info, "B_any", 0.5, 10)
    assert ok_a and ok_b


def test_cache_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    info = {"passed": True, "arrested": True, "n_failures": 2,
            "fail_segs": [3, 7],
            "fail_mids": np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            "fail_drone_xy": np.array([[0.05, 0.0], [0.15, 0.0]]),
            "R_d": 0.2, "outcome": "arrested", "energy_error": 1e-6}
    path = str(tmp_path / "e.h5")
    _write_cache(path, info)
    back = _read_cache(path)
    assert back["arrested"] is True
    assert back["n_failures"] == 2
    assert back["fail_drone_xy"].shape == (2, 2)


@pytest.mark.slow
def test_minimum_mass_smoke():
    cfg = _cfg(n_s=6, t_end=0.25)
    mm = MminConfig(criterion="A", tol=0.15, impact_points=[(0.0, 0.0)],
                    n_procs=1)
    r = minimum_mass(cfg, mm)
    assert r.m_min > 0
    assert r.s_min > 0
