"""Validation 5b (pytest): Algorithm 2 minimum-mass bisection."""

import os

import numpy as np
import pytest

from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.mmin import (MminConfig, analytical_s0, minimum_mass,
                         monotonicity_scan, evaluate, _criterion_pass,
                         _write_cache, _read_cache)


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
    def __init__(self, mids):
        self.failure_midpoints = np.asarray(mids, dtype=float).reshape(-1, 3)
        self.failures = np.zeros((self.failure_midpoints.shape[0], 3))


class _FakeRes:
    def __init__(self, arrested, n_failures, mids=()):
        self.arrested = arrested
        self.n_failures = n_failures
        self.trajectory = _FakeTraj(mids)


def test_criterion_logic():
    # A: arrested and no failures.
    assert _criterion_pass(_FakeRes(True, 0), (0, 0), "A", 0.5, 10)
    assert not _criterion_pass(_FakeRes(True, 1, [(0, 0, 0)]), (0, 0), "A", 0.5, 10)
    assert not _criterion_pass(_FakeRes(False, 0), (0, 0), "A", 0.5, 10)
    # B: arrested, < k_max failures, all within R_max of p.
    assert _criterion_pass(_FakeRes(True, 2, [(0.1, 0, 0), (0.2, 0, 0)]),
                           (0, 0), "B", 0.5, 10)
    assert not _criterion_pass(_FakeRes(True, 1, [(0.9, 0, 0)]),
                               (0, 0), "B", 0.5, 10)  # outside R_max
    assert not _criterion_pass(_FakeRes(True, 10, [(0, 0, 0)] * 10),
                               (0, 0), "B", 0.5, 10)  # >= k_max


def test_cache_roundtrip(tmp_path):
    pytest.importorskip("h5py")
    info = {"passed": True, "arrested": True, "n_failures": 2,
            "fail_segs": [3, 7], "fail_mids": np.array([[0.1, 0.0, 0.0],
                                                        [0.2, 0.0, 0.0]])}
    path = str(tmp_path / "e.h5")
    _write_cache(path, info)
    back = _read_cache(path)
    assert back["passed"] and back["n_failures"] == 2
    assert list(back["fail_segs"]) == [3, 7]


def test_minimum_mass_centre_passes():
    # The centre impact is easy to arrest; m_min must be positive and the
    # analytical guess must bracket it (s_min <= a few * s0 is plausible).
    cfg = _cfg()
    mm = MminConfig(criterion="A", tol=0.2, impact_points=[(0.0, 0.0)])
    r = minimum_mass(cfg, mm)
    assert r.m_min > 0.0
    assert r.s_min > 0.0
    assert r.worst_point == (0.0, 0.0)


def test_monotonicity_scan_structure():
    cfg = _cfg()
    mm = MminConfig(criterion="B", impact_points=[(0.5, 0.0)])
    sc = monotonicity_scan(cfg, mm, (0.5, 0.0), n=5)
    assert len(sc["pattern"]) == 5
    assert isinstance(sc["monotone"], bool)
    # Largest scanned s must pass (it brackets the minimum).
    assert sc["pattern"][-1] is True
