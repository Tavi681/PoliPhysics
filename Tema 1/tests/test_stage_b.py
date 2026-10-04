"""Unit / smoke checks for Stage B (alg:sweep) — no full grid."""

import numpy as np
import pytest

from ballooning.fields import KinematicSimulation
from ballooning.params import Params
from ballooning.fields import make_flow
from ballooning import sweep as SB


def test_turb_renormalize_default_off():
    ks = KinematicSimulation(sigma=0.25, ell=1.0, N_k=200, seed=0,
                             L=0.5, N_t=100, renormalize=False)
    assert ks.renorm_factor == 1.0
    assert ks.energy_fraction < 0.98  # mesh-limited deficit, Stage A behaviour


def test_turb_renormalize_sets_energy_to_one():
    ks = KinematicSimulation(sigma=0.25, ell=1.0, N_k=200, seed=0,
                             L=0.5, N_t=100, renormalize=True)
    assert ks.renorm_factor > 1.0
    assert abs(ks.energy_fraction - 1.0) < 1e-12
    # factor is seed-independent
    ks2 = KinematicSimulation(sigma=0.25, ell=1.0, N_k=200, seed=99,
                              L=0.5, N_t=100, renormalize=True)
    assert abs(ks.renorm_factor - ks2.renorm_factor) < 1e-12


def test_params_turb_renormalize_default_false():
    P = Params()
    assert P.turb_renormalize is False


def test_make_flow_passes_renormalize():
    P = Params(flow_model="kinematic", sigma_w=0.2, ell=0.5, turb_N_k=80,
               turb_seed=1, turb_renormalize=True, N_t=50, L=0.5)
    flow = make_flow(P)
    assert P.turb["turb_renormalize"] is True
    assert P.turb["renorm_factor"] > 1.0
    assert hasattr(flow, "u")


def test_sweep_seed_deterministic():
    a = SB.sweep_seed(4, 0.30, 0, 0)
    b = SB.sweep_seed(4, 0.30, 0, 0)
    c = SB.sweep_seed(4, 0.30, 0, 1)
    assert a == b
    assert a != c
    assert 0 <= a < 2**31 - 1


def test_fbar_grid_identity():
    # Fbar_l = 1 + x K/(w_s h) => U0 h / K = x
    N, sigma_w, x = 2, 0.15, -2
    fbar = SB.fbar_from_x(N, sigma_w, x)
    P = Params(N=N, N_t=100, L=0.5, m=SB.SWEEP_M_KG)
    w_s = SB.settling_speed(P)
    U0 = w_s * (fbar - 1.0)
    K = SB.eddy_diffusivity(sigma_w)
    assert abs(U0 * SB.SWEEP_H / K - x) < 1e-9


def test_P_dd_zero_drift_and_wilson():
    assert abs(SB.P_dd(0.0, 1.0) - SB.SWEEP_Z0 / SB.SWEEP_H) < 1e-12
    # upward drift -> P > z0/h
    assert SB.P_dd(1.0, 1.0) > SB.SWEEP_Z0 / SB.SWEEP_H
    # downward drift -> P < z0/h
    assert SB.P_dd(-1.0, 1.0) < SB.SWEEP_Z0 / SB.SWEEP_H
    p, lo, hi = SB.wilson_ci(10, 10)
    assert p == 1.0 and lo > 0.5 and hi >= 1.0 - 1e-12


def test_grid_count_and_order():
    pts = SB.iter_grid_points(M=2)
    assert len(pts) == 4 * 2 * 7 * 2
    # N order: 1,2,4, then 8
    Ns = [p.N for p in pts]
    assert Ns[0] == 1
    assert Ns[-1] == 8
    first_8 = next(i for i, p in enumerate(pts) if p.N == 8)
    assert all(p.N != 8 for p in pts[:first_8])


def test_aggregate_phase_wilson_excludes_timeout():
    rows = [
        {"N": 1, "sigma_w": 0.15, "x": 0, "Fbar_l": 1.0,
         "outcome": "rise", "exit_time": 1.0},
        {"N": 1, "sigma_w": 0.15, "x": 0, "Fbar_l": 1.0,
         "outcome": "fall", "exit_time": 2.0},
        {"N": 1, "sigma_w": 0.15, "x": 0, "Fbar_l": 1.0,
         "outcome": "timeout", "exit_time": 60.0},
    ]
    phase = SB.aggregate_phase(rows)
    assert len(phase) == 1
    assert phase[0]["n_up"] == 1
    assert phase[0]["n_down"] == 1
    assert phase[0]["n_timeout"] == 1
    assert abs(phase[0]["P"] - 0.5) < 1e-12


@pytest.mark.slow
def test_one_short_sweep_job():
    """One clamped+kinematic realization at tiny resolution."""
    pt = SB.SweepPoint(N=1, sigma_w=0.15, x=0, i=0)
    row = SB.run_sweep_job({
        "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
        "N_t": 20, "t_end": 0.15, "output_dt": 0.05,
        "write_hdf5": False, "hdf5_dir": "",
        "snapshot": False, "snapshot_dir": "",
    })
    assert row["outcome"] in ("rise", "fall", "timeout")
    assert row["newton_failures"] >= 0
    assert row["renorm_factor"] > 1.0
