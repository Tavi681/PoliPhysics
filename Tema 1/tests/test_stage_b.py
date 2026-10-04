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


def test_modes_independent_of_Nt():
    """After the field fix, modes depend only on (seed, sigma, ell, U_h, N_k, λ)."""
    kwargs = dict(sigma=0.30, ell=1.0, U_h=1.0, N_k=200, seed=42,
                  lambda_=0.5, renormalize=True, L=0.5)
    a = KinematicSimulation(N_t=50, **kwargs)
    b = KinematicSimulation(N_t=100, **kwargs)
    assert a.modes_equal(b)
    assert abs(a.k_max - 2.0 * np.pi / (5.0 * 0.005)) < 1e-12
    # Pre-fix mesh-limited fields at the T2 point differed.
    old50 = KinematicSimulation(N_t=50, mesh_limited=True, **kwargs)
    old100 = KinematicSimulation(N_t=100, mesh_limited=True, **kwargs)
    assert not old50.modes_equal(old100)
    # T1 band: N_t=100 mesh-limited == fixed k_max
    t1 = dict(sigma=0.25, ell=1.0, U_h=0.0, N_k=200, seed=0,
              renormalize=True, L=0.5)
    assert KinematicSimulation(N_t=100, mesh_limited=False, **t1).modes_equal(
        KinematicSimulation(N_t=100, mesh_limited=True, **t1)
    )


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


def test_production_grid_counts_and_order():
    pts = SB.iter_production_points()
    assert len(pts) == 8600
    assert sum(1 for p in pts if p.split == "main") == 8000
    assert sum(1 for p in pts if p.split == "gen_N") == 300
    assert sum(1 for p in pts if p.split == "gen_sigma") == 300
    assert pts[0].split in ("gen_N", "gen_sigma")
    main = [p for p in pts if p.split == "main"]
    first_8 = next(i for i, p in enumerate(main) if p.N == 8)
    assert all(p.N != 8 for p in main[:first_8])
    # M by x on the main grid
    n_core = sum(1 for p in main if p.x in SB.SWEEP_X_CORE)
    n_wing = sum(1 for p in main if p.x not in SB.SWEEP_X_CORE)
    assert n_core == 4 * 2 * 3 * 200
    assert n_wing == 4 * 2 * 4 * 100


def test_shard_assignment_partitions():
    pts = SB.iter_production_points()
    assign = SB.shard_assignment(pts, 3)
    assert len(assign) == len(pts)
    assert set(assign) == {0, 1, 2}
    loads = [0.0, 0.0, 0.0]
    for pt, s in zip(pts, assign):
        loads[s] += SB.cost_weight(pt.N)
    # LPT should keep the three shards within 20% of each other
    assert max(loads) / min(loads) < 1.2


def test_write_ml_hdf5_layout(tmp_path):
    from ballooning.geometry import Topology
    from ballooning.integrator import Trajectory
    from ballooning.io_hdf5 import write_ml_hdf5
    import h5py

    P = Params(N=1, N_t=4, L=0.5, flow_model="kinematic",
               sigma_w=0.15, ell=1.0, turb_N_k=8, turb_seed=1,
               turb_renormalize=True)
    from ballooning.fields import make_flow
    make_flow(P)
    topo = Topology(P)
    T, n, ne = 3, topo.n_nodes, topo.n_edges
    x = np.zeros((T, n, 3))
    x[:, 0, 2] = [0.5, 0.51, 0.52]
    traj = Trajectory(
        t=np.array([0.0, 0.01, 0.02]),
        x=x, v=np.zeros((T, n, 3)),
        theta=np.zeros((T, ne)),
        edges=topo.edges.copy(),
        thread_id=topo.thread_id.copy(),
        q_node=np.zeros(n),
        outcome={"status": "timeout", "exit_time": 0.02, "entangled": False,
                 "diag": {"newton_failures": 0, "n_steps": 2}},
    )
    path = tmp_path / "ml.h5"
    write_ml_hdf5(str(path), P, traj, float32=True)
    with h5py.File(path, "r") as f:
        assert "theta" not in f
        assert f["t"].dtype == np.float32
        assert f["x"].shape == (T, n, 3)
        assert f["twist"].shape == (T, ne)
        assert f["u_air"].shape == (T, n, 3)
        assert f["charge"].shape == (n,)
        assert "thread_id" in f["topology"]
        assert "node_type" in f["topology"]
        assert "edges" in f["topology"]
        assert "renorm_factor" in f["turb"].attrs or "renorm_factor" in f["turb"]
        assert f["outcome"].attrs["status"] == "timeout"


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
