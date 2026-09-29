"""Tests 3-8: physical validation of the time stepper (Algorithm 1).

These run the implicit integrator and are marked ``slow``.
Run only the fast tests with:  pytest -m "not slow"
"""

import numpy as np
import pytest

from ballooning.params import Params
from ballooning.geometry import Topology
from ballooning.integrator import simulate
from ballooning import observables


@pytest.mark.slow
@pytest.mark.parametrize("Q_s", [0.0, 3e-12])
def test_terminal_velocity(Q_s):
    """Test 3: N=1, tip charge, constant E, still air, within 1% of

    U = (N q E + Q_s E - m g) / (N eta_par L + zeta_s).
    Q_s = 0, and the default Q_s = 3 pC.
    """
    from ballooning.studies import run_terminal_velocity

    out = run_terminal_velocity(Q_s)
    print(f"\n[Test 3] Q_s={Q_s:.3e}  V={out['V']:.6e}  U={out['U']:.6e}  "
          f"rel={out['rel_err']:.3e}")
    assert out["status"] == "steady"
    assert out["rel_err"] < 0.01


@pytest.mark.slow
def test_chamber_tip_steady_velocity():
    """Test 4a: chamber field, tip charge, m=0.9 mg, Q_s=0, spider Stokes drag on.

    q = (m g + U_target (eta_par L + zeta_s)) / E_inf with U_target = 0.085 m/s.
    Pass/fail: |V - 0.085| / 0.085 < 2%. t_95 is reported next to 3 t_s,
    t_s = m / (eta_par L + zeta_s); it is not asserted.
    """
    from ballooning.studies import run_chamber_tip

    out = run_chamber_tip()
    print(f"\n[Test 4a] q={out['q']:.5g} C  V={out['V']:.6f} m/s  "
          f"rel={out['rel_err']:.3e}")
    print(f"    t95={out['t95']:.4f} s  3 t_s={out['t95_reference']:.4f} s")
    assert out["status"] == "steady"
    assert out["rel_err"] < 0.02


@pytest.mark.slow
def test_chamber_spider_charge_report():
    """Test 4b: same chamber case with all charge on the spider. Report only."""
    from ballooning.studies import run_chamber_spider

    out = run_chamber_spider()
    print(f"\n[Test 4b] q={out['q']:.5g} C  status={out['status']}  "
          f"V={out['V']:.6f} m/s  t95={out['t95']:.4f} s  "
          f"3 t_s={out['t95_reference']:.4f} s  runtime={out['runtime_s']:.2f} s")
    assert len(out["t"]) == len(out["z0"]) == len(out["vz0"])
    assert np.all(np.isfinite(out["z0"])) and np.all(np.isfinite(out["vz0"]))
    assert out["z0"][-1] > out["z0"][0]


@pytest.mark.slow
def test_uniform_flow_invariance():
    """Test 5: UniformFlow(0.5 z_hat) -> spider-frame shape unchanged, V += 0.5."""
    base = dict(N=1, N_t=30, L=0.5, Q_s=0.0, charge_model="tip",
                field_model="constant", E_constant=8000.0, t_end=4.0,
                dt0=1e-4, dt_max=1e-2, output_dt=2e-2, delta=1e-7, t_w=0.1)
    P0 = Params(flow_model="zero", **base)
    P1 = Params(flow_model="uniform", flow_velocity=(0.0, 0.0, 0.5), **base)
    tr0 = simulate(P0)
    tr1 = simulate(P1)
    V0 = observables.spider_vertical_velocity(tr0.v[-1])
    V1 = observables.spider_vertical_velocity(tr1.v[-1])
    X0 = tr0.x[-1] - tr0.x[-1][0]
    X1 = tr1.x[-1] - tr1.x[-1][0]
    assert abs((V1 - V0) - 0.5) < 1e-6
    assert np.max(np.abs(X0 - X1)) / P0.L < 1e-6


@pytest.mark.slow
def test_first_integral():
    """Test 6: N=2, tip, still air -> T^beta sin(theta) constant within 2%.

    The physical exponent is beta = eta_perp / eta_par (RFT steady-thread
    invariant), which is the default ``Params.beta_first_integral``.
    """
    P = Params(N=2, N_t=40, L=0.5, Q_s=0.0, Q_t=5e-9, charge_model="tip",
               field_model="constant", E_constant=8000.0, flow_model="zero",
               t_end=6.0, dt0=1e-4, dt_max=1e-2, output_dt=5e-2,
               delta=1e-7, t_w=0.15)
    traj = simulate(P)
    topo = Topology(P)
    X = traj.x[-1]
    fi = observables.first_integral(P, topo, X)
    for j in range(topo.N):
        e = topo.thread_edges[j]
        sel = e[2:-2]  # exclude the first and last two edges
        vals = fi[sel]
        spread = (vals.max() - vals.min()) / abs(np.mean(vals))
        assert spread < 0.02


@pytest.mark.slow
def test_twist_stays_zero():
    """Test 7: isotropic threads with free ends keep |theta| < 1e-8 rad."""
    P = Params(N=2, N_t=20, L=0.5, Q_s=0.0, charge_model="tip",
               field_model="constant", E_constant=8000.0, flow_model="zero",
               t_end=0.2, dt0=1e-4, dt_max=2e-3, output_dt=1e-2)
    traj = simulate(P)
    assert np.max(np.abs(traj.theta)) < 1e-8


@pytest.mark.slow
def test_normalized_velocity_two_threads():
    """Habchi & Jawed test: N=2, Fbar_l = 2, E = 7.41e3 V/m. Report vbar_t (target ~2)."""
    from ballooning.studies import run_normalized

    out = run_normalized(50)
    print(f"\n[Paper test 2] N_t=50  Fbar_l={out['Fbar_l']:.4f}  "
          f"vbar_t={out['vbar_t']:.4f} (target ~2)  "
          f"R/L={out['R_over_L']:.4e}  status={out['status']}")
    assert out["status"] == "steady"
    assert abs(out["Fbar_l"] - 2.0) < 1e-12
    assert abs(out["vbar_t"] - 2.0) < 0.25


@pytest.mark.slow
def test_convergence_table():
    """Test 8: N=2, Fbar_l=2. Print vbar_t, R/L and CPU time. No pass/fail."""
    from ballooning.studies import run_normalized

    print("\n[Test 8] N=2, E=7.41e3 V/m, Fbar_l=2:")
    print(f"    {'N_t':>5} {'vbar_t':>10} {'R/L':>12} {'CPU [s]':>10} {'status':>10}")
    for N_t in (25, 50, 100, 200):
        out = run_normalized(N_t)
        print(f"    {N_t:>5} {out['vbar_t']:>10.4f} {out['R_over_L']:>12.6e} "
              f"{out['runtime_s']:>10.2f} {out['status']:>10}")
