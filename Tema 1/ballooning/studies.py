"""Named validation runs shared by the pytest suite and scripts/produce_results.py.

Every study is deterministic: fixed parameters, no random draws.
"""

from __future__ import annotations

import time

import numpy as np

from .params import Params
from .geometry import Topology
from .integrator import simulate
from . import observables
from .fields import ChamberField


def analytic_terminal_velocity(P: Params, E: float) -> float:
    """U = (N q E + Q_s E - m g) / (N eta_par L + zeta_s)."""
    lift = (P.N * P.Q_t + P.Q_s) * E
    drag = P.N * P.eta_par * P.L + P.zeta_s
    return (lift - P.m * P.g) / drag


def normalized_terminal_velocity(U: float, P: Params, E: float) -> float:
    """vbar_t = U * mu * N * L / (N q E - m g)."""
    return U * P.mu * P.N * P.L / (P.N * P.Q_t * E - P.m * P.g)


def charge_for_lift_ratio(P: Params, E: float, fbar: float) -> float:
    """Per-thread charge such that Fbar_l = N q E / (m g) = fbar."""
    return fbar * P.m * P.g / (P.N * E)


def time_to_fraction(t: np.ndarray, v: np.ndarray, frac: float = 0.95) -> float:
    """First time at which v reaches ``frac`` of its final value."""
    goal = frac * v[-1]
    if v[-1] >= 0.0:
        hits = np.where(v >= goal)[0]
    else:
        hits = np.where(v <= goal)[0]
    if len(hits) == 0:
        return float(t[-1])
    return float(t[hits[0]])


def settling_time(t: np.ndarray, v: np.ndarray, tol: float = 0.05) -> float:
    """Earliest time after which |v - v_final| <= tol |v_final| for good."""
    outside = np.where(np.abs(v - v[-1]) > tol * abs(v[-1]))[0]
    if len(outside) == 0:
        return float(t[0])
    return float(t[min(outside[-1] + 1, len(t) - 1)])


def _steady(**overrides) -> dict:
    base = dict(
        flow_model="zero",
        dt0=1e-4,
        dt_max=1e-2,
        output_dt=1e-3,
        delta=1e-6,
        t_w=0.1,
        t_end=6.0,
    )
    base.update(overrides)
    return base


def terminal_velocity_params(Q_s: float, N_t: int = 100) -> Params:
    """Test 3: N=1, tip charge, constant E, still air."""
    return Params(
        N=1, N_t=N_t, L=0.5, Q_s=Q_s, Q_t=1.28e-9, charge_model="tip",
        field_model="constant", E_constant=8000.0, **_steady(),
    )


CHAMBER_U_TARGET = 0.085


def chamber_charge(P: Params, U_target: float = CHAMBER_U_TARGET) -> float:
    """q = (m g + U_target (eta_par L + zeta_s)) / E_inf  (single thread, Q_s = 0)."""
    Einf = ChamberField().Einf
    return (P.m * P.g + U_target * (P.eta_par * P.L + P.zeta_s)) / Einf


def stokes_time(P: Params) -> float:
    """t_s = m / (N eta_par L + zeta_s)."""
    return P.m / (P.N * P.eta_par * P.L + P.zeta_s)


def chamber_tip_params(N_t: int = 100) -> Params:
    """Test 4a: tip charge in the chamber field, q from ``chamber_charge``."""
    P = Params(
        N=1, N_t=N_t, L=0.5, m=0.9e-6, Q_s=0.0,
        charge_model="tip", field_model="chamber", z0=0.0,
        **_steady(dt_max=1e-3),
    )
    P.Q_t = chamber_charge(P)
    return P


def chamber_spider_params(N_t: int = 100) -> Params:
    """Test 4b: the same charge placed on the spider (Morley & Gorham 1-D analogue)."""
    P = Params(
        N=1, N_t=N_t, L=0.5, m=0.9e-6, Q_t=0.0,
        charge_model="tip", field_model="chamber", z0=0.0,
        **_steady(dt0=1e-5, dt_max=1e-3, delta=1e-5, t_w=0.05, t_end=2.0),
    )
    P.Q_s = chamber_charge(P)
    return P


def paper_test2_params(N_t: int) -> Params:
    """Habchi & Jawed multi-thread case: N=2, Fbar_l = 2, E = 7.41 kV/m."""
    E = 7.41e3
    P = Params(
        N=2, N_t=N_t, L=0.5, m=1e-6, Q_s=3e-12, charge_model="tip",
        field_model="constant", E_constant=E, **_steady(),
    )
    P.Q_t = charge_for_lift_ratio(P, E, 2.0)
    return P


def spider_velocity_series(traj) -> np.ndarray:
    return np.array([observables.spider_vertical_velocity(v) for v in traj.v])


def run_terminal_velocity(Q_s: float) -> dict:
    P = terminal_velocity_params(Q_s)
    t0 = time.perf_counter()
    traj = simulate(P)
    runtime = time.perf_counter() - t0
    V = observables.spider_vertical_velocity(traj.v[-1])
    U = analytic_terminal_velocity(P, P.E_constant)
    return {
        "Q_s": Q_s,
        "V": V,
        "U": U,
        "rel_err": abs(V - U) / abs(U),
        "status": traj.outcome["status"],
        "t_exit": float(traj.t[-1]),
        "runtime_s": runtime,
    }


def _run_chamber(P: Params) -> dict:
    t0 = time.perf_counter()
    traj = simulate(P)
    runtime = time.perf_counter() - t0
    vz0 = spider_velocity_series(traj)
    V = float(vz0[-1])
    t_s = stokes_time(P)
    return {
        "q": P.Q_t + P.Q_s,
        "V": V,
        "U_target": CHAMBER_U_TARGET,
        "rel_err": abs(V - CHAMBER_U_TARGET) / CHAMBER_U_TARGET,
        "t95": time_to_fraction(traj.t, vz0, 0.95),
        "t_s": t_s,
        "t95_reference": 3.0 * t_s,
        "v_peak": float(np.max(vz0)),
        "t_settle_5pct": settling_time(traj.t, vz0, 0.05),
        "t": traj.t.copy(),
        "z0": traj.x[:, 0, 2].copy(),
        "vz0": vz0,
        "status": traj.outcome["status"],
        "t_exit": float(traj.t[-1]),
        "runtime_s": runtime,
    }


def run_chamber_tip(N_t: int = 100) -> dict:
    return _run_chamber(chamber_tip_params(N_t))


def run_chamber_spider(N_t: int = 100) -> dict:
    return _run_chamber(chamber_spider_params(N_t))


def run_normalized(N_t: int) -> dict:
    P = paper_test2_params(N_t)
    E = P.E_constant
    t0 = time.perf_counter()
    traj = simulate(P)
    runtime = time.perf_counter() - t0
    topo = Topology(P)
    V = observables.spider_vertical_velocity(traj.v[-1])
    R = observables.tip_radius(topo, traj.x[-1])
    return {
        "N_t": N_t,
        "Q_t": P.Q_t,
        "Fbar_l": P.N * P.Q_t * E / (P.m * P.g),
        "V": V,
        "vbar_t": normalized_terminal_velocity(V, P, E),
        "R_over_L": R / P.L,
        "status": traj.outcome["status"],
        "t_exit": float(traj.t[-1]),
        "runtime_s": runtime,
    }


# ---------------------------------------------------------------------------
# Production studies (scripts/produce_results.py)
# ---------------------------------------------------------------------------
PRODUCTION_E = 7.41e3


def production_params(N: int, fbar: float, L: float = 0.5, N_t: int = 100,
                      w: float = 0.0, output_dt: float = 1e-2) -> Params:
    """Tip charge, constant E = 7.41 kV/m, m = 1 mg, Q_s = 0, q from Fbar_l.

    ``w`` is the vertical velocity of a uniform flow (0 = still air).
    """
    flow = dict(flow_model="zero") if w == 0.0 else dict(
        flow_model="uniform", flow_velocity=(0.0, 0.0, w))
    base = _steady(output_dt=output_dt, t_end=8.0)
    base.update(flow)
    P = Params(
        N=N, N_t=N_t, L=L, m=1e-6, Q_s=0.0, charge_model="tip",
        field_model="constant", E_constant=PRODUCTION_E, **base,
    )
    P.Q_t = charge_for_lift_ratio(P, PRODUCTION_E, fbar)
    return P


def straight_thread_vbar(P: Params) -> float:
    """vbar_t of N straight vertical threads: mu N L / (N eta_par L + zeta_s)."""
    return P.mu * P.N * P.L / (P.N * P.eta_par * P.L + P.zeta_s)


def run_production(P: Params) -> dict:
    """Run to steady state and return scalar observables plus the final state."""
    t0 = time.perf_counter()
    traj = simulate(P)
    runtime = time.perf_counter() - t0
    topo = Topology(P)
    X = traj.x[-1]
    V = observables.spider_vertical_velocity(traj.v[-1])
    E = P.E_constant
    lift = P.N * P.Q_t * E
    return {
        "N": P.N,
        "N_t": P.N_t,
        "L": P.L,
        "Q_t": P.Q_t,
        "Fbar_l": lift / (P.m * P.g),
        "V": V,
        "U_analytic": analytic_terminal_velocity(P, E),
        "vbar_t": normalized_terminal_velocity(V, P, E),
        "vbar_t_straight": straight_thread_vbar(P),
        "R_over_L": observables.tip_radius(topo, X) / P.L,
        "theta_L": observables.tip_angle(topo, X),
        "d_min": observables.min_interthread_distance(topo, X),
        "t": traj.t.copy(),
        "z0": traj.x[:, 0, 2].copy(),
        "x_final": X.copy(),
        "status": traj.outcome["status"],
        "t_exit": float(traj.t[-1]),
        "runtime_s": runtime,
    }
