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
from .fields import ChamberField, KinematicSimulation, von_karman_E


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


INVARIANCE_EQUAL_T_OUTPUT = 0.1
INVARIANCE_EQUAL_T_END = 2.0
INVARIANCE_EQUAL_T_W = 0.5
# Candidate fixed steps (tried in order). 1e-4 and 5e-5 fail Newton at eps=1e-10;
# 3e-5 is the largest that converges with default K=20.
INVARIANCE_EQUAL_T_DT_CANDIDATES = (1e-4, 5e-5, 3e-5, 2e-5, 1e-5)


def invariance_equal_t_params(w: float, dt: float, N: int = 4, fbar: float = 2.0,
                              N_t: int = 100,
                              t_end: float = INVARIANCE_EQUAL_T_END,
                              output_dt: float = INVARIANCE_EQUAL_T_OUTPUT) -> Params:
    """Fixed-dt invariance params: default eps/K, no adaptivity, integrate to t_end."""
    P = production_params(N=N, fbar=fbar, N_t=N_t, w=w, output_dt=output_dt)
    P.dt0 = dt
    P.dt_max = dt
    P.adaptive_dt = False
    P.stop_on_steady = False
    P.t_end = t_end
    # Keep production Newton settings (eps=1e-10, K=20).
    return P


def _equal_t_compare_grid(output_dt: float, t_end: float) -> np.ndarray:
    """Sample times output_dt, 2*output_dt, ..., <= t_end."""
    if t_end + 1e-15 < output_dt:
        raise ValueError(f"t_end={t_end} < output_dt={output_dt}")
    n = int(np.floor(t_end / output_dt + 1e-12))
    return output_dt * np.arange(1, n + 1)


def _uniform_vertical_velocity(P: Params, w: float) -> np.ndarray:
    """xi_dot with every node velocity = w * z_hat (theta dots = 0)."""
    xd = np.zeros(P.n_dof)
    for i in range(P.n_nodes):
        xd[3 * i + 2] = w
    return xd


def _select_fixed_dt() -> float:
    """Largest fixed dt among candidates that takes one Newton step at eps=1e-10."""
    from .geometry import initial_state

    for dt in INVARIANCE_EQUAL_T_DT_CANDIDATES:
        P = invariance_equal_t_params(0.0, dt, t_end=dt, output_dt=dt)
        xi0, xd0, _ = initial_state(P)
        try:
            simulate(P, xi0=xi0, xi_dot0=xd0)
            print(f"equal-t fixed dt selected: {dt:.3e}")
            return dt
        except RuntimeError as exc:
            print(f"equal-t candidate dt={dt:.3e} rejected: {exc}")
    raise RuntimeError(
        "No fixed dt in INVARIANCE_EQUAL_T_DT_CANDIDATES converges at eps=1e-10, K=20"
    )


def _spider_frame(X: np.ndarray) -> np.ndarray:
    return X - X[0]


def max_shape_dev_equal_t(traj0, traj1, L: float,
                          t_compare: np.ndarray | None = None,
                          t_tol: float | None = None,
                          output_dt: float = INVARIANCE_EQUAL_T_OUTPUT,
                          t_end: float = INVARIANCE_EQUAL_T_END) -> dict:
    """Max spider-frame node deviation / L at matching output times."""
    if t_compare is None:
        t_compare = _equal_t_compare_grid(output_dt, t_end)
    # Fixed-dt runs accumulate float error (e.g. 0.10002 vs 0.1); allow half an
    # output interval unless the caller overrides.
    if t_tol is None:
        t_tol = 0.5 * output_dt
    per_t = []
    t_used = []
    for tc in t_compare:
        i0 = int(np.argmin(np.abs(traj0.t - tc)))
        i1 = int(np.argmin(np.abs(traj1.t - tc)))
        if abs(traj0.t[i0] - tc) > t_tol or abs(traj1.t[i1] - tc) > t_tol:
            raise RuntimeError(
                f"Missing matched output at t={tc}: got {traj0.t[i0]} and {traj1.t[i1]}"
            )
        d = _spider_frame(traj0.x[i0]) - _spider_frame(traj1.x[i1])
        per_t.append(float(np.max(np.abs(d)) / L))
        t_used.append(0.5 * (float(traj0.t[i0]) + float(traj1.t[i1])))
    return {
        "t_compare": np.asarray(t_compare, dtype=float).copy(),
        "t_used": np.array(t_used),
        "per_t": np.array(per_t),
        "max_shape_dev_equal_t": float(np.max(per_t)),
    }


def _decay_time(t_compare: np.ndarray, per_t: np.ndarray) -> float:
    """First compared time at which per_t <= max(per_t)/e (1/e decay)."""
    peak = float(np.max(per_t))
    if peak <= 0.0:
        return float(t_compare[0])
    thr = peak / np.e
    for t, v in zip(t_compare, per_t):
        if v <= thr:
            return float(t)
    return float(t_compare[-1])


def _simulate_invariance_job(spec: tuple) -> dict:
    """Pickable worker: (label, w, v0_mode, dt, t_end, output_dt) -> trajectory.

    v0_mode: 'zero' | 'comoving'  (comoving => all nodes start at w*z_hat).
    """
    from .geometry import initial_state

    label, w, v0_mode, dt, t_end, output_dt = spec
    P = invariance_equal_t_params(w, dt, t_end=t_end, output_dt=output_dt)
    xi0, xd0, _ = initial_state(P)
    if v0_mode == "comoving" and w != 0.0:
        xd0 = _uniform_vertical_velocity(P, w)
        # sanity: every node z-velocity is w
        assert np.allclose(xd0[2:3 * P.n_nodes:3], w)
    elif v0_mode == "zero":
        xd0 = np.zeros(P.n_dof)
    else:
        raise ValueError(v0_mode)
    t0 = time.perf_counter()
    traj = simulate(P, xi0=xi0, xi_dot0=xd0, progress=False)
    return {
        "label": label,
        "w": w,
        "v0_mode": v0_mode,
        "P": P,
        "traj": traj,
        "runtime_s": time.perf_counter() - t0,
    }


def _pack_pair(name: str, description: str, r0: dict, r1: dict,
               t_s: float | None = None,
               t_compare: np.ndarray | None = None) -> dict:
    P = r0["P"]
    cmp = max_shape_dev_equal_t(
        r0["traj"], r1["traj"], P.L,
        t_compare=t_compare,
        output_dt=P.output_dt,
        t_end=P.t_end,
    )
    V0 = observables.spider_vertical_velocity(r0["traj"].v[-1])
    V1 = observables.spider_vertical_velocity(r1["traj"].v[-1])
    out = {
        "label": description,
        "per_t_dev_over_L": cmp["per_t"],
        "max_shape_dev_equal_t": cmp["max_shape_dev_equal_t"],
        "t_compare": cmp["t_compare"],
        "t_used": cmp["t_used"],
        "dV_m_per_s": V1 - V0,
        "status": {0.0: r0["traj"].outcome["status"],
                   0.5: r1["traj"].outcome["status"]},
        "runtime_s": {0.0: r0["runtime_s"], 0.5: r1["runtime_s"]},
        "t_exit_s": {0.0: float(r0["traj"].t[-1]), 0.5: float(r1["traj"].t[-1])},
    }
    if t_s is not None:
        out["t_s"] = t_s
        out["decay_time_s"] = _decay_time(cmp["t_compare"], cmp["per_t"])
        out["decay_definition"] = (
            "first compared t at which per_t_dev_over_L <= max(per_t)/e"
        )
    return out


def run_invariance_equal_t(parallel: bool = True, dt: float | None = None,
                           cache_dir: str | None = None,
                           t_end: float = INVARIANCE_EQUAL_T_END,
                           output_dt: float = INVARIANCE_EQUAL_T_OUTPUT) -> dict:
    """Equal-time invariance: co-moving IC (primary) and transient v0=0 (secondary).

    Primary (comoving): w-run starts with every node velocity = w z_hat (at rest
    relative to the air); w=0 keeps v0=0. Positions identical.

    Secondary (transient): both runs use v0=0 (different relative initial state);
    reports decay time vs Stokes time t_s.

    If ``cache_dir`` is set, each finished trajectory is written as an ``.npz`` so a
    later ``--from-cache`` can rebuild the comparison without re-integrating.
    """
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    if dt is None:
        dt = _select_fixed_dt()
    w = INVARIANCE_EQUAL_T_W
    jobs = [
        ("w0", 0.0, "zero", dt, t_end, output_dt),
        ("w_comoving", w, "comoving", dt, t_end, output_dt),
        ("w_transient", w, "zero", dt, t_end, output_dt),
    ]
    if parallel:
        with ProcessPoolExecutor(max_workers=3) as pool:
            outs = list(pool.map(_simulate_invariance_job, jobs))
    else:
        outs = [_simulate_invariance_job(j) for j in jobs]
    by_label = {o["label"]: o for o in outs}
    for o in outs:
        print(f"invariance equal-t {o['label']}: t_exit={o['traj'].t[-1]:.4f} "
              f"n_out={len(o['traj'].t)} runtime={o['runtime_s']:.1f} s "
              f"status={o['traj'].outcome['status']}", flush=True)

    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        for o in outs:
            path = cdir / f"{o['label']}.npz"
            tr = o["traj"]
            np.savez_compressed(
                path,
                t=tr.t, x=tr.x, v=tr.v, theta=tr.theta,
                runtime_s=o["runtime_s"], w=o["w"],
                v0_mode=np.array(o["v0_mode"]),
                dt=dt, t_end=t_end, output_dt=output_dt,
                status=np.array(tr.outcome.get("status", "")),
            )
            print(f"cached {path}", flush=True)

    return _finalize_invariance_equal_t(by_label, dt, t_end=t_end, output_dt=output_dt)


def _traj_from_cache(path, P):
    """Minimal trajectory-like object from an ``.npz`` cache file."""
    from types import SimpleNamespace
    data = np.load(path, allow_pickle=False)
    return SimpleNamespace(
        t=data["t"], x=data["x"], v=data["v"], theta=data["theta"],
        outcome={"status": str(data["status"])},
    )


def load_invariance_equal_t_cache(cache_dir: str, dt: float,
                                  t_end: float | None = None,
                                  output_dt: float | None = None) -> dict:
    """Rebuild the equal-t result dict from cached ``.npz`` trajectories."""
    from pathlib import Path
    cdir = Path(cache_dir)
    by_label = {}
    for label, w, v0_mode in (("w0", 0.0, "zero"),
                              ("w_comoving", INVARIANCE_EQUAL_T_W, "comoving"),
                              ("w_transient", INVARIANCE_EQUAL_T_W, "zero")):
        path = cdir / f"{label}.npz"
        data = np.load(path, allow_pickle=False)
        te = float(data["t_end"]) if t_end is None and "t_end" in data.files else (
            INVARIANCE_EQUAL_T_END if t_end is None else t_end)
        od = float(data["output_dt"]) if output_dt is None and "output_dt" in data.files else (
            INVARIANCE_EQUAL_T_OUTPUT if output_dt is None else output_dt)
        if t_end is not None:
            te = t_end
        if output_dt is not None:
            od = output_dt
        P = invariance_equal_t_params(float(data["w"]), dt, t_end=te, output_dt=od)
        runtime = float(data["runtime_s"])
        by_label[label] = {
            "label": label, "w": float(data["w"]), "v0_mode": v0_mode,
            "P": P, "traj": _traj_from_cache(path, P), "runtime_s": runtime,
        }
        print(f"loaded cache {path}: t_exit={by_label[label]['traj'].t[-1]:.4f} "
              f"n_out={len(by_label[label]['traj'].t)} runtime={runtime:.1f} s")
    # Prefer times stored in the first cache file.
    sample = np.load(cdir / "w0.npz", allow_pickle=False)
    te = float(sample["t_end"]) if "t_end" in sample.files else (
        t_end if t_end is not None else INVARIANCE_EQUAL_T_END)
    od = float(sample["output_dt"]) if "output_dt" in sample.files else (
        output_dt if output_dt is not None else INVARIANCE_EQUAL_T_OUTPUT)
    return _finalize_invariance_equal_t(by_label, dt, t_end=te, output_dt=od)


def _finalize_invariance_equal_t(by_label: dict, dt: float,
                                 t_end: float = INVARIANCE_EQUAL_T_END,
                                 output_dt: float = INVARIANCE_EQUAL_T_OUTPUT) -> dict:
    P0 = by_label["w0"]["P"]
    t_s = stokes_time(P0)
    t_compare = _equal_t_compare_grid(output_dt, t_end)
    comoving = _pack_pair(
        "comoving",
        "co-moving initial velocity: w-run has v0 = w z_hat on all nodes "
        "(at rest relative to the air); w=0 has v0=0; positions identical",
        by_label["w0"], by_label["w_comoving"],
        t_compare=t_compare,
    )
    transient = _pack_pair(
        "transient",
        "transient, different relative initial state: v0=0 for both runs",
        by_label["w0"], by_label["w_transient"],
        t_s=t_s,
        t_compare=t_compare,
    )
    print(f"comoving max_shape_dev_equal_t = {comoving['max_shape_dev_equal_t']:.6e}",
          flush=True)
    print(f"transient max_shape_dev_equal_t = {transient['max_shape_dev_equal_t']:.6e} "
          f"decay_time={transient['decay_time_s']:.3f} s  t_s={t_s:.3f} s",
          flush=True)

    return {
        "dt": dt,
        "output_dt": output_dt,
        "t_end": t_end,
        "eps": P0.eps,
        "K": P0.K,
        "adaptive_dt": False,
        "stop_on_steady": False,
        "w": INVARIANCE_EQUAL_T_W,
        "params": {
            "w0": by_label["w0"]["P"],
            "w_comoving": by_label["w_comoving"]["P"],
            "w_transient": by_label["w_transient"]["P"],
        },
        "traj": {
            "w0": by_label["w0"]["traj"],
            "w_comoving": by_label["w_comoving"]["traj"],
            "w_transient": by_label["w_transient"]["traj"],
        },
        "runtime_s": {lab: by_label[lab]["runtime_s"]
                      for lab in ("w0", "w_comoving", "w_transient")},
        "comoving": comoving,
        "transient": transient,
        "max_shape_dev_equal_t": comoving["max_shape_dev_equal_t"],
        "dV_m_per_s": comoving["dV_m_per_s"],
        "t_compare": comoving["t_compare"],
        "per_t_dev_over_L": comoving["per_t_dev_over_L"],
    }


# ---------------------------------------------------------------------------
# Stage A -- Kinematic Simulation validation (fig_ksvalid, fig_ksvalid_pdf)
# ---------------------------------------------------------------------------
KSVALID_SIGMA = 0.25
KSVALID_ELL = 1.0
KSVALID_SEEDS = 50
KSVALID_NPTS = 4096
KSVALID_SPAN_ELL = 20.0
KSVALID_NBINS = 60


def _longitudinal_target(k1: float) -> float:
    """One-sided longitudinal spectrum E_11(k1) implied by E(k) for isotropic turb.

    E_11(k1) = int_{k1}^inf (E(k)/k)(1 - (k1/k)^2) dk (Pope, Turbulent Flows);
    with this convention int_0^inf E_11 dk1 = sigma^2.
    """
    from scipy import integrate
    if k1 <= 0.0:
        k1 = 1e-12
    val, _ = integrate.quad(
        lambda k: von_karman_E(k, KSVALID_SIGMA, KSVALID_ELL) / k
        * (1.0 - (k1 / k) ** 2),
        k1, np.inf, limit=200)
    return float(val)


def ks_validation_spectrum(N_k: int, renormalize: bool = False) -> dict:
    """Measured vs target 1-D longitudinal spectrum of w, averaged over seeds.

    A line of ``KSVALID_NPTS`` points spanning ``KSVALID_SPAN_ELL * ell`` along z;
    w = u_z (longitudinal). One-sided periodogram averaged over ``KSVALID_SEEDS``
    seeds, then log-binned. Returns arrays k, E_target, E_measured.
    """
    span = KSVALID_SPAN_ELL * KSVALID_ELL
    N = KSVALID_NPTS
    dz = span / N
    z = np.linspace(0.0, span, N, endpoint=False)
    acc = np.zeros(N // 2 + 1)
    for s in range(KSVALID_SEEDS):
        ks = KinematicSimulation(sigma=KSVALID_SIGMA, ell=KSVALID_ELL, U_h=0.0,
                                 N_k=N_k, seed=s, L=0.5, N_t=100,
                                 renormalize=renormalize)
        line = np.zeros((N, 3))
        line[:, 2] = z
        w = ks.u(line, 0.0)[:, 2]
        W = np.fft.rfft(w)
        psd = (np.abs(W) ** 2) * dz / N
        psd[1:-1] *= 2.0  # one-sided
        acc += psd
    acc /= KSVALID_SEEDS
    k1 = 2.0 * np.pi * np.fft.rfftfreq(N, d=dz)
    measured = acc / (2.0 * np.pi)  # int_0^inf measured dk1 = variance

    # log-bin over the resolved band (skip k1=0 DC bin)
    lo = k1[1]
    hi = k1[-1]
    edges = np.geomspace(lo, hi, KSVALID_NBINS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    E_meas = np.full(KSVALID_NBINS, np.nan)
    for b in range(KSVALID_NBINS):
        m = (k1 >= edges[b]) & (k1 < edges[b + 1])
        if np.any(m):
            E_meas[b] = float(np.mean(measured[m]))
    keep = ~np.isnan(E_meas)
    centers = centers[keep]
    E_meas = E_meas[keep]
    E_tgt = np.array([_longitudinal_target(k) for k in centers])
    return {"N_k": N_k, "k": centers, "E_target": E_tgt, "E_measured": E_meas}


def ks_validation_pdf(N_k: int, n_bins: int = 61, renormalize: bool = False) -> dict:
    """PDF of w (=u_z) over all seeds/points vs a Gaussian with std sigma."""
    span = KSVALID_SPAN_ELL * KSVALID_ELL
    N = KSVALID_NPTS
    z = np.linspace(0.0, span, N, endpoint=False)
    samples = []
    for s in range(KSVALID_SEEDS):
        ks = KinematicSimulation(sigma=KSVALID_SIGMA, ell=KSVALID_ELL, U_h=0.0,
                                 N_k=N_k, seed=s, L=0.5, N_t=100,
                                 renormalize=renormalize)
        line = np.zeros((N, 3))
        line[:, 2] = z
        samples.append(ks.u(line, 0.0)[:, 2])
    w = np.concatenate(samples)
    lim = 4.0 * KSVALID_SIGMA
    edges = np.linspace(-lim, lim, n_bins + 1)
    pdf, _ = np.histogram(w, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gauss = (np.exp(-centers ** 2 / (2.0 * KSVALID_SIGMA ** 2))
             / (KSVALID_SIGMA * np.sqrt(2.0 * np.pi)))
    return {"N_k": N_k, "w_bin_center": centers, "pdf": pdf, "gaussian": gauss}


def run_ks_validation(N_k_values=(100, 200), renormalize: bool = False) -> dict:
    """Full KS validation for the requested N_k values (fig_ksvalid / _pdf)."""
    spectra = [ks_validation_spectrum(nk, renormalize=renormalize)
               for nk in N_k_values]
    pdfs = [ks_validation_pdf(nk, renormalize=renormalize) for nk in N_k_values]
    energy_fraction = {}
    renorm_factor = {}
    for nk in N_k_values:
        ks = KinematicSimulation(sigma=KSVALID_SIGMA, ell=KSVALID_ELL, U_h=0.0,
                                 N_k=nk, seed=0, L=0.5, N_t=100,
                                 renormalize=renormalize)
        energy_fraction[nk] = ks.energy_fraction
        renorm_factor[nk] = ks.renorm_factor
    return {"spectra": spectra, "pdfs": pdfs, "energy_fraction": energy_fraction,
            "renorm_factor": renorm_factor, "renormalize": renormalize,
            "sigma": KSVALID_SIGMA, "ell": KSVALID_ELL}


# ---------------------------------------------------------------------------
# Stage A -- Table w_c (tab:wc)  ->  results/tab_wc.csv
# ---------------------------------------------------------------------------
WC_E = 7.41e3
WC_N = (1, 2, 4, 8)
WC_M_MG = (0.1, 1.0, 10.0)
WC_Q_NC = (0.0, 0.6)


def wc_params(N: int, m_mg: float, q_nC: float, N_t: int = 100,
              t_end: float | None = None) -> Params:
    """Tip charge, constant E, still air, Q_s=0; q per thread set directly.

    Stage B: rows with m = 10 mg use ``t_end = 40``; other rows keep Stage A
    default ``t_end = 8``.
    """
    if t_end is None:
        t_end = 40.0 if abs(m_mg - 10.0) < 1e-12 else 8.0
    P = Params(
        N=N, N_t=N_t, L=0.5, m=m_mg * 1e-6, Q_s=0.0, charge_model="tip",
        field_model="constant", E_constant=WC_E,
        **_steady(t_end=t_end, t_w=0.1),
    )
    P.Q_t = q_nC * 1e-9
    return P


def wc_analytic(P: Params) -> float:
    """w_c = w_s (1 - Fbar_l), w_s = m g/(N eta_par L + zeta_s), Fbar_l = N q E/(m g)."""
    w_s = P.m * P.g / (P.N * P.eta_par * P.L + P.zeta_s)
    fbar_l = P.N * P.Q_t * P.E_constant / (P.m * P.g)
    return w_s * (1.0 - fbar_l)


def _threads_fold_below_spider(topo: Topology, X: np.ndarray) -> bool:
    """True if any thread node ends up below the spider node (z < z_spider)."""
    z0 = X[0, 2]
    return bool(np.any(X[1:, 2] < z0 - 1e-9))


def run_wc_job(spec: tuple) -> dict:
    """Pickable worker for one (N, m_mg, q_nC) tab_wc entry."""
    N, m_mg, q_nC = spec
    P = wc_params(N, m_mg, q_nC)
    t0 = time.perf_counter()
    traj = simulate(P)
    runtime = time.perf_counter() - t0
    topo = Topology(P)
    V = observables.spider_vertical_velocity(traj.v[-1])
    wc_num = -V
    wc_ana = wc_analytic(P)
    denom = abs(wc_ana) if abs(wc_ana) > 1e-30 else 1.0
    fold = _threads_fold_below_spider(topo, traj.x[-1])
    return {
        "N": N, "m_mg": m_mg, "q_nC": q_nC,
        "wc_numeric": wc_num, "wc_analytic": wc_ana,
        "rel_err_pct": 100.0 * abs(wc_num - wc_ana) / denom,
        "fold_below_spider": fold,
        "status": traj.outcome["status"],
        "t_exit": float(traj.t[-1]),
        "runtime_s": runtime,
    }


def wc_specs() -> list:
    return [(N, m, q) for N in WC_N for m in WC_M_MG for q in WC_Q_NC]


# ---------------------------------------------------------------------------
# Stage A -- Lateral relaxation from results/equal_t_cache  ->  results/relax.csv
# ---------------------------------------------------------------------------
def _exp_decay_time(t: np.ndarray, y: np.ndarray, t_min: float = 0.5) -> float:
    """Fit y ~ A exp(-t/tau) for t >= t_min via least squares on log(y)."""
    m = (t >= t_min) & (y > 0.0)
    if np.count_nonzero(m) < 2:
        return float("nan")
    slope, _ = np.polyfit(t[m], np.log(y[m]), 1)
    if slope >= 0.0:
        return float("inf")
    return float(-1.0 / slope)


def run_lateral_relaxation(cache_dir: str, dt: float) -> dict:
    """R(t) and shape relaxation from the transient (v0=0) equal-t cache runs.

    Uses w0.npz (w=0) and w_transient.npz (w=0.5, v0=0). Reports dR/L, dshape/L
    per compared time and exponential decay times for t >= 0.5 s.
    """
    from pathlib import Path
    cdir = Path(cache_dir)
    P = invariance_equal_t_params(0.0, dt)
    topo = Topology(P)
    d0 = np.load(cdir / "w0.npz", allow_pickle=False)
    dw = np.load(cdir / "w_transient.npz", allow_pickle=False)
    t0, x0 = d0["t"], d0["x"]
    tw, xw = dw["t"], dw["x"]
    output_dt = float(d0["output_dt"]) if "output_dt" in d0.files else INVARIANCE_EQUAL_T_OUTPUT
    t_end = float(d0["t_end"]) if "t_end" in d0.files else INVARIANCE_EQUAL_T_END
    t_compare = _equal_t_compare_grid(output_dt, t_end)
    t_tol = 0.5 * output_dt

    t_used, dR_over_L, dshape_over_L = [], [], []
    for tc in t_compare:
        i0 = int(np.argmin(np.abs(t0 - tc)))
        iw = int(np.argmin(np.abs(tw - tc)))
        if abs(t0[i0] - tc) > t_tol or abs(tw[iw] - tc) > t_tol:
            raise RuntimeError(f"relax: missing matched output at t={tc}")
        R0 = observables.tip_radius(topo, x0[i0])
        Rw = observables.tip_radius(topo, xw[iw])
        dR_over_L.append(abs(Rw - R0) / P.L)
        dshape = np.max(np.abs((x0[i0] - x0[i0][0]) - (xw[iw] - xw[iw][0])))
        dshape_over_L.append(dshape / P.L)
        t_used.append(0.5 * (float(t0[i0]) + float(tw[iw])))
    t_used = np.array(t_used)
    dR_over_L = np.array(dR_over_L)
    dshape_over_L = np.array(dshape_over_L)
    return {
        "t": t_used,
        "dR_over_L": dR_over_L,
        "dshape_over_L": dshape_over_L,
        "tau_R_s": _exp_decay_time(t_used, dR_over_L, 0.5),
        "tau_shape_s": _exp_decay_time(t_used, dshape_over_L, 0.5),
        "t_s": stokes_time(P),
        "fit_t_min_s": 0.5,
    }


# ---------------------------------------------------------------------------
# Stage A -- Pilot cost (alg:sweep)  ->  results/pilot.csv  (NOT production)
# ---------------------------------------------------------------------------
PILOT_N = (1, 2, 4, 8)
PILOT_SEEDS = (0, 1, 2)
PILOT_FBAR = 1.0
PILOT_SIGMA_W = 0.25
PILOT_ELL = 1.0
PILOT_U_H = 1.0
PILOT_N_K = 100
PILOT_Z0 = 0.5
PILOT_H = 2.0
PILOT_T_END = 60.0


def pilot_params(N: int, seed: int, N_t: int = 100,
                 release_mode: str = "clamped", t_end: float = PILOT_T_END) -> Params:
    """Pilot config: Fbar_l=1, kinematic turbulence, clamped release, Alg.2 stop."""
    P = Params(
        N=N, N_t=N_t, L=0.5, m=1e-6, Q_s=0.0, charge_model="tip",
        field_model="constant", E_constant=WC_E,
        flow_model="kinematic", sigma_w=PILOT_SIGMA_W, ell=PILOT_ELL,
        U_h=PILOT_U_H, turb_N_k=PILOT_N_K, turb_seed=seed,
        z0=PILOT_Z0, h=PILOT_H, t_end=t_end,
        release_mode=release_mode,
        use_alg2_stopping=True, stop_on_steady=False,
        adaptive_dt=True, dt0=1e-4, dt_max=1e-2, output_dt=1e-2,
        delta=1e-6, t_w=0.05,
    )
    P.Q_t = charge_for_lift_ratio(P, WC_E, PILOT_FBAR)
    return P


def run_pilot_job(spec: tuple) -> dict:
    """Pickable worker for one pilot run; reports cost + outcome.

    spec = (N, seed) or (N, seed, t_end, N_t). ``t_end``/``N_t`` allow a cheap
    smoke run without touching the production defaults.
    """
    if len(spec) == 2:
        N, seed = spec
        t_end, N_t = PILOT_T_END, 100
    else:
        N, seed, t_end, N_t = spec
    P = pilot_params(N, seed, N_t=N_t, t_end=t_end)
    t0 = time.perf_counter()
    traj = simulate(P)
    wall = time.perf_counter() - t0
    diag = traj.outcome.get("diag", {})
    sim_time = float(traj.t[-1])
    return {
        "N": N, "seed": seed,
        "outcome": traj.outcome.get("status", "timeout"),
        "exit_time": float(traj.outcome.get("exit_time") or sim_time),
        "simulated_time": sim_time,
        "wall_time_s": wall,
        "wall_over_sim": wall / sim_time if sim_time > 0 else float("nan"),
        "mean_dt": float(diag.get("mean_dt", float("nan"))),
        "min_dt": float(diag.get("min_dt", float("nan"))),
        "newton_failures": int(diag.get("newton_failures", 0)),
        "max_abs_u": float(diag.get("max_abs_u", float("nan"))),
        "n_steps": int(diag.get("n_steps", 0)),
    }


def pilot_specs(t_end: float = PILOT_T_END, N_t: int = 100) -> list:
    return [(N, s, t_end, N_t) for N in PILOT_N for s in PILOT_SEEDS]
