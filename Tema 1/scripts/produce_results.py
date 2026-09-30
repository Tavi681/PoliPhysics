"""Run every production study and write ``results/``.

One command, deterministic physics (fixed parameters, no random input).
Wall-clock times are measured and are not bit-reproducible. Stdout is copied
to ``results/log.txt``.

Format: comma-separated CSV with a header row, SI units unless the column
name says otherwise. N_t = 100 unless stated.

results/tab_vt.csv            L, ln_L_over_r, vt_analytic, vt_numeric, rel_err_pct
results/tab_valid.csv         test, config, reference, value, rel_err_pct
results/tab_conv.csv          N_t, vt, dvt_pct, R_over_L, dR_pct, cpu_s
results/tab_steady.csv        N, q_nC, Lambda, tau, RL_an, RL_num, theta_L, vt, dmin_um
results/fig_test1.csv         case, t, z0, vz0
results/fig_collapse.csv      N, Fl_bar, vt_num, vt_an, R_over_L, theta_L
results/fig_invariant.csv     s_over_L, T, theta, invariant, invariant_norm
results/fig_shapes.csv        N, thread, s, x, y, z
results/fig_invariance.csv    w, thread, s, x_rel, y_rel, z_rel
results/fig_invariance_t.csv  w, t, z0
results/meta.json             parameters per study, environment, wall-clock
results/log.txt

The convergence runs execute one at a time in the main process so that
``cpu_s`` is not affected by other runs. All other runs are independent and
execute in a process pool, one BLAS thread per worker.

Usage (from ``Tema 1``):
    python scripts/produce_results.py [--workers K]
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import dataclasses
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import scipy

from ballooning import observables
from ballooning.geometry import Topology
from ballooning.io_hdf5 import git_commit_hash
from ballooning.params import Params
from ballooning.studies import (
    PRODUCTION_E,
    chamber_spider_params,
    chamber_tip_params,
    paper_test2_params,
    production_params,
    run_chamber_spider,
    run_chamber_tip,
    run_invariance_equal_t,
    run_normalized,
    run_production,
    run_terminal_velocity,
    terminal_velocity_params,
)

VT_LENGTHS = (0.1, 0.5, 1.0)
VT_FBAR = 2.0
COLLAPSE_N = (1, 2, 4, 8)
COLLAPSE_FBAR = (1.2, 1.5, 2.0, 3.0, 5.0)
STEADY_N = (2, 4, 8)
STEADY_FBAR = 2.0
CONV_NT = (25, 50, 100, 200)
CONV_REF_NT = 200
INVARIANCE_N = 4
INVARIANCE_W = (0.0, 0.5)
INVARIANT_N = 2
INVARIANT_THREAD = 0
INVARIANT_TRIM = 2  # edges excluded at each end when normalizing the invariant


# ---------------------------------------------------------------------------
# Jobs (top-level so they can be pickled by the process pool)
# ---------------------------------------------------------------------------
def _job(spec: tuple):
    kind, args = spec
    if kind == "production":
        return run_production(production_params(**args))
    if kind == "chamber_tip":
        return run_chamber_tip()
    if kind == "chamber_spider":
        return run_chamber_spider()
    if kind == "terminal_velocity":
        return run_terminal_velocity(**args)
    raise ValueError(kind)


def _vt_params(L: float) -> dict:
    return dict(N=1, fbar=VT_FBAR, L=L)


def _collapse_params(N: int, fbar: float) -> dict:
    return dict(N=N, fbar=fbar)


def _invariance_params(w: float) -> dict:
    return dict(N=INVARIANCE_N, fbar=STEADY_FBAR, w=w)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _cmd(args: list[str]) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _environment() -> dict:
    cpu = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    perf = _cmd(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
    eff = _cmd(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
    gpu = ""
    if sys.platform == "darwin":
        for line in _cmd(["system_profiler", "SPDisplaysDataType"]).splitlines():
            if "Chipset Model" in line:
                gpu = line.split(":", 1)[1].strip()
                break
    return {
        "git_commit": git_commit_hash() or "",
        "cpu_model": cpu,
        "cpu_count_logical": os.cpu_count(),
        "cpu_performance_cores": int(perf) if perf.isdigit() else None,
        "cpu_efficiency_cores": int(eff) if eff.isdigit() else None,
        "gpu": gpu or None,
        "gpu_used": False,
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
    }


def _params_dict(P: Params) -> dict:
    d = dataclasses.asdict(P)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.9e}"
    return str(v)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    print(f"wrote {path.name} ({len(rows)} rows)")


def _rel_pct(value: float, reference: float) -> float:
    return 100.0 * abs(value - reference) / abs(reference)


def _thread_arclength(P: Params) -> np.ndarray:
    """Reference arc length of the spider (s=0) and the N_t thread nodes."""
    return P.l0 * np.arange(P.N_t + 1)


def _thread_points(topo: Topology, X: np.ndarray, j: int) -> np.ndarray:
    return np.vstack([X[0], X[topo.thread_nodes[j]]])


def _shape_rows(P: Params, X: np.ndarray, lead: list) -> list[list]:
    topo = Topology(P)
    s = _thread_arclength(P)
    rows = []
    for j in range(P.N):
        pts = _thread_points(topo, X, j) - X[0]
        for k in range(len(s)):
            rows.append(lead + [j, s[k], pts[k, 0], pts[k, 1], pts[k, 2]])
    return rows


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ns = ap.parse_args()

    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        if old.is_file():
            old.unlink()
    log = open(out / "log.txt", "w")

    class _Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            log.write(s)

        def flush(self):
            sys.__stdout__.flush()
            log.flush()

    sys.stdout = _Tee()
    t_script = time.perf_counter()
    env = _environment()
    workers = ns.workers or env["cpu_performance_cores"] or max(1, (os.cpu_count() or 2) // 2)
    print(f"environment: {json.dumps(env)}")
    print(f"workers: {workers}")
    meta: dict = {"environment": env, "workers": workers, "studies": {}}

    try:
        # --- Convergence (sequential, timed) ---
        t0 = time.perf_counter()
        conv = {}
        for N_t in CONV_NT:
            r = run_normalized(N_t)
            conv[N_t] = r
            print(f"conv N_t={N_t} vbar_t={r['vbar_t']:.6f} R/L={r['R_over_L']:.6e} "
                  f"cpu={r['runtime_s']:.2f} s status={r['status']}")
        conv_wall = time.perf_counter() - t0

        # --- Everything else in a pool ---
        jobs: dict[str, tuple] = {}
        for N in sorted(COLLAPSE_N, reverse=True):
            for fb in COLLAPSE_FBAR:
                jobs[f"collapse_N{N}_F{fb}"] = ("production", _collapse_params(N, fb))
        jobs["invariance_w0.5"] = ("production", _invariance_params(0.5))
        for L in VT_LENGTHS:
            jobs[f"vt_L{L}"] = ("production", _vt_params(L))
        jobs["test4a"] = ("chamber_tip", {})
        jobs["test4b"] = ("chamber_spider", {})
        jobs["test3_Qs0"] = ("terminal_velocity", {"Q_s": 0.0})
        jobs["test3_Qs3pC"] = ("terminal_velocity", {"Q_s": 3e-12})

        t0 = time.perf_counter()
        res: dict[str, dict] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {name: pool.submit(_job, spec) for name, spec in jobs.items()}
            for name, fut in futs.items():
                res[name] = fut.result()
                r = res[name]
                print(f"done {name}: status={r['status']} t_exit={r['t_exit']:.4f} s "
                      f"runtime={r['runtime_s']:.1f} s")
        pool_wall = time.perf_counter() - t0
        print(f"pool wall-clock {pool_wall:.1f} s")

        def _run_time(names):
            return float(sum(res[n]["runtime_s"] for n in names))

        # --- tab_vt ---
        rows, names = [], []
        for L in VT_LENGTHS:
            name = f"vt_L{L}"
            names.append(name)
            r = res[name]
            P = production_params(**_vt_params(L))
            rows.append([L, np.log(L / P.r), r["U_analytic"], r["V"],
                         _rel_pct(r["V"], r["U_analytic"])])
        _write_csv(out / "tab_vt.csv",
                   ["L", "ln_L_over_r", "vt_analytic", "vt_numeric", "rel_err_pct"], rows)
        meta["studies"]["tab_vt"] = {
            "description": "N=1, tip charge, constant E, still air, Q_s=0; "
                           "vt in m/s; vt_analytic = (qE - mg)/(eta_par L + zeta_s)",
            "Fbar_l": VT_FBAR,
            "q_rule": "q = Fbar_l m g / E",
            "params": {str(L): _params_dict(production_params(**_vt_params(L)))
                       for L in VT_LENGTHS},
            "status": {n: res[n]["status"] for n in names},
            "run_time_s": _run_time(names),
        }

        # --- fig_test1 ---
        rows = []
        for case, key in (("4a", "test4a"), ("4b", "test4b")):
            r = res[key]
            for t, z, v in zip(r["t"], r["z0"], r["vz0"]):
                rows.append([case, t, z, v])
        _write_csv(out / "fig_test1.csv", ["case", "t", "z0", "vz0"], rows)

        # --- tab_valid ---
        a, b = res["test4a"], res["test4b"]
        q3a, q3b = res["test3_Qs0"], res["test3_Qs3pC"]
        c100 = conv[100]
        print(f"test4a q={a['q']:.5g} C V={a['V']:.6f} m/s t95={a['t95']:.4f} s "
              f"3t_s={a['t95_reference']:.4f} s")
        print(f"test4b q={b['q']:.5g} C V={b['V']:.6f} m/s t95={b['t95']:.4f} s "
              f"3t_s={b['t95_reference']:.4f} s")
        print(f"test4b vz0 peak={b['v_peak']:.6f} m/s, settles within 5% at "
              f"t={b['t_settle_5pct']:.4f} s")
        rows = [
            ["4a velocity", "N=1 chamber tip charge", a["U_target"], a["V"],
             _rel_pct(a["V"], a["U_target"])],
            ["4a t_95", "N=1 chamber tip charge (reference 3 t_s)", a["t95_reference"],
             a["t95"], _rel_pct(a["t95"], a["t95_reference"])],
            ["4b velocity", "N=1 chamber spider charge", b["U_target"], b["V"],
             _rel_pct(b["V"], b["U_target"])],
            ["4b t_95", "N=1 chamber spider charge (reference 3 t_s)", b["t95_reference"],
             b["t95"], _rel_pct(b["t95"], b["t95_reference"])],
            ["5 vbar_t", "N=2 constant E Fbar_l=2 N_t=100", 2.0, c100["vbar_t"],
             _rel_pct(c100["vbar_t"], 2.0)],
            ["3", "N=1 constant E Q_s=0", q3a["U"], q3a["V"], 100.0 * q3a["rel_err"]],
            ["3", "N=1 constant E Q_s=3pC", q3b["U"], q3b["V"], 100.0 * q3b["rel_err"]],
        ]
        _write_csv(out / "tab_valid.csv",
                   ["test", "config", "reference", "value", "rel_err_pct"], rows)
        P4a, P4b = chamber_tip_params(), chamber_spider_params()
        meta["studies"]["tab_valid"] = {
            "description": "velocities in m/s, times in s, vbar_t dimensionless",
            "4a": {"params": _params_dict(P4a), "q_C": a["q"],
                   "q_rule": "q = (m g + U_target (eta_par L + zeta_s)) / E_inf",
                   "U_target": a["U_target"], "t_s": a["t_s"],
                   "status": a["status"], "run_time_s": a["runtime_s"]},
            "4b": {"params": _params_dict(P4b), "q_C": b["q"],
                   "q_rule": "same q as 4a, placed on the spider (Q_s)",
                   "t_s": b["t_s"], "v_peak": b["v_peak"],
                   "t_settle_5pct": b["t_settle_5pct"],
                   "status": b["status"], "run_time_s": b["runtime_s"]},
            "5": {"params": _params_dict(paper_test2_params(100)),
                  "note": "same run as tab_conv N_t=100", "status": c100["status"]},
            "3": {"params_Qs0": _params_dict(terminal_velocity_params(0.0)),
                  "params_Qs3pC": _params_dict(terminal_velocity_params(3e-12)),
                  "U_rule": "(N q E + Q_s E - m g) / (N eta_par L + zeta_s)",
                  "run_time_s": q3a["runtime_s"] + q3b["runtime_s"]},
        }
        meta["studies"]["fig_test1"] = {
            "description": "Test 4a/4b spider trajectories; t [s], z0 [m], vz0 [m/s]",
            "output_dt": P4a.output_dt,
            "run_time_s": a["runtime_s"] + b["runtime_s"],
        }

        # --- tab_conv ---
        ref = conv[CONV_REF_NT]
        rows = []
        for N_t in CONV_NT:
            r = conv[N_t]
            rows.append([N_t, r["vbar_t"], _rel_pct(r["vbar_t"], ref["vbar_t"]),
                         r["R_over_L"], _rel_pct(r["R_over_L"], ref["R_over_L"]),
                         r["runtime_s"]])
        _write_csv(out / "tab_conv.csv",
                   ["N_t", "vt", "dvt_pct", "R_over_L", "dR_pct", "cpu_s"], rows)
        meta["studies"]["tab_conv"] = {
            "description": "N=2, tip, constant E=7.41e3, Fbar_l=2; vt = vbar_t "
                           "(dimensionless); deviations relative to N_t=200; cpu_s is "
                           "wall-clock of a single run executed alone",
            "params": {str(n): _params_dict(paper_test2_params(n)) for n in CONV_NT},
            "status": {str(n): conv[n]["status"] for n in CONV_NT},
            "wall_clock_s": conv_wall,
        }

        # --- fig_collapse ---
        rows, names = [], []
        for N in COLLAPSE_N:
            for fb in COLLAPSE_FBAR:
                name = f"collapse_N{N}_F{fb}"
                names.append(name)
                r = res[name]
                rows.append([N, fb, r["vbar_t"], r["vbar_t_straight"],
                             r["R_over_L"], r["theta_L"]])
        _write_csv(out / "fig_collapse.csv",
                   ["N", "Fl_bar", "vt_num", "vt_an", "R_over_L", "theta_L"], rows)
        meta["studies"]["fig_collapse"] = {
            "description": "tip charge, constant E, still air, Q_s=0, m=1 mg, L=0.5; "
                           "vt = vbar_t = U mu N L / (N q E - m g); vt_an = straight "
                           "vertical threads, mu N L / (N eta_par L + zeta_s); "
                           "R_over_L and theta_L [rad] as in tab_steady",
            "q_rule": "q = Fl_bar m g / (N E)",
            "E": PRODUCTION_E,
            "params": {n: _params_dict(production_params(**_collapse_params(N, fb)))
                       for n, (N, fb) in zip(names, [(N, fb) for N in COLLAPSE_N
                                                     for fb in COLLAPSE_FBAR])},
            "status": {n: res[n]["status"] for n in names},
            "run_time_s": _run_time(names),
        }

        # --- tab_steady (reuses the Fbar_l = 2 collapse runs) ---
        rows, names = [], []
        for N in STEADY_N:
            name = f"collapse_N{N}_F{STEADY_FBAR}"
            names.append(name)
            r = res[name]
            rows.append([N, r["Q_t"] * 1e9, None, None, None, r["R_over_L"],
                         r["theta_L"], r["vbar_t"], r["d_min"] * 1e6])
        _write_csv(out / "tab_steady.csv",
                   ["N", "q_nC", "Lambda", "tau", "RL_an", "RL_num", "theta_L", "vt",
                    "dmin_um"], rows)
        meta["studies"]["tab_steady"] = {
            "description": "tip, still air, constant E=7.41e3, m=1 mg, L=0.5, Fbar_l=2, "
                           "Q_s=0; q in nC per thread; theta_L in rad (mean tip-edge "
                           "angle from vertical); vt = vbar_t (dimensionless); d_min in "
                           "um. Lambda, tau, RL_an left empty (no analytic model here).",
            "source_runs": names,
            "status": {n: res[n]["status"] for n in names},
        }

        # --- fig_shapes ---
        rows = []
        for N in STEADY_N:
            r = res[f"collapse_N{N}_F{STEADY_FBAR}"]
            P = production_params(**_collapse_params(N, STEADY_FBAR))
            rows += _shape_rows(P, r["x_final"], [N])
        _write_csv(out / "fig_shapes.csv", ["N", "thread", "s", "x", "y", "z"], rows)
        meta["studies"]["fig_shapes"] = {
            "description": "final (steady) node positions relative to the spider; s is "
                           "reference arc length from the spider [m]",
            "source_runs": [f"collapse_N{N}_F{STEADY_FBAR}" for N in STEADY_N],
        }

        # --- fig_invariant ---
        P = production_params(**_collapse_params(INVARIANT_N, STEADY_FBAR))
        topo = Topology(P)
        X = res[f"collapse_N{INVARIANT_N}_F{STEADY_FBAR}"]["x_final"]
        e = topo.thread_edges[INVARIANT_THREAD]
        T = observables.edge_tensions(P, topo, X)[e]
        th = observables.edge_angles_from_vertical(topo, X)[e]
        inv = observables.first_integral(P, topo, X)[e]
        ref_inv = float(np.mean(inv[INVARIANT_TRIM:-INVARIANT_TRIM]))
        s_mid = (np.arange(P.N_t) + 0.5) * P.l0 / P.L
        rows = [[s_mid[k], T[k], th[k], inv[k], inv[k] / ref_inv] for k in range(P.N_t)]
        _write_csv(out / "fig_invariant.csv",
                   ["s_over_L", "T", "theta", "invariant", "invariant_norm"], rows)
        spread = (inv[INVARIANT_TRIM:-INVARIANT_TRIM].max()
                  - inv[INVARIANT_TRIM:-INVARIANT_TRIM].min()) / abs(ref_inv)
        print(f"fig_invariant: spread over interior edges {100 * spread:.3f} %")
        meta["studies"]["fig_invariant"] = {
            "description": "per-edge values on one thread of the N=2 steady run; s at "
                           "edge midpoints; T [N]; theta [rad] from vertical; invariant "
                           "= T^beta sin(theta) [N^beta]; invariant_norm = invariant / "
                           "mean over edges excluding the first and last "
                           f"{INVARIANT_TRIM}",
            "beta": P.beta_first_integral,
            "thread": INVARIANT_THREAD,
            "source_run": f"collapse_N{INVARIANT_N}_F{STEADY_FBAR}",
            "interior_spread_pct": 100 * spread,
        }

        # --- fig_invariance / fig_invariance_t ---
        inv_runs = {0.0: res[f"collapse_N{INVARIANCE_N}_F{STEADY_FBAR}"],
                    0.5: res["invariance_w0.5"]}
        rows, rows_t = [], []
        for w in INVARIANCE_W:
            r = inv_runs[w]
            P = production_params(**_invariance_params(w))
            rows += _shape_rows(P, r["x_final"], [w])
            rows_t += [[w, t, z] for t, z in zip(r["t"], r["z0"])]
        _write_csv(out / "fig_invariance.csv",
                   ["w", "thread", "s", "x_rel", "y_rel", "z_rel"], rows)
        _write_csv(out / "fig_invariance_t.csv", ["w", "t", "z0"], rows_t)
        dX = inv_runs[0.5]["x_final"] - inv_runs[0.5]["x_final"][0] - (
            inv_runs[0.0]["x_final"] - inv_runs[0.0]["x_final"][0])
        dV = inv_runs[0.5]["V"] - inv_runs[0.0]["V"]
        print(f"invariance (final / adaptive): V(w=0.5)-V(w=0) = {dV:.9f} m/s, "
              f"max shape deviation / L = {np.max(np.abs(dX)) / 0.5:.3e}")

        # Fixed-dt equal-time spider-frame comparison (t = 0.1, 0.2, ..., 2.0 s)
        eq = run_invariance_equal_t()
        print(f"invariance equal-t (comoving): max = {eq['comoving']['max_shape_dev_equal_t']:.6e}")
        print(f"invariance equal-t (transient): max = {eq['transient']['max_shape_dev_equal_t']:.6e}, "
              f"decay_time = {eq['transient']['decay_time_s']:.3f} s, "
              f"t_s = {eq['transient']['t_s']:.3f} s")

        def _eq_branch(branch: dict) -> dict:
            out = {
                "label": branch["label"],
                "per_t_dev_over_L": list(branch["per_t_dev_over_L"]),
                "max_shape_dev_equal_t": branch["max_shape_dev_equal_t"],
                "dV_m_per_s": branch["dV_m_per_s"],
                "status": {str(k): v for k, v in branch["status"].items()},
                "runtime_s": {str(k): v for k, v in branch["runtime_s"].items()},
                "t_exit_s": {str(k): v for k, v in branch["t_exit_s"].items()},
            }
            for key in ("t_s", "decay_time_s", "decay_definition"):
                if key in branch:
                    out[key] = branch[key]
            return out

        meta["studies"]["fig_invariance"] = {
            "description": "N=4, Fbar_l=2, uniform vertical flow w [m/s]; CSV shapes from "
                           "adaptive production runs. equal_t.comoving: fixed dt, default "
                           "eps/K, w-run starts with v0=w z_hat on all nodes; "
                           "equal_t.transient: both v0=0 (different relative IC), with "
                           "decay_time vs Stokes t_s.",
            "params": {str(w): _params_dict(production_params(**_invariance_params(w)))
                       for w in INVARIANCE_W},
            "params_equal_t": {k: _params_dict(eq["params"][k])
                               for k in ("w0", "w_comoving", "w_transient")},
            "status": {str(w): inv_runs[w]["status"] for w in INVARIANCE_W},
            "t_exit_s": {str(w): inv_runs[w]["t_exit"] for w in INVARIANCE_W},
            "dV_m_per_s": dV,
            "max_shape_dev_over_L": float(np.max(np.abs(dX)) / 0.5),
            "max_shape_dev_equal_t": eq["comoving"]["max_shape_dev_equal_t"],
            "equal_t": {
                "dt": eq["dt"],
                "output_dt": eq["output_dt"],
                "t_end": eq["t_end"],
                "eps": eq["eps"],
                "K": eq["K"],
                "adaptive_dt": eq["adaptive_dt"],
                "stop_on_steady": eq["stop_on_steady"],
                "w": eq["w"],
                "t_compare": list(eq["t_compare"]),
                "comoving": _eq_branch(eq["comoving"]),
                "transient": _eq_branch(eq["transient"]),
            },
            "run_time_s": (inv_runs[0.5]["runtime_s"]
                           + sum(eq["runtime_s"].values())),
        }

        meta["pool_wall_clock_s"] = pool_wall
        meta["total_wall_clock_s"] = time.perf_counter() - t_script
        meta["note"] = ("run_time_s is the sum of per-run wall-clock times; the pool "
                        "runs execute concurrently")
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print("wrote meta.json")
        print(f"total wall-clock {meta['total_wall_clock_s']:.1f} s")
    finally:
        sys.stdout = sys.__stdout__
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
