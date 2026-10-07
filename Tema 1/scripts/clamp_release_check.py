"""Clamped-release diagnostic (read-only; does not modify production code).

Outputs under results/clamp_check/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning import observables, sweep as SB
from ballooning.fields import ZeroFlow, make_field
from ballooning.geometry import Topology, initial_state
from ballooning.integrator import simulate
from ballooning.params import Params
from ballooning.studies import WC_E, charge_for_lift_ratio

OUT = ROOT / "results" / "clamp_check"
KE = 8.9875517923e9


def S_N(N: int) -> float:
    return float(sum(1.0 / math.sin(math.pi * j / N) for j in range(1, N)))


def opening_analytic(N: int, m: float = SB.SWEEP_M_KG, L: float = 0.5,
                     E: float = WC_E) -> dict:
    """(S_N Lambda/4)^{1/3} for F_l=1 (tau=1 → g(tau)=1)."""
    q = m * 9.81 / (N * E)
    Lam = KE * q / (E * L ** 2)
    sn = S_N(N)
    rl = (sn * Lam / 4.0) ** (1.0 / 3.0)
    return {"N": N, "q": q, "Lambda": Lam, "S_N": sn, "RL_an": rl}


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{float(v):.9e}"
    return str(v)


def _write_csv(path: Path, header: list[str], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            if isinstance(r, dict):
                w.writerow([_fmt(r.get(h, "")) for h in header])
            else:
                w.writerow([_fmt(x) for x in r])
    print(f"wrote {path}", flush=True)


def clamp_params(N: int, N_t: int = 50, t_end: float = 10.0,
                 stop_on_steady: bool = False,
                 fbar: float = 1.0) -> Params:
    """Still-air clamp setup: spider fixed, F_l = fbar, tip charge."""
    P = Params(
        N=N, N_t=N_t, L=0.5, m=SB.SWEEP_M_KG, Q_s=0.0, charge_model="tip",
        field_model="constant", E_constant=WC_E,
        flow_model="zero",
        z0=SB.SWEEP_Z0, h=SB.SWEEP_H, t_end=t_end,
        release_mode="free",
        use_alg2_stopping=False, stop_on_steady=stop_on_steady,
        adaptive_dt=True, dt0=1e-4, dt_max=SB.load_dt_max(),
        output_dt=0.01, delta=1e-6, t_w=0.05,
    )
    P.Q_t = charge_for_lift_ratio(P, WC_E, fbar)
    return P


def run_clamp_phase(N: int, t_end: float, stop_on_steady: bool) -> dict:
    """Phase-1 only: fixed spider, still air."""
    P = clamp_params(N, t_end=t_end, stop_on_steady=stop_on_steady, fbar=1.0)
    field = make_field(P)
    xi0, xi_dot0, topo = initial_state(P)
    t0 = time.perf_counter()
    traj = simulate(P, field=field, flow=ZeroFlow(), xi0=xi0,
                    xi_dot0=xi_dot0, fixed_dofs=[0, 1, 2])
    wall = time.perf_counter() - t0
    R = np.array([observables.tip_radius(topo, traj.x[k]) / P.L
                  for k in range(len(traj.t))])
    th = np.array([observables.tip_angle(topo, traj.x[k])
                   for k in range(len(traj.t))])
    return {
        "N": N, "t": traj.t, "R_over_L": R, "theta_L": th,
        "outcome": traj.outcome, "wall_s": wall, "P": P, "topo": topo,
        "X_final": traj.x[-1], "theta_final": traj.theta[-1],
    }


def time_to_frac(t: np.ndarray, y: np.ndarray, frac: float = 0.99) -> float:
    """First time y reaches frac * y[-1] (assuming y approaches from below)."""
    target = frac * float(y[-1])
    if y[-1] >= y[0]:
        idx = np.where(y >= target)[0]
    else:
        idx = np.where(y <= target)[0]
    return float(t[idx[0]]) if len(idx) else float("nan")


def do_part1_and_2() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Clamp steady criterion (from code) ===", flush=True)
    print(
        "Quantity: spider vertical velocity zdot_0 = xi_dot[2]\n"
        "Test: |zdot_0^n - zdot_0^{n-1}| / max(|zdot_0^n|, 1) < delta\n"
        "  with delta = 1e-6 (sweep_params / Params default)\n"
        "Minimum duration: continuous satisfaction for t_w = 0.05 s\n"
        "Maximum duration: parent t_end (SWEEP_T_END = 60 s for production clamp)\n"
        "Phase 1: stop_on_steady=True, use_alg2_stopping=False, ZeroFlow, "
        "fixed_dofs=[0,1,2]",
        flush=True,
    )

    # Actual clamp duration under production stopping (to steady)
    dur_rows = []
    for N in (2, 4, 8):
        rec = run_clamp_phase(N, t_end=60.0, stop_on_steady=True)
        t_exit = float(rec["outcome"].get("exit_time", rec["t"][-1]))
        dur_rows.append({
            "N": N, "sigma_w": 0.30, "x": 0,
            "phase1_t_exit": t_exit,
            "status": rec["outcome"].get("status"),
            "steady_state": bool(rec["outcome"].get("steady_state", False)),
            "R_over_L_at_stop": float(rec["R_over_L"][-1]),
            "theta_L_at_stop": float(rec["theta_L"][-1]),
            "wall_s": rec["wall_s"],
            "delta": 1e-6, "t_w": 0.05, "t_end_cap": 60.0,
        })
        print(f"  N={N}: clamp stopped at t={t_exit:.6f}s "
              f"status={rec['outcome'].get('status')} "
              f"R/L={rec['R_over_L'][-1]:.6e}", flush=True)
    _write_csv(OUT / "clamp_duration.csv",
               ["N", "sigma_w", "x", "phase1_t_exit", "status", "steady_state",
                "R_over_L_at_stop", "theta_L_at_stop", "wall_s",
                "delta", "t_w", "t_end_cap"],
               dur_rows)

    # 10 s forced clamp (no early stop)
    print("=== 10 s still-air clamp (F_l=1) ===", flush=True)
    series_rows = []
    summary = []
    for N in (2, 4, 8):
        rec = run_clamp_phase(N, t_end=10.0, stop_on_steady=False)
        an = opening_analytic(N)
        R_asym = float(rec["R_over_L"][-1])
        t99 = time_to_frac(rec["t"], rec["R_over_L"], 0.99)
        for ti, ri, thi in zip(rec["t"], rec["R_over_L"], rec["theta_L"]):
            series_rows.append({"N": N, "t": ti, "R_over_L": ri, "theta_L": thi})
        summary.append({
            "N": N, "R_over_L_asym": R_asym, "t_99pct": t99,
            "RL_an": an["RL_an"], "Lambda": an["Lambda"], "S_N": an["S_N"],
            "rel_err_vs_an_pct": 100.0 * abs(R_asym - an["RL_an"]) / an["RL_an"],
            "frame0_from_opening": {2: 0.065872058, 4: 0.072946437,
                                   8: 0.073883723}[N],
        })
        print(f"  N={N}: R/L(10s)={R_asym:.6e}  an={an['RL_an']:.6e}  "
              f"err={summary[-1]['rel_err_vs_an_pct']:.2f}%  "
              f"t99={t99:.4f}s  frame0={summary[-1]['frame0_from_opening']:.6e}",
              flush=True)
        # save final relaxed state for item 3
        np.savez_compressed(
            OUT / f"relaxed_N{N}.npz",
            X=rec["X_final"], theta=rec["theta_final"],
            R_over_L=R_asym, t=rec["t"][-1],
        )
    _write_csv(OUT / "clamp_10s_series.csv",
               ["N", "t", "R_over_L", "theta_L"], series_rows)
    _write_csv(OUT / "clamp_10s_summary.csv",
               ["N", "R_over_L_asym", "t_99pct", "RL_an", "Lambda", "S_N",
                "rel_err_vs_an_pct", "frame0_from_opening"],
               summary)
    return {"duration": dur_rows, "summary_10s": summary}


def _relaxed_job(spec: dict) -> dict:
    """One sweep realization released from precomputed fully-relaxed clamp state."""
    N = int(spec["N"])
    sigma_w = float(spec["sigma_w"])
    x = int(spec["x"])
    i = int(spec["i"])
    pt = SB.SweepPoint(N=N, sigma_w=sigma_w, x=x, i=i, split="main")
    P = SB.sweep_params(pt, N_t=SB.SWEEP_N_T, t_end=SB.SWEEP_T_END,
                        output_dt=SB.SWEEP_OUTPUT_DT, dt_max=SB.load_dt_max())
    # load relaxed clamp state
    z = np.load(OUT / f"relaxed_N{N}.npz")
    X = np.asarray(z["X"], dtype=float)
    th = np.asarray(z["theta"], dtype=float)
    # place spider at z0 (relaxed state already has spider at z0 from clamp)
    X = X.copy()
    X[0] = np.array([0.0, 0.0, SB.SWEEP_Z0])
    xi0 = np.concatenate([X.reshape(-1), th])
    xi_dot0 = np.zeros(P.n_dof)
    # free release into turbulence (bypass clamp phase)
    P2 = replace(P, release_mode="free")
    t0 = time.perf_counter()
    traj = simulate(P2, xi0=xi0, xi_dot0=xi_dot0)
    wall = time.perf_counter() - t0
    topo = Topology(P2)
    R = [observables.tip_radius(topo, traj.x[k]) / P2.L for k in range(len(traj.t))]
    status = traj.outcome.get("status", "timeout")
    exit_time = traj.outcome.get("exit_time")
    if exit_time is None:
        exit_time = float(traj.t[-1])
    return {
        "N": N, "sigma_w": sigma_w, "x": x, "i": i, "seed": pt.seed,
        "Fbar_l": pt.Fbar_l, "split": "main",
        "outcome": status, "exit_time": float(exit_time),
        "R_over_L_mean": float(np.mean(R)),
        "R_over_L_std": float(np.std(R)),
        "wall_time_s": wall,
    }


def do_part3(workers: int, M: int = 200) -> None:
    print("=== Part 3: release from fully relaxed clamp ===", flush=True)
    specs = []
    for N in (4, 8):
        if not (OUT / f"relaxed_N{N}.npz").exists():
            raise SystemExit(f"missing relaxed_N{N}.npz — run part 1-2 first")
        for x in (0, 1):
            for i in range(M):
                specs.append({"N": N, "sigma_w": 0.30, "x": x, "i": i})
    out_csv = OUT / "relaxed_release_sweep.csv"
    header = ["N", "sigma_w", "x", "i", "seed", "Fbar_l", "split", "outcome",
              "exit_time", "R_over_L_mean", "R_over_L_std", "wall_time_s"]
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            for r in csv.DictReader(f):
                done.add((int(r["N"]), float(r["sigma_w"]), int(r["x"]), int(r["i"])))
    pending = [s for s in specs
               if (s["N"], s["sigma_w"], s["x"], s["i"]) not in done]
    print(f"  total={len(specs)} done={len(done)} pending={len(pending)} "
          f"workers={workers}", flush=True)
    new_file = not out_csv.exists()
    t0 = time.perf_counter()
    n_fin = 0
    with ProcessPoolExecutor(max_workers=workers) as pool, \
            open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n",
                           extrasaction="ignore")
        if new_file:
            w.writeheader()
        futs = {pool.submit(_relaxed_job, s): s for s in pending}
        for fut in as_completed(futs):
            row = fut.result()
            w.writerow({h: _fmt(row.get(h, "")) for h in header})
            f.flush()
            n_fin += 1
            if n_fin % 10 == 0 or n_fin == len(pending):
                elapsed = time.perf_counter() - t0
                rate = n_fin / elapsed if elapsed > 0 else 0
                eta = (len(pending) - n_fin) / rate if rate > 0 else float("inf")
                print(f"  progress {n_fin}/{len(pending)} "
                      f"elapsed={elapsed/3600:.2f}h ETA={eta/3600:.2f}h "
                      f"last N={row['N']} x={row['x']} i={row['i']} "
                      f"{row['outcome']}", flush=True)
    _write_phase_compare()


def _write_phase_compare() -> None:
    """Compare relaxed-release P and R/L to original sweep."""
    path = OUT / "relaxed_release_sweep.csv"
    if not path.exists():
        return
    from collections import defaultdict
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            groups[(int(r["N"]), float(r["sigma_w"]), int(float(r["x"])))].append(r)
    # original sweep aggregates
    sweep_g = defaultdict(list)
    with open(ROOT / "results" / "sweep.csv", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("valid", "true")).lower() not in ("true", "1", "yes"):
                continue
            key = (int(r["N"]), float(r["sigma_w"]), int(float(r["x"])))
            if key[0] in (4, 8) and abs(key[1] - 0.30) < 1e-12 and key[2] in (0, 1):
                sweep_g[key].append(r)

    rows = []
    for key in sorted(groups):
        grp = groups[key]
        n_up = sum(1 for r in grp if r["outcome"] == "rise")
        n_down = sum(1 for r in grp if r["outcome"] == "fall")
        n_to = sum(1 for r in grp if r["outcome"] == "timeout")
        P, Plo, Phi = SB.wilson_ci(n_up, n_up + n_down)
        Rmean = float(np.mean([float(r["R_over_L_mean"]) for r in grp]))
        sg = sweep_g.get(key, [])
        if sg:
            s_up = sum(1 for r in sg if r["outcome"] == "rise")
            s_down = sum(1 for r in sg if r["outcome"] == "fall")
            sP, sPlo, sPhi = SB.wilson_ci(s_up, s_up + s_down)
            sR = float(np.mean([float(r["R_over_L_mean"]) for r in sg]))
        else:
            sP = sPlo = sPhi = sR = float("nan")
        rows.append({
            "N": key[0], "sigma_w": key[1], "x": key[2], "M": len(grp),
            "n_up": n_up, "n_down": n_down, "n_timeout": n_to,
            "P_relaxed": P, "P_lo": Plo, "P_hi": Phi,
            "R_over_L_mean_relaxed": Rmean,
            "P_sweep": sP, "P_lo_sweep": sPlo, "P_hi_sweep": sPhi,
            "R_over_L_mean_sweep": sR,
            "M_sweep": len(sg),
        })
        print(f"  N={key[0]} x={key[2]}: P_rel={P:.4f}[{Plo:.4f},{Phi:.4f}] "
              f"P_sw={sP:.4f}[{sPlo:.4f},{sPhi:.4f}]  "
              f"R/L_rel={Rmean:.4e} R/L_sw={sR:.4e}", flush=True)
    _write_csv(OUT / "relaxed_vs_sweep.csv",
               ["N", "sigma_w", "x", "M", "n_up", "n_down", "n_timeout",
                "P_relaxed", "P_lo", "P_hi", "R_over_L_mean_relaxed",
                "P_sweep", "P_lo_sweep", "P_hi_sweep", "R_over_L_mean_sweep",
                "M_sweep"],
               rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts12", action="store_true", help="criterion + durations + 10s")
    ap.add_argument("--part3", action="store_true", help="relaxed-release M=200 grid")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--M", type=int, default=200)
    ns = ap.parse_args()
    if ns.compare_only:
        _write_phase_compare()
        return 0
    if not ns.parts12 and not ns.part3:
        ns.parts12 = True
    report = {}
    if ns.parts12:
        report = do_part1_and_2()
        with open(OUT / "parts12_report.json", "w") as f:
            json.dump({
                "duration": report["duration"],
                "summary_10s": report["summary_10s"],
                "criterion": {
                    "quantity": "zdot_0 (spider vertical velocity)",
                    "test": "|zdot_0^n - zdot_0^{n-1}| / max(|zdot_0^n|, 1) < delta",
                    "delta": 1e-6,
                    "t_w_min_s": 0.05,
                    "t_end_max_s": 60.0,
                },
            }, f, indent=2)
    if ns.part3:
        do_part3(ns.workers, M=ns.M)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
