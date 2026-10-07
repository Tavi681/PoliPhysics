"""Stage B Mac-side checks (no VM).

1. Rerun tab_wc m=10 mg (t_end=40); report V(t_end) and |dV/dt|.
2. Entanglement audit from sweep.csv + selected GCS HDF5.
3. Fill wall_time_s from legacy shard CSVs.
4. Opening baseline R/L at frame 0 (GCS HDF5).
5. Tracer T_L + phase study.

Usage (from Tema 1/):
  .venv/bin/python scripts/stage_b_checks.py --all
  .venv/bin/python scripts/stage_b_checks.py --tab-wc --wall-time --entangle --opening --tracer
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning import observables, studies
from ballooning import sweep as SB
from ballooning.fields import KinematicSimulation
from ballooning.geometry import Topology
from ballooning.integrator import simulate
from ballooning.params import Params

RESULTS = ROOT / "results"
GCS_SHARDS = [
    "gs://authorship-verification-poliphysics/tema1-stageb/shard-0/ml_samples",
    "gs://authorship-verification-poliphysics/tema1-stageb/shard-1/ml_samples",
]
CONTACT_UM = 0.6  # 2 * r = 2 * 300 nm


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
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
                w.writerow([_fmt(v) for v in r])
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# 1. tab_wc m=10
# ---------------------------------------------------------------------------
def _wc_m10_one(spec: tuple) -> dict:
    N, m_mg, q_nC = spec
    P = studies.wc_params(N, m_mg, q_nC, t_end=40.0)
    traj = simulate(P)
    V = traj.v[:, 0, 2]
    t = traj.t
    V_end = float(V[-1])
    if len(t) >= 2 and t[-1] > t[-2]:
        dVdt = abs(float((V[-1] - V[-2]) / (t[-1] - t[-2])))
    else:
        dVdt = float("nan")
    wc_num = -V_end
    wc_ana = studies.wc_analytic(P)
    denom = abs(wc_ana) if abs(wc_ana) > 1e-30 else 1.0
    return {
        "N": N, "m_mg": m_mg, "q_nC": q_nC,
        "wc_numeric": wc_num, "wc_analytic": wc_ana,
        "rel_err_pct": 100.0 * abs(wc_num - wc_ana) / denom,
        "status": traj.outcome.get("status"),
        "t_end": float(t[-1]),
        "V_end": V_end,
        "abs_dVdt_end": dVdt,
    }


def do_tab_wc(workers: int) -> list[dict]:
    print("=== 1. tab_wc m=10 mg, t_end=40 ===", flush=True)
    specs = [(N, 10.0, q) for N in studies.WC_N for q in studies.WC_Q_NC]
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        new_rows = list(pool.map(_wc_m10_one, specs))
    path = RESULTS / "tab_wc.csv"
    existing = []
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if abs(float(r["m_mg"]) - 10.0) > 1e-12:
                    existing.append(r)
    by_key = {(int(r["N"]), float(r["m_mg"]), float(r["q_nC"])): r
              for r in existing}
    for r in new_rows:
        by_key[(r["N"], r["m_mg"], r["q_nC"])] = {
            "N": r["N"], "m_mg": r["m_mg"], "q_nC": r["q_nC"],
            "wc_numeric": r["wc_numeric"], "wc_analytic": r["wc_analytic"],
            "rel_err_pct": r["rel_err_pct"],
        }
        print(f"  N={r['N']} q={r['q_nC']} status={r['status']} "
              f"wc={r['wc_numeric']:.4e} err={r['rel_err_pct']:.3f}% "
              f"V_end={r['V_end']:.6e} |dV/dt|={r['abs_dVdt_end']:.3e} "
              f"t={r['t_end']:.3f}", flush=True)
    ordered = sorted(by_key.values(),
                     key=lambda r: (int(r["N"]), float(r["m_mg"]), float(r["q_nC"])))
    _write_csv(path,
               ["N", "m_mg", "q_nC", "wc_numeric", "wc_analytic", "rel_err_pct"],
               [[r["N"], r["m_mg"], r["q_nC"], r["wc_numeric"],
                 r["wc_analytic"], r["rel_err_pct"]] for r in ordered])
    return new_rows


# ---------------------------------------------------------------------------
# 3. wall_time_s from legacy shards
# ---------------------------------------------------------------------------
def _row_key(r: dict) -> tuple:
    N = int(r["N"])
    sigma_w = float(r["sigma_w"])
    x = int(float(r["x"]))
    seed = int(r["seed"])
    split = str(r.get("split", "main") or "main")
    M = SB.realizations_for(x, split)
    for i in range(M + 8):
        if SB.sweep_seed(N, sigma_w, x, i) == seed:
            return (N, sigma_w, x, i)
    return (N, sigma_w, x, seed)


def do_wall_time() -> dict:
    print("=== 3. fill wall_time_s from legacy shards ===", flush=True)
    legacy: dict[tuple, float] = {}
    for name in ("sweep_shard_00.csv", "sweep_shard_01.csv"):
        p = RESULTS / name
        if not p.exists():
            print(f"  missing {p}", flush=True)
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    legacy[_row_key(r)] = float(r["wall_time_s"])
                except (KeyError, ValueError):
                    pass
    sweep_path = RESULTS / "sweep.csv"
    rows = []
    with open(sweep_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            k = _row_key(r)
            if k in legacy:
                r["wall_time_s"] = f"{legacy[k]:.9e}"
            else:
                r["wall_time_s"] = ""  # NaN when read back
            rows.append(r)
    n_filled = sum(1 for r in rows if r["wall_time_s"] != "")
    n_nan = len(rows) - n_filled
    with open(sweep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  filled={n_filled} nan={n_nan} total={len(rows)}", flush=True)
    return {"filled": n_filled, "nan": n_nan, "total": len(rows)}


# ---------------------------------------------------------------------------
# 2. Entanglement
# ---------------------------------------------------------------------------
def _gsutil_cp(uri: str, dest: Path) -> bool:
    try:
        subprocess.check_call(
            ["gsutil", "-q", "cp", uri, str(dest)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


def _find_gcs_uri(name: str) -> str | None:
    for base in GCS_SHARDS:
        uri = f"{base}/{name}"
        try:
            subprocess.check_call(
                ["gsutil", "-q", "stat", uri],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return uri
        except Exception:
            continue
    return None


def _closest_approach(path: Path) -> dict:
    """Time and node pair of global min inter-thread distance."""
    with h5py.File(path, "r") as f:
        t = np.asarray(f["t"][:], dtype=float)
        X = np.asarray(f["x"][:], dtype=float)
        N = int(f["params"].attrs["N"])
        N_t = int(f["params"].attrs["N_t"])
        r = float(f["params"].attrs.get("r", 300e-9))
        factor = float(f["params"].attrs.get("entangle_contact_factor", 2.0))
        entangled_end = bool(f["outcome"].attrs.get("entangled", False))
    P = Params(N=N, N_t=N_t)
    topo = Topology(P)
    dmin, k_best, na_best, nb_best = np.inf, -1, -1, -1
    if topo.N >= 2:
        for ja in range(topo.N):
            na = np.asarray(topo.thread_nodes[ja][2:], dtype=int)
            if na.size == 0:
                continue
            for jb in range(ja + 1, topo.N):
                nb = np.asarray(topo.thread_nodes[jb][2:], dtype=int)
                if nb.size == 0:
                    continue
                A = X[:, na, :]
                B = X[:, nb, :]
                d = np.linalg.norm(A[:, :, None, :] - B[:, None, :, :], axis=-1)
                # d: (T, na, nb)
                flat = d.reshape(d.shape[0], -1)
                idx = np.argmin(flat, axis=1)
                drow = flat[np.arange(flat.shape[0]), idx]
                k = int(np.argmin(drow))
                if drow[k] < dmin:
                    ia = int(na[idx[k] // nb.size])
                    ib = int(nb[idx[k] % nb.size])
                    dmin, k_best, na_best, nb_best = float(drow[k]), k, ia, ib
    return {
        "dmin_m": dmin,
        "dmin_um": dmin * 1e6,
        "t_closest": float(t[k_best]) if k_best >= 0 else float("nan"),
        "frame": int(k_best),
        "node_a": na_best,
        "node_b": nb_best,
        "threshold_m": factor * r,
        "entangled_end": entangled_end,
        "t_end": float(t[-1]),
    }


def do_entangle() -> dict:
    print("=== 2. entanglement audit ===", flush=True)
    rows = []
    with open(RESULTS / "sweep.csv", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("valid", "true")).lower() not in ("true", "1", "yes"):
                continue
            try:
                dmin = float(r["dmin_min_um"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(dmin):
                continue
            rows.append(r)
    below = [r for r in rows if float(r["dmin_min_um"]) < CONTACT_UM]
    n_ent = sum(1 for r in rows if str(r.get("entangled", "")).lower() == "true")
    print(f"  valid runs with finite dmin: {len(rows)}", flush=True)
    print(f"  dmin_min < {CONTACT_UM} um: {len(below)}", flush=True)
    print(f"  entangled=True among valid: {n_ent}", flush=True)
    print("  criterion: entangled iff d_min(FINAL state) < "
          "entangle_contact_factor * r (default 2r = 0.6 um). "
          "Checked only at exit, not along the trajectory. "
          "dmin_min_um is the min over saved frames.", flush=True)

    all_sorted = sorted(rows, key=lambda r: float(r["dmin_min_um"]))
    top5 = all_sorted[:5]
    print(f"  five smallest dmin (um): "
          f"{[float(r['dmin_min_um']) for r in top5]}", flush=True)
    details = []
    with tempfile.TemporaryDirectory(prefix="ent_") as td:
        td = Path(td)
        for r in top5:
            h5name = Path(r["hdf5_path"]).name
            uri = _find_gcs_uri(h5name)
            local = td / h5name
            if uri is None or not _gsutil_cp(uri, local):
                details.append({
                    "N": int(r["N"]), "sigma_w": float(r["sigma_w"]),
                    "x": int(float(r["x"])), "dmin_min_um": float(r["dmin_min_um"]),
                    "error": f"download failed for {h5name}",
                })
                continue
            ca = _closest_approach(local)
            rec = {
                "N": int(r["N"]), "sigma_w": float(r["sigma_w"]),
                "x": int(float(r["x"])), "seed": int(r["seed"]),
                "dmin_min_um_csv": float(r["dmin_min_um"]),
                "dmin_um_recomputed": ca["dmin_um"],
                "t_closest": ca["t_closest"],
                "node_a": ca["node_a"], "node_b": ca["node_b"],
                "entangled_end": ca["entangled_end"],
                "t_end": ca["t_end"],
            }
            details.append(rec)
            print(f"  closest: N={rec['N']} sw={rec['sigma_w']} x={rec['x']} "
                  f"dmin={rec['dmin_um_recomputed']:.4f} um "
                  f"t={rec['t_closest']:.4f}s nodes=({rec['node_a']},{rec['node_b']}) "
                  f"entangled_end={rec['entangled_end']}", flush=True)
    return {
        "n_valid_finite_dmin": len(rows),
        "n_dmin_below_2r": len(below),
        "n_entangled_true": n_ent,
        "criterion": (
            "entangled iff min_interthread_distance(final_xi) < "
            "entangle_contact_factor * r (default factor=2 → 2r=0.6 um). "
            "Evaluated only at simulation exit in integrator._entangled; "
            "dmin_min_um is the trajectory minimum over output frames."
        ),
        "five_smallest": details,
    }


# ---------------------------------------------------------------------------
# 4. Opening baseline
# ---------------------------------------------------------------------------
def _opening_one(args: tuple) -> dict | None:
    name, uri0, uri1, N, sigma_w, x, R_run = args
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / name
        ok = _gsutil_cp(uri0, local) or _gsutil_cp(uri1, local)
        if not ok:
            return None
        with h5py.File(local, "r") as f:
            X0 = np.asarray(f["x"][0], dtype=float)
            N_t = int(f["params"].attrs["N_t"])
            L = float(f["params"].attrs.get("L", 0.5))
        P = Params(N=N, N_t=N_t, L=L)
        topo = Topology(P)
        R0 = observables.tip_radius(topo, X0) / L
        return {"N": N, "sigma_w": sigma_w, "x": x, "R0_over_L": R0,
                "R_over_L_mean_run": R_run}


def do_opening(workers: int) -> list[dict]:
    print("=== 4. opening baseline R/L at frame 0 ===", flush=True)
    targets = [(2, 0.30, 0), (4, 0.30, 0), (8, 0.30, 0)]
    # index sweep rows
    by_file = {}
    with open(RESULTS / "sweep.csv", newline="") as f:
        for r in csv.DictReader(f):
            N, sw, x = int(r["N"]), float(r["sigma_w"]), int(float(r["x"]))
            if not any(N == tn and abs(sw - tsw) < 1e-12 and x == tx
                       for tn, tsw, tx in targets):
                continue
            name = Path(r["hdf5_path"]).name
            by_file[name] = float(r["R_over_L_mean"])

    jobs = []
    for N, sw, x in targets:
        for i in range(SB.realizations_for(x, "main")):
            name = f"N{N}_sw{sw:.2f}_x{x:+d}_i{i:04d}.h5"
            if name not in by_file:
                continue
            jobs.append((
                name,
                f"{GCS_SHARDS[0]}/{name}",
                f"{GCS_SHARDS[1]}/{name}",
                N, sw, x, by_file[name],
            ))

    print(f"  jobs={len(jobs)} workers={workers}", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_opening_one, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  opening {done}/{len(jobs)}", flush=True)
            if rec:
                results.append(rec)

    summary = []
    for N, sw, x in targets:
        grp = [r for r in results
               if r["N"] == N and abs(r["sigma_w"] - sw) < 1e-12 and r["x"] == x]
        if not grp:
            continue
        R0 = np.array([r["R0_over_L"] for r in grp])
        Rr = np.array([r["R_over_L_mean_run"] for r in grp])
        rec = {
            "N": N, "sigma_w": sw, "x": x, "M": len(grp),
            "R0_over_L_mean": float(np.mean(R0)),
            "R0_over_L_std": float(np.std(R0)),
            "R_over_L_run_mean": float(np.mean(Rr)),
            "R_over_L_run_std": float(np.std(Rr)),
        }
        summary.append(rec)
        print(f"  N={N} sw={sw} x={x} M={rec['M']}: "
              f"R0/L={rec['R0_over_L_mean']:.6e}±{rec['R0_over_L_std']:.3e}  "
              f"run R/L={rec['R_over_L_run_mean']:.6e}±{rec['R_over_L_run_std']:.3e}",
              flush=True)
    _write_csv(RESULTS / "opening_baseline.csv",
               ["N", "sigma_w", "x", "M", "R0_over_L_mean", "R0_over_L_std",
                "R_over_L_run_mean", "R_over_L_run_std"],
               summary)
    return summary


# ---------------------------------------------------------------------------
# 5. Tracer study
# ---------------------------------------------------------------------------
def _make_ks(sigma_w: float, seed: int) -> KinematicSimulation:
    return KinematicSimulation(
        sigma=sigma_w, ell=SB.SWEEP_ELL, U_h=SB.SWEEP_U_H,
        N_k=SB.SWEEP_N_K, seed=seed, L=0.5, N_t=50,
        lambda_=0.5, renormalize=True, mesh_limited=False,
    )


def _tracer_path_w(args: tuple) -> np.ndarray:
    """Return w(t) along a 60 s Lagrangian path (Euler, dt=1e-3)."""
    sigma_w, seed, t_end, dt = args
    ks = _make_ks(sigma_w, seed)
    n = int(round(t_end / dt))
    w = np.empty(n + 1, dtype=float)
    x = np.array([0.0, 0.0, SB.SWEEP_Z0])
    t = 0.0
    for i in range(n + 1):
        u = ks.u(x, t)
        w[i] = float(u[2])
        if i < n:
            x = x + dt * u
            t += dt
    return w


def _acf_mean(series_list: list[np.ndarray], max_lag: int) -> np.ndarray:
    """Mean normalized ACF over paths; R[0]=1."""
    acc = np.zeros(max_lag + 1, dtype=float)
    n_used = 0
    for w in series_list:
        w = w - np.mean(w)
        var = float(np.dot(w, w) / len(w))
        if var < 1e-30:
            continue
        # full correlation via FFT
        n = len(w)
        nfft = 1 << (2 * n - 1).bit_length()
        f = np.fft.rfft(w, n=nfft)
        acf = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n].real
        acf /= acf[0]
        L = min(max_lag, n - 1)
        acc[: L + 1] += acf[: L + 1]
        n_used += 1
    if n_used == 0:
        return np.full(max_lag + 1, np.nan)
    return acc / n_used


def _integrate_TL(R: np.ndarray, dt: float, to_zero: bool, tmax: float | None):
    """Trapezoid integral of R(tau)."""
    if to_zero:
        # first zero crossing
        end = len(R)
        for i in range(1, len(R)):
            if R[i] <= 0.0:
                end = i
                break
        R = R[:end]
    elif tmax is not None:
        n = int(round(tmax / dt)) + 1
        R = R[: min(n, len(R))]
    if len(R) < 2:
        return float("nan")
    tau = np.arange(len(R)) * dt
    return float(np.trapezoid(R, tau))


def _tracer_drift_one(args: tuple) -> str:
    """Integrate dz/dt = w + U0 until exit. Return outcome."""
    sigma_w, seed, U0, z0, h, t_end, dt = args
    ks = _make_ks(sigma_w, seed)
    x = np.array([0.0, 0.0, float(z0)])
    t = 0.0
    while t < t_end - 0.5 * dt:
        u = ks.u(x, t)
        # RK4 with imposed vertical drift on top of turbulent w
        def rhs(xx, tt):
            uu = ks.u(xx, tt)
            return np.array([uu[0], uu[1], uu[2] + U0])
        k1 = rhs(x, t)
        k2 = rhs(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = rhs(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = rhs(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if x[2] >= h:
            return "rise"
        if x[2] <= 0.0:
            return "fall"
    return "timeout"


def _fit_pdd(xs: np.ndarray, Ps: np.ndarray) -> tuple[float, float]:
    """Fit P(x)=(exp(-a x p0)-1)/(exp(-a x)-1); free (a, p0)."""
    from scipy.optimize import curve_fit

    def model(x, a, p0):
        out = np.empty_like(x, dtype=float)
        for i, xi in enumerate(x):
            if abs(xi) < 1e-12 or abs(a * xi) < 1e-12:
                out[i] = p0
            else:
                out[i] = (math.exp(-a * xi * p0) - 1.0) / (math.exp(-a * xi) - 1.0)
        return out

    popt, _ = curve_fit(model, xs, Ps, p0=[1.0, SB.SWEEP_Z0 / SB.SWEEP_H],
                        bounds=([0.01, 0.01], [10.0, 0.99]), maxfev=5000)
    return float(popt[0]), float(popt[1])


def do_tracer(workers: int, M: int = 2000) -> dict:
    print("=== 5. tracer study ===", flush=True)
    dt = 1e-3
    t_end_path = 60.0
    max_lag = int(round(20.0 / dt))
    tl_rows = []
    for sigma_w in (0.15, 0.30):
        print(f"  (a) Lagrangian ACF sigma_w={sigma_w} M={M}", flush=True)
        seeds = [int(SB.sweep_seed(1, sigma_w, 0, i)) for i in range(M)]
        args = [(sigma_w, s, t_end_path, dt) for s in seeds]
        series = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_tracer_path_w, a) for a in args]
            done = 0
            for fut in as_completed(futs):
                series.append(fut.result())
                done += 1
                if done % 200 == 0:
                    print(f"    paths {done}/{M}", flush=True)
        R = _acf_mean(series, max_lag)
        TL_zero = _integrate_TL(R, dt, to_zero=True, tmax=None)
        TL_20 = _integrate_TL(R, dt, to_zero=False, tmax=20.0)
        for label, TL in (("first_zero", TL_zero), ("to_20s", TL_20)):
            KL = sigma_w ** 2 * TL
            a = (sigma_w * SB.SWEEP_ELL) / KL if KL > 0 else float("nan")
            tl_rows.append({
                "sigma_w": sigma_w, "ell": SB.SWEEP_ELL, "M": M,
                "integration": label, "T_L": TL, "K_L": KL, "a": a,
            })
            print(f"    {label}: T_L={TL:.6f} K_L={KL:.6f} a={a:.4f}", flush=True)

    _write_csv(RESULTS / "tracer_TL.csv",
               ["sigma_w", "ell", "M", "integration", "T_L", "K_L", "a"],
               tl_rows)

    phase_rows = []
    fit_rows = []
    for sigma_w in (0.15, 0.30):
        K = SB.eddy_diffusivity(sigma_w)
        print(f"  (b) drift tracers sigma_w={sigma_w} K={K}", flush=True)
        xs, Ps, P_los, P_his = [], [], [], []
        for x in SB.SWEEP_X:
            U0 = float(x) * K / SB.SWEEP_H
            seeds = [int(SB.sweep_seed(1, sigma_w, int(x), i)) for i in range(M)]
            args = [(sigma_w, s, U0, SB.SWEEP_Z0, SB.SWEEP_H, SB.SWEEP_T_END, dt)
                    for s in seeds]
            outcomes = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_tracer_drift_one, a) for a in args]
                for fut in as_completed(futs):
                    outcomes.append(fut.result())
            n_up = sum(1 for o in outcomes if o == "rise")
            n_down = sum(1 for o in outcomes if o == "fall")
            n_to = sum(1 for o in outcomes if o == "timeout")
            P, Plo, Phi = SB.wilson_ci(n_up, n_up + n_down)
            Pdd = SB.P_dd(U0, K)
            phase_rows.append({
                "sigma_w": sigma_w, "x": int(x), "U0": U0, "K": K,
                "M": M, "n_up": n_up, "n_down": n_down, "n_timeout": n_to,
                "P": P, "P_lo": Plo, "P_hi": Phi, "P_dd": Pdd,
            })
            print(f"    x={x:+d} U0={U0:.4f} P={P:.4f} [{Plo:.4f},{Phi:.4f}] "
                  f"P_dd={Pdd:.4f} up={n_up} down={n_down} to={n_to}", flush=True)
            xs.append(float(x))
            Ps.append(P)
            P_los.append(Plo)
            P_his.append(Phi)
        a_fit, p0_fit = _fit_pdd(np.asarray(xs), np.asarray(Ps))
        fit_rows.append({"sigma_w": sigma_w, "a": a_fit, "p0": p0_fit,
                         "a_theory": 1.0, "p0_theory": SB.SWEEP_Z0 / SB.SWEEP_H})
        print(f"    fit (a,p0)=({a_fit:.4f},{p0_fit:.4f})  "
              f"theory (1, {SB.SWEEP_Z0 / SB.SWEEP_H})", flush=True)

    _write_csv(RESULTS / "tracer_phase.csv",
               ["sigma_w", "x", "U0", "K", "M", "n_up", "n_down", "n_timeout",
                "P", "P_lo", "P_hi", "P_dd"],
               phase_rows)
    return {"TL": tl_rows, "phase": phase_rows, "fit": fit_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tab-wc", action="store_true")
    ap.add_argument("--wall-time", action="store_true")
    ap.add_argument("--entangle", action="store_true")
    ap.add_argument("--opening", action="store_true")
    ap.add_argument("--tracer", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--tracer-M", type=int, default=2000)
    ns = ap.parse_args()
    if ns.all or not any([ns.tab_wc, ns.wall_time, ns.entangle, ns.opening, ns.tracer]):
        ns.tab_wc = ns.wall_time = ns.entangle = ns.opening = ns.tracer = True

    report: dict = {}
    t0 = time.perf_counter()
    if ns.tab_wc:
        report["tab_wc_m10"] = do_tab_wc(ns.workers)
    if ns.wall_time:
        report["wall_time"] = do_wall_time()
    if ns.entangle:
        report["entangle"] = do_entangle()
    if ns.opening:
        report["opening"] = do_opening(ns.workers)
    if ns.tracer:
        report["tracer"] = do_tracer(ns.workers, M=ns.tracer_M)

    # short text report
    lines = ["Stage B Mac checks report", f"elapsed_s={time.perf_counter() - t0:.1f}", ""]
    if "tab_wc_m10" in report:
        lines.append("## tab_wc m=10 mg (t_end=40)")
        for r in report["tab_wc_m10"]:
            lines.append(
                f"N={r['N']} q={r['q_nC']}: wc_num={r['wc_numeric']:.6e} "
                f"analytic={r['wc_analytic']:.6e} err={r['rel_err_pct']:.3f}% "
                f"V(t_end)={r['V_end']:.6e} |dV/dt|={r['abs_dVdt_end']:.3e} "
                f"status={r['status']}"
            )
        lines.append("")
    if "wall_time" in report:
        w = report["wall_time"]
        lines.append(f"## wall_time_s: filled={w['filled']} NaN={w['nan']} "
                     f"(of {w['total']})")
        lines.append("")
    if "entangle" in report:
        e = report["entangle"]
        lines.append("## entanglement")
        lines.append(e["criterion"])
        lines.append(f"valid with dmin_min < 0.6 um: {e['n_dmin_below_2r']} / "
                     f"{e['n_valid_finite_dmin']}")
        lines.append(f"entangled=True count: {e['n_entangled_true']}")
        for d in e["five_smallest"]:
            lines.append(str(d))
        lines.append("")
    if "opening" in report:
        lines.append("## opening baseline")
        for r in report["opening"]:
            lines.append(str(r))
        lines.append("")
    if "tracer" in report:
        lines.append("## tracer")
        for r in report["tracer"]["TL"]:
            lines.append(str(r))
        for r in report["tracer"]["fit"]:
            lines.append(f"fit sigma_w={r['sigma_w']}: a={r['a']:.4f} p0={r['p0']:.4f} "
                         f"(theory a={r['a_theory']} p0={r['p0_theory']})")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (RESULTS / "stage_b_checks_report.txt").write_text(text)
    # JSON: drop huge arrays
    with open(RESULTS / "stage_b_checks_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
