"""Stage B: production sweep (Algorithm alg:sweep).

Additive. Does not modify scripts/produce_results.py or Stage A defaults.

Usage (from Tema 1/):
  python scripts/stage_b.py --ksvalid              # renormalized fig_ksvalid (N_k=200)
  python scripts/stage_b.py --tab-wc-m10           # rerun m=10 mg rows, t_end=40
  python scripts/stage_b.py --smoke                # tiny local pipeline check
  python scripts/stage_b.py --cost-probe           # 1 grid point / N, M=5; print ETA
  python scripts/stage_b.py --sweep                # full 11200-run grid (resume-safe)

Do not launch --sweep / --cost-probe until confirmed. Cloud launch TBD.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ballooning.io_hdf5 import git_commit_hash
from ballooning import studies
from ballooning import sweep as SB

RESULTS = ROOT / "results"
SWEEP_CSV = RESULTS / "sweep.csv"
PHASE_CSV = RESULTS / "tab_phase.csv"
LOG_PATH = RESULTS / "sweep_log.txt"
ML_DIR = RESULTS / "ml_samples"
SNAP_DIR = RESULTS / "snapshots"

SWEEP_HEADER = [
    "N", "sigma_w", "ell", "x", "Fbar_l", "q_nC", "seed", "outcome",
    "exit_time", "R_over_L_mean", "R_over_L_std", "theta_L_mean",
    "dmin_min_um", "entangled", "newton_failures", "wall_time_s",
]


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


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    print(f"wrote {path.name} ({len(rows)} rows)", flush=True)


def _append_sweep_row(row: dict) -> None:
    """Append one completed realization to sweep.csv (create with header if needed)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    new_file = not SWEEP_CSV.exists()
    with open(SWEEP_CSV, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new_file:
            w.writerow(SWEEP_HEADER)
        w.writerow([_fmt(row[k]) for k in SWEEP_HEADER])


def _load_done_keys() -> set[tuple]:
    """Keys (N, sigma_w, x, seed) already present in sweep.csv."""
    done: set[tuple] = set()
    if not SWEEP_CSV.exists():
        return done
    with open(SWEEP_CSV, newline="") as f:
        for r in csv.DictReader(f):
            done.add((int(r["N"]), float(r["sigma_w"]), int(float(r["x"])),
                      int(r["seed"])))
    return done


def _load_sweep_rows() -> list[dict]:
    if not SWEEP_CSV.exists():
        return []
    with open(SWEEP_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _log(msg: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _load_meta() -> dict:
    p = RESULTS / "meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict) -> None:
    with open(RESULTS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Stage B item 1: renormalized KS validation (N_k = 200 only)
# ---------------------------------------------------------------------------
def do_ksvalid() -> None:
    print("=== Stage B KS validation (N_k=200, turb_renormalize=True) ===",
          flush=True)
    res = studies.run_ks_validation(N_k_values=(200,), renormalize=True)
    rows = []
    for spec in res["spectra"]:
        for k, et, em in zip(spec["k"], spec["E_target"], spec["E_measured"]):
            rows.append([spec["N_k"], k, et, em])
    _write_csv(RESULTS / "fig_ksvalid.csv",
               ["N_k", "k", "E_target", "E_measured"], rows)
    rows = []
    for pdf in res["pdfs"]:
        for c, p, g in zip(pdf["w_bin_center"], pdf["pdf"], pdf["gaussian"]):
            rows.append([pdf["N_k"], c, p, g])
    _write_csv(RESULTS / "fig_ksvalid_pdf.csv",
               ["N_k", "w_bin_center", "pdf", "gaussian"], rows)
    meta = _load_meta()
    sa = meta.setdefault("stage_b", {})
    sa["ks_validation"] = {
        "N_k": 200,
        "renormalize": True,
        "renorm_factor": res["renorm_factor"].get(200),
        "energy_fraction": res["energy_fraction"].get(200),
        "sigma": res["sigma"],
        "ell": res["ell"],
    }
    _save_meta(meta)
    print(f"renorm_factor={res['renorm_factor'].get(200)}", flush=True)


# ---------------------------------------------------------------------------
# Stage B item 2: tab_wc m=10 mg, t_end=40
# ---------------------------------------------------------------------------
def do_tab_wc_m10(workers: int) -> None:
    print("=== Stage B tab_wc: m=10 mg, t_end=40 s ===", flush=True)
    specs = [(N, 10.0, q) for N in studies.WC_N for q in studies.WC_Q_NC]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        new_rows = list(pool.map(studies.run_wc_job, specs))
    # Merge: keep non-m=10 rows, replace m=10.
    existing = []
    path = RESULTS / "tab_wc.csv"
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if abs(float(r["m_mg"]) - 10.0) > 1e-12:
                    existing.append(r)
    by_key = {(int(r["N"]), float(r["m_mg"]), float(r["q_nC"])): r
              for r in existing}
    for r in new_rows:
        by_key[(r["N"], r["m_mg"], r["q_nC"])] = r
        print(f"  N={r['N']} m=10 q={r['q_nC']} status={r['status']} "
              f"wc_num={r['wc_numeric']:.4e} err={r['rel_err_pct']:.2f}%",
              flush=True)
    ordered = sorted(by_key.values(),
                     key=lambda r: (int(r["N"]) if not isinstance(r["N"], int)
                                    else r["N"],
                                    float(r["m_mg"]), float(r["q_nC"])))
    rows = [[r["N"], r["m_mg"], r["q_nC"], r["wc_numeric"],
             r["wc_analytic"], r["rel_err_pct"]] for r in ordered]
    _write_csv(path,
               ["N", "m_mg", "q_nC", "wc_numeric", "wc_analytic", "rel_err_pct"],
               rows)


# ---------------------------------------------------------------------------
# Sweep runner (resume-safe, progress every 10 min)
# ---------------------------------------------------------------------------
def _pending_specs(M: int, N_values, N_t: int, t_end: float,
                   write_hdf5: bool, smoke: bool) -> list[dict]:
    done = _load_done_keys()
    specs = []
    for pt in SB.iter_grid_points(M=M, N_values=N_values):
        key = (pt.N, pt.sigma_w, pt.x, pt.seed)
        if key in done:
            continue
        snap = SB.is_snapshot_point(pt)
        specs.append({
            "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
            "N_t": N_t, "t_end": t_end,
            "output_dt": (SB.SWEEP_SNAPSHOT_DT if snap
                          else SB.SWEEP_OUTPUT_DT),
            "write_hdf5": write_hdf5 and not smoke,
            "hdf5_dir": str(ML_DIR),
            "snapshot": snap,
            "snapshot_dir": str(SNAP_DIR),
        })
    return specs


def do_sweep(workers: int, M: int, N_values, N_t: int, t_end: float,
             write_hdf5: bool, smoke: bool = False) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    if write_hdf5:
        ML_DIR.mkdir(parents=True, exist_ok=True)

    specs = _pending_specs(M, N_values, N_t, t_end, write_hdf5, smoke)
    total_grid = len(SB.iter_grid_points(M=M, N_values=N_values))
    n_done0 = total_grid - len(specs)
    _log(f"sweep start: pending={len(specs)} already_done={n_done0} "
         f"total={total_grid} workers={workers} M={M} N_t={N_t} t_end={t_end}")

    if not specs:
        _log("nothing pending; rebuilding tab_phase.csv")
        _write_phase()
        return

    t_start = time.perf_counter()
    last_progress = t_start
    n_finished = 0
    wall_acc = 0.0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(SB.run_sweep_job, s): s for s in specs}
        for fut in as_completed(futures):
            row = fut.result()
            _append_sweep_row(row)
            n_finished += 1
            wall_acc += float(row["wall_time_s"])
            now = time.perf_counter()
            if now - last_progress >= 600.0 or n_finished == len(specs):
                elapsed = now - t_start
                rate = n_finished / elapsed if elapsed > 0 else 0.0
                remain = len(specs) - n_finished
                eta = remain / rate if rate > 0 else float("inf")
                _log(f"progress {n_done0 + n_finished}/{total_grid} "
                     f"(+{n_finished} this session) "
                     f"elapsed={elapsed / 3600:.2f}h ETA={eta / 3600:.2f}h "
                     f"last={row['outcome']} N={row['N']} x={row['x']} "
                     f"i={row.get('i', '?')}")
                last_progress = now

    _write_phase()
    ml_size = _ml_total_bytes() if write_hdf5 else 0
    _log(f"sweep session done: finished={n_finished} "
         f"cpu_wall_sum={wall_acc / 3600:.2f}h ml_bytes={ml_size}")
    meta = _load_meta()
    meta.setdefault("stage_b", {})["sweep"] = {
        "git_commit": git_commit_hash(),
        "M": M,
        "N_values": list(N_values),
        "total_grid": total_grid,
        "ml_bytes": ml_size,
    }
    _save_meta(meta)
    # Marker for cloud watchers (GCP scripts poll this file).
    done_path = RESULTS / "sweep_DONE"
    done_path.write_text(
        f"finished={n_finished}\ntotal={total_grid}\n"
        f"git={git_commit_hash()}\nml_bytes={ml_size}\n",
        encoding="utf-8",
    )
    _log(f"wrote {done_path.name}")


def _write_phase() -> None:
    rows = _load_sweep_rows()
    if not rows:
        return
    # Normalise types from CSV strings.
    norm = []
    for r in rows:
        norm.append({
            "N": int(r["N"]),
            "sigma_w": float(r["sigma_w"]),
            "x": int(float(r["x"])),
            "Fbar_l": float(r["Fbar_l"]),
            "outcome": r["outcome"],
            "exit_time": float(r["exit_time"]),
        })
    phase = SB.aggregate_phase(norm)
    _write_csv(PHASE_CSV,
               ["N", "sigma_w", "x", "Fbar_l", "M", "n_up", "n_down",
                "n_timeout", "P", "P_lo", "P_hi", "P_dd", "mean_exit_time"],
               [[p[k] for k in ("N", "sigma_w", "x", "Fbar_l", "M", "n_up",
                                "n_down", "n_timeout", "P", "P_lo", "P_hi",
                                "P_dd", "mean_exit_time")] for p in phase])


def _ml_total_bytes() -> int:
    if not ML_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in ML_DIR.glob("*.h5"))


# ---------------------------------------------------------------------------
# Cost probe: 1 grid point per N, M=5
# ---------------------------------------------------------------------------
def do_cost_probe(workers: int) -> None:
    """1 grid point per N (sigma_w=0.15, x=0), M=5; print ETA for full grid."""
    print("=== Stage B cost probe (1 point/N, M=5) ===", flush=True)
    # One point per N: sigma_w=0.15, x=0, i=0..4
    specs = []
    for N in SB.SWEEP_N:
        for i in range(5):
            pt = SB.SweepPoint(N=N, sigma_w=0.15, x=0, i=i)
            specs.append({
                "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
                "N_t": 100, "t_end": SB.SWEEP_T_END,
                "output_dt": SB.SWEEP_OUTPUT_DT,
                "write_hdf5": False, "hdf5_dir": str(ML_DIR),
                "snapshot": False, "snapshot_dir": str(SNAP_DIR),
            })
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        rows = list(pool.map(SB.run_sweep_job, specs))
    wall = time.perf_counter() - t0

    by_n: dict[int, list[float]] = {n: [] for n in SB.SWEEP_N}
    for r in rows:
        by_n[int(r["N"])].append(float(r["wall_time_s"]))
        print(f"  N={r['N']} i={r.get('i', '?')} {r['outcome']} "
              f"exit={r['exit_time']:.3f}s wall={r['wall_time_s']:.1f}s",
              flush=True)

    # Full grid: 2 sigma × 7 x × M=200 = 2800 jobs per N
    jobs_per_N = len(SB.SWEEP_SIGMA_W) * len(SB.SWEEP_X) * SB.SWEEP_M_REALIZATIONS
    mean_wall = {n: float(np.mean(v)) for n, v in by_n.items()}
    # Makespan estimate: greedy pack on `workers` cores, using mean wall/N.
    durs = []
    for n in SB.SWEEP_N:
        durs.extend([mean_wall[n]] * jobs_per_N)
    load = [0.0] * workers
    for d in sorted(durs, reverse=True):
        i = load.index(min(load))
        load[i] += d
    eta = max(load)
    total_cpu = sum(durs)
    print(f"\nCost probe wall (20 jobs): {wall / 60:.1f} min", flush=True)
    print(f"Mean wall/run: " + ", ".join(
        f"N={n}:{mean_wall[n]:.1f}s" for n in SB.SWEEP_N), flush=True)
    print(f"Full grid estimate ({SB.SWEEP_M_REALIZATIONS}×{len(SB.SWEEP_N)}×"
          f"{len(SB.SWEEP_SIGMA_W)}×{len(SB.SWEEP_X)} = "
          f"{len(SB.SWEEP_N)*jobs_per_N} runs, {workers} workers):", flush=True)
    print(f"  total CPU ≈ {total_cpu / 3600:.1f} h", flush=True)
    print(f"  wall ETA  ≈ {eta / 3600:.1f} h", flush=True)
    print("Waiting for confirmation before launching --sweep.", flush=True)
    meta = _load_meta()
    meta.setdefault("stage_b", {})["cost_probe"] = {
        "mean_wall_s": mean_wall,
        "jobs_per_N": jobs_per_N,
        "eta_wall_h": eta / 3600,
        "total_cpu_h": total_cpu / 3600,
        "workers": workers,
    }
    _save_meta(meta)
    (RESULTS / "cost_probe_DONE").write_text(
        f"eta_wall_h={eta / 3600:.4f}\ntotal_cpu_h={total_cpu / 3600:.4f}\n"
        f"workers={workers}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Smoke: tiny pipeline (short t_end, small N_t, M=1)
# ---------------------------------------------------------------------------
def do_smoke(workers: int) -> None:
    print("=== Stage B SMOKE ===", flush=True)
    # Seed + Fbar + P_dd unit checks
    s = SB.sweep_seed(1, 0.15, 0, 0)
    s2 = SB.sweep_seed(1, 0.15, 0, 0)
    assert s == s2
    assert SB.sweep_seed(1, 0.15, 0, 1) != s
    P, lo, hi = SB.wilson_ci(50, 100)
    assert abs(P - 0.5) < 1e-12 and lo < P < hi
    assert abs(SB.P_dd(0.0, 1.0) - SB.SWEEP_Z0 / SB.SWEEP_H) < 1e-12
    print(f"  seed hash ok ({s}); Wilson/P_dd ok", flush=True)

    # One short realization per N
    smoke_csv = RESULTS / "sweep_smoke.csv"
    if smoke_csv.exists():
        smoke_csv.unlink()
    # Temporarily redirect SWEEP_CSV? Keep smoke separate: run jobs and write smoke csv.
    rows = []
    specs = []
    for N in SB.SWEEP_N:
        pt = SB.SweepPoint(N=N, sigma_w=0.15, x=0, i=0)
        specs.append({
            "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
            "N_t": 20, "t_end": 0.2,
            "output_dt": 0.05,
            "write_hdf5": True, "hdf5_dir": str(RESULTS / "ml_samples_smoke"),
            "snapshot": (N == 4),
            "snapshot_dir": str(RESULTS / "snapshots_smoke"),
        })
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        rows = list(pool.map(SB.run_sweep_job, specs))
    with open(smoke_csv, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(SWEEP_HEADER)
        for r in rows:
            w.writerow([_fmt(r[k]) for k in SWEEP_HEADER])
            print(f"  smoke N={r['N']} {r['outcome']} wall={r['wall_time_s']:.2f}s "
                  f"nf={r['newton_failures']} renorm={r.get('renorm_factor')}",
                  flush=True)
    phase = SB.aggregate_phase(rows)
    assert len(phase) == 4
    print(f"  wrote {smoke_csv.name}; phase rows={len(phase)}", flush=True)
    print("SMOKE OK", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ksvalid", action="store_true")
    ap.add_argument("--tab-wc-m10", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cost-probe", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="full production grid (resume-safe); wait for confirmation")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--M", type=int, default=SB.SWEEP_M_REALIZATIONS)
    ap.add_argument("--N", type=int, nargs="+", default=list(SB.SWEEP_N))
    ap.add_argument("--t-end", type=float, default=SB.SWEEP_T_END)
    ap.add_argument("--N-t", type=int, default=100)
    ap.add_argument("--no-hdf5", action="store_true")
    ns = ap.parse_args()

    if not any([ns.ksvalid, ns.tab_wc_m10, ns.smoke, ns.cost_probe, ns.sweep]):
        ap.error("select --ksvalid / --tab-wc-m10 / --smoke / --cost-probe / --sweep")

    workers = ns.workers or (os.cpu_count() or 4)
    print(f"stage_b: workers={workers} git={git_commit_hash()[:10]}", flush=True)

    if ns.ksvalid:
        do_ksvalid()
    if ns.tab_wc_m10:
        do_tab_wc_m10(workers)
    if ns.smoke:
        do_smoke(workers)
    if ns.cost_probe:
        do_cost_probe(workers)
    if ns.sweep:
        do_sweep(workers, M=ns.M, N_values=ns.N, N_t=ns.N_t, t_end=ns.t_end,
                 write_hdf5=not ns.no_hdf5, smoke=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
