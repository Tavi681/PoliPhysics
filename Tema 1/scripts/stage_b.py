"""Stage B: production sweep (Algorithm alg:sweep) + B0 harness.

Additive. Does not modify scripts/produce_results.py or Stage A defaults.

Usage (from Tema 1/):
  python scripts/stage_b.py --ksvalid              # renormalized fig_ksvalid (N_k=200)
  python scripts/stage_b.py --tab-wc-m10           # rerun m=10 mg rows, t_end=40
  python scripts/stage_b.py --smoke                # tiny local pipeline check
  python scripts/stage_b.py --cost-probe           # 1 grid point / N, M=5; print ETA
  python scripts/stage_b.py --b0                   # T1–T6 + scaled ETA (cloud)
  python scripts/stage_b.py --calibrate            # 2 short runs / N on a new machine
  python scripts/stage_b.py --sweep                # B1+B2 production grid (resume-safe)

Do not launch --sweep until R and B0 both PASS and are confirmed.
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

import h5py
import numpy as np

from ballooning.io_hdf5 import git_commit_hash, git_provenance
from ballooning import studies
from ballooning import sweep as SB
from ballooning.fields import KS_K_MAX, KinematicSimulation

RESULTS = ROOT / "results"
SWEEP_CSV = RESULTS / "sweep.csv"
PHASE_CSV = RESULTS / "tab_phase.csv"
LOG_PATH = RESULTS / "sweep_log.txt"
ML_DIR = RESULTS / "ml_samples"
SNAP_DIR = RESULTS / "snapshots"
B0_CONFIG = RESULTS / "b0_config.json"

# B3 sweep.csv columns (plus N_t / split / hdf5_path from the final spec).
SWEEP_HEADER = [
    "N", "N_t", "sigma_w", "ell", "x", "Fbar_l", "q_nC", "seed", "split",
    "outcome", "exit_time", "R_over_L_mean", "R_over_L_std", "theta_L_mean",
    "dmin_min_um", "entangled", "newton_failures", "wall_time_s", "hdf5_path",
]

# Stored cost-probe means (N_t=100, c2-standard-16). Do not re-probe.
COST_PROBE_MEAN_WALL_S = {
    1: 675.7149487728,
    2: 1754.4442739546,
    4: 2576.15476944,
    8: 6021.623079003801,
}


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


def _append_sweep_row(row: dict, csv_path: Path | None = None) -> None:
    """Append one completed realization (create with header if needed)."""
    path = csv_path or SWEEP_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new_file:
            w.writerow(SWEEP_HEADER)
        w.writerow([_fmt(row[k]) for k in SWEEP_HEADER])


def _row_done_key(r: dict) -> tuple:
    """Resume key (N, sigma_w, x, i). ``i`` recovered from seed when absent."""
    N = int(r["N"])
    sigma_w = float(r["sigma_w"])
    x = int(float(r["x"]))
    if "i" in r and r["i"] not in ("", None):
        return (N, sigma_w, x, int(r["i"]))
    seed = int(r["seed"])
    split = str(r.get("split", "main") or "main")
    M = SB.realizations_for(x, split)
    for i in range(M + 8):
        if SB.sweep_seed(N, sigma_w, x, i) == seed:
            return (N, sigma_w, x, i)
    return (N, sigma_w, x, seed)


def _load_done_keys(csv_path: Path | None = None) -> set[tuple]:
    """Keys (N, sigma_w, x, i) already present in sweep.csv."""
    path = csv_path or SWEEP_CSV
    done: set[tuple] = set()
    if not path.exists():
        return done
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            done.add(_row_done_key(r))
    return done


def _load_sweep_rows(csv_path: Path | None = None) -> list[dict]:
    path = csv_path or SWEEP_CSV
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _log(msg: str, log_path: Path | None = None) -> None:
    path = log_path or LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
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


def _cost_means() -> dict[int, float]:
    meta = _load_meta()
    stored = (meta.get("stage_b") or {}).get("cost_probe", {}).get("mean_wall_s")
    if stored:
        return {int(k): float(v) for k, v in stored.items()}
    return dict(COST_PROBE_MEAN_WALL_S)


# ---------------------------------------------------------------------------
# Stage B item 1: renormalized KS validation (N_k = 200 only)
# ---------------------------------------------------------------------------
def do_ksvalid() -> dict:
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
    return res


# ---------------------------------------------------------------------------
# Stage B item 2: tab_wc m=10 mg, t_end=40
# ---------------------------------------------------------------------------
def do_tab_wc_m10(workers: int) -> None:
    print("=== Stage B tab_wc: m=10 mg, t_end=40 s ===", flush=True)
    specs = [(N, 10.0, q) for N in studies.WC_N for q in studies.WC_Q_NC]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        new_rows = list(pool.map(studies.run_wc_job, specs))
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
# Sweep runner (resume-safe, sharded, progress every 10 min)
# ---------------------------------------------------------------------------
def _job_spec(pt: SB.SweepPoint, N_t: int, t_end: float, output_dt: float,
              dt_max: float, write_hdf5: bool, smoke: bool) -> dict:
    snap = SB.is_snapshot_point(pt)
    return {
        "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
        "split": pt.split,
        "N_t": N_t, "t_end": t_end, "output_dt": output_dt, "dt_max": dt_max,
        "write_hdf5": write_hdf5 and not smoke,
        "hdf5_dir": str(ML_DIR),
        "snapshot": snap,
        "snapshot_dir": str(SNAP_DIR),
    }


def _select_points(production: bool, M: int, N_values) -> list[SB.SweepPoint]:
    if production:
        return SB.iter_production_points()
    return SB.iter_grid_points(M=M, N_values=N_values)


def _pending_specs(points: list[SB.SweepPoint], done: set[tuple],
                   N_t: int, t_end: float, output_dt: float, dt_max: float,
                   write_hdf5: bool, smoke: bool) -> list[dict]:
    specs = []
    for pt in points:
        key = (pt.N, pt.sigma_w, pt.x, pt.i)
        if key in done:
            continue
        specs.append(_job_spec(pt, N_t, t_end, output_dt, dt_max,
                               write_hdf5, smoke))
    return specs


def sweep_csv_path(shard: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return SWEEP_CSV
    return RESULTS / f"sweep_shard_{shard:02d}.csv"


def do_sweep(workers: int, production: bool, M: int, N_values,
             N_t: int, t_end: float, output_dt: float, dt_max: float,
             write_hdf5: bool, shard: int = 0, num_shards: int = 1,
             smoke: bool = False, csv_path: Path | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    if write_hdf5:
        ML_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = csv_path or sweep_csv_path(shard, num_shards)
    points = _select_points(production, M, N_values)
    if num_shards > 1:
        assign = SB.shard_assignment(points, num_shards, _cost_means())
        points = [pt for pt, s in zip(points, assign) if s == shard]

    done = _load_done_keys(csv_path)
    specs = _pending_specs(points, done, N_t, t_end, output_dt, dt_max,
                           write_hdf5, smoke)
    total_grid = len(points)
    n_done0 = total_grid - len(specs)
    _log(f"sweep start: pending={len(specs)} already_done={n_done0} "
         f"total={total_grid} workers={workers} production={production} "
         f"M={M} N_t={N_t} t_end={t_end} dt_max={dt_max} "
         f"shard={shard}/{num_shards} csv={csv_path.name}")

    if not specs:
        _log("nothing pending; rebuilding tab_phase.csv")
        _write_phase(csv_path)
        return

    t_start = time.perf_counter()
    last_progress = t_start
    n_finished = 0
    wall_acc = 0.0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(SB.run_sweep_job, s): s for s in specs}
        for fut in as_completed(futures):
            row = fut.result()
            _append_sweep_row(row, csv_path)
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
                     f"i={row.get('i', '?')} shard={shard}/{num_shards}")
                last_progress = now

    _write_phase(csv_path)
    ml_size = _ml_total_bytes() if write_hdf5 else 0
    _log(f"sweep session done: finished={n_finished} "
         f"cpu_wall_sum={wall_acc / 3600:.2f}h ml_bytes={ml_size}")
    meta = _load_meta()
    meta.setdefault("stage_b", {})["sweep"] = {
        "git_commit": git_commit_hash(),
        "git_dirty": git_provenance()["dirty"],
        "production": production,
        "M": M,
        "N_values": list(N_values),
        "N_t": N_t,
        "dt_max": dt_max,
        "shard": shard,
        "num_shards": num_shards,
        "total_grid": total_grid,
        "ml_bytes": ml_size,
    }
    _save_meta(meta)
    done_path = RESULTS / "sweep_DONE"
    if num_shards > 1:
        done_path = RESULTS / f"sweep_DONE_shard_{shard:02d}"
    done_path.write_text(
        f"finished={n_finished}\ntotal={total_grid}\n"
        f"git={git_commit_hash()}\ndirty={git_provenance()['dirty']}\n"
        f"ml_bytes={ml_size}\n"
        f"shard={shard}\nnum_shards={num_shards}\n",
        encoding="utf-8",
    )
    _log(f"wrote {done_path.name}")


def _write_phase(csv_path: Path | None = None) -> None:
    rows = _load_sweep_rows(csv_path)
    if not rows:
        return
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
# Cost probe: 1 grid point per N, M=5  (kept; do not re-run for the estimate)
# ---------------------------------------------------------------------------
def do_cost_probe(workers: int) -> None:
    """1 grid point per N (sigma_w=0.15, x=0), M=5; print ETA for the old grid."""
    print("=== Stage B cost probe (1 point/N, M=5) ===", flush=True)
    specs = []
    for N in SB.SWEEP_N:
        for i in range(5):
            pt = SB.SweepPoint(N=N, sigma_w=0.15, x=0, i=i)
            specs.append({
                "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
                "N_t": 100, "t_end": SB.SWEEP_T_END,
                "output_dt": SB.SWEEP_OUTPUT_DT, "dt_max": SB.SWEEP_DT_MAX,
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

    jobs_per_N = len(SB.SWEEP_SIGMA_W) * len(SB.SWEEP_X) * SB.SWEEP_M_REALIZATIONS
    mean_wall = {n: float(np.mean(v)) for n, v in by_n.items()}
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
    s = SB.sweep_seed(1, 0.15, 0, 0)
    s2 = SB.sweep_seed(1, 0.15, 0, 0)
    assert s == s2
    assert SB.sweep_seed(1, 0.15, 0, 1) != s
    P, lo, hi = SB.wilson_ci(50, 100)
    assert abs(P - 0.5) < 1e-12 and lo < P < hi
    assert abs(SB.P_dd(0.0, 1.0) - SB.SWEEP_Z0 / SB.SWEEP_H) < 1e-12
    prod = SB.iter_production_points()
    assert len(prod) == 8600
    assert sum(1 for p in prod if p.split == "main") == 8000
    print(f"  seed hash ok ({s}); Wilson/P_dd ok; production grid=8600",
          flush=True)

    smoke_csv = RESULTS / "sweep_smoke.csv"
    if smoke_csv.exists():
        smoke_csv.unlink()
    specs = []
    for N in SB.SWEEP_N:
        pt = SB.SweepPoint(N=N, sigma_w=0.15, x=0, i=0)
        specs.append({
            "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
            "N_t": 20, "t_end": 0.2,
            "output_dt": 0.05, "dt_max": SB.SWEEP_DT_MAX,
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


# ---------------------------------------------------------------------------
# B0 T1–T6 + scaled ETA
# ---------------------------------------------------------------------------
def _z0_on_grid(traj, t_max: float = 2.0, dt: float = 0.01):
    t = np.asarray(traj.t, dtype=float)
    z0 = np.asarray(traj.x[:, 0, 2], dtype=float)
    tmax = min(float(t_max), float(t[-1]))
    if tmax < 0.0:
        return np.array([0.0]), np.array([float(z0[0])])
    grid = np.arange(0.0, tmax + 0.5 * dt, dt)
    grid = grid[grid <= t_max + 1e-12]
    return grid, np.interp(grid, t, z0)


def _max_abs_dz0(traj_a, traj_b, t_max: float = 2.0) -> float:
    ga, za = _z0_on_grid(traj_a, t_max)
    gb, zb = _z0_on_grid(traj_b, t_max)
    tmax = min(ga[-1], gb[-1], t_max)
    dt = 0.01
    grid = np.arange(0.0, tmax + 0.5 * dt, dt)
    if grid.size == 0:
        return 0.0
    return float(np.max(np.abs(np.interp(grid, ga, za) - np.interp(grid, gb, zb))))


def _run_specs(specs: list[dict], workers: int) -> list[dict]:
    if not specs:
        return []
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        return list(pool.map(SB.run_sweep_job, specs))


def _t2t3_spec(i: int, N_t: int, dt_max: float) -> dict:
    pt = SB.SweepPoint(N=4, sigma_w=0.30, x=0, i=i)
    return {
        "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i, "split": "main",
        "N_t": N_t, "t_end": SB.SWEEP_T_END,
        "output_dt": SB.SWEEP_OUTPUT_DT, "dt_max": dt_max,
        "write_hdf5": False, "hdf5_dir": "",
        "snapshot": False, "snapshot_dir": "",
    }


def _b0_pair_job(spec: dict) -> dict:
    """Pickable T2/T3 worker: one realization + z0(t) for the 1 cm check."""
    from ballooning.integrator import simulate

    pt = SB.SweepPoint(N=int(spec["N"]), sigma_w=float(spec["sigma_w"]),
                       x=int(spec["x"]), i=int(spec["i"]))
    P = SB.sweep_params(pt, N_t=int(spec["N_t"]), t_end=float(spec["t_end"]),
                        output_dt=float(spec["output_dt"]),
                        dt_max=float(spec["dt_max"]))
    t0 = time.perf_counter()
    traj = simulate(P)
    wall = time.perf_counter() - t0
    status = traj.outcome.get("status", "timeout")
    exit_time = traj.outcome.get("exit_time")
    if exit_time is None:
        exit_time = float(traj.t[-1])
    return {
        "N_t": int(spec["N_t"]),
        "dt_max": float(spec["dt_max"]),
        "i": int(spec["i"]),
        "outcome": status,
        "exit_time": float(exit_time),
        "wall_time_s": wall,
        "t": np.asarray(traj.t, dtype=float),
        "z0": np.asarray(traj.x[:, 0, 2], dtype=float),
    }


class _Z0Traj:
    """Minimal stand-in so ``_max_abs_dz0`` can interpolate z0(t)."""

    def __init__(self, t, z0):
        self.t = np.asarray(t, dtype=float)
        self.x = np.zeros((self.t.size, 1, 3))
        self.x[:, 0, 2] = np.asarray(z0, dtype=float)


def _b0_t1() -> dict:
    print("=== B0 T1: renormalized KS, std(w) over 200 seeds ===", flush=True)
    ks_res = do_ksvalid()
    sigma = float(studies.KSVALID_SIGMA)
    samples = []
    n_pts = 64
    z = np.linspace(0.0, 20.0 * studies.KSVALID_ELL, n_pts, endpoint=False)
    line = np.zeros((n_pts, 3))
    line[:, 2] = z
    for s in range(200):
        ks = KinematicSimulation(sigma=sigma, ell=studies.KSVALID_ELL, U_h=0.0,
                                 N_k=200, seed=s, L=0.5, N_t=100,
                                 renormalize=True)
        samples.append(ks.u(line, 0.0)[:, 2])
    w = np.concatenate(samples)
    std_w = float(np.std(w, ddof=0))
    rel = abs(std_w - sigma) / sigma
    passed = rel <= 0.02
    factor = ks_res["renorm_factor"].get(200)
    print(f"  renorm_factor={factor}  std(w)={std_w:.6f}  sigma={sigma}  "
          f"rel={100*rel:.3f}%  PASS={passed}", flush=True)
    return {
        "pass": passed,
        "std_w": std_w,
        "sigma": sigma,
        "rel": rel,
        "renorm_factor": factor,
        "n_seeds": 200,
    }


def _b0_t2(workers: int) -> dict:
    print("=== B0 T2: N_t=50 vs 100 (N=4, σ_w=0.30, x=0, i=0..5) ===",
          flush=True)
    specs50 = [_t2t3_spec(i, 50, SB.SWEEP_DT_MAX) for i in range(6)]
    specs100 = [_t2t3_spec(i, 100, SB.SWEEP_DT_MAX) for i in range(6)]
    jobs = specs50 + specs100
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        results = list(pool.map(_b0_pair_job, jobs))

    by_key = {}
    for rec in results:
        by_key[(int(rec["N_t"]), int(rec["i"]))] = rec
        print(f"  N_t={rec['N_t']} i={rec['i']} {rec['outcome']} "
              f"exit={rec['exit_time']:.3f} wall={rec['wall_time_s']:.1f}s",
              flush=True)

    dz = []
    n_match = 0
    details = []
    for i in range(6):
        a = by_key[(50, i)]
        b = by_key[(100, i)]
        d = _max_abs_dz0(_Z0Traj(a["t"], a["z0"]), _Z0Traj(b["t"], b["z0"]), 2.0)
        dz.append(d)
        match = a["outcome"] == b["outcome"]
        n_match += int(match)
        details.append({
            "i": i, "dz0_m": d, "outcome_50": a["outcome"],
            "outcome_100": b["outcome"],
            "exit_50": a["exit_time"], "exit_100": b["exit_time"],
            "wall_50": a["wall_time_s"], "wall_100": b["wall_time_s"],
            "match": match,
        })
        print(f"  i={i} max|Δz0|(t≤2s)={100*d:.3f} cm  "
              f"{a['outcome']} vs {b['outcome']}", flush=True)
    max_dz = max(dz) if dz else float("nan")
    passed = (max_dz <= 0.01) and (n_match >= 5)
    print(f"  max|Δz0|={100*max_dz:.3f} cm  match={n_match}/6  PASS={passed}",
          flush=True)
    return {
        "pass": passed, "max_abs_dz0": max_dz, "n_outcome_match": n_match,
        "details": details,
    }


def _b0_t3(workers: int) -> dict:
    print("=== B0 T3: dt_max=0.01 vs 0.005 (N_t=50, same 6 seeds) ===",
          flush=True)
    specs01 = [_t2t3_spec(i, 50, 0.01) for i in range(6)]
    specs005 = [_t2t3_spec(i, 50, 0.005) for i in range(6)]
    jobs = specs01 + specs005
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        results = list(pool.map(_b0_pair_job, jobs))

    by_key = {}
    for rec in results:
        by_key[(float(rec["dt_max"]), int(rec["i"]))] = rec
        print(f"  dt_max={rec['dt_max']} i={rec['i']} {rec['outcome']} "
              f"exit={rec['exit_time']:.3f} wall={rec['wall_time_s']:.1f}s",
              flush=True)

    dz = []
    n_match = 0
    ratios = []
    details = []
    for i in range(6):
        a = by_key[(0.01, i)]
        b = by_key[(0.005, i)]
        d = _max_abs_dz0(_Z0Traj(a["t"], a["z0"]), _Z0Traj(b["t"], b["z0"]), 2.0)
        dz.append(d)
        match = a["outcome"] == b["outcome"]
        n_match += int(match)
        ratio = (b["wall_time_s"] / a["wall_time_s"]
                 if a["wall_time_s"] > 0 else float("nan"))
        ratios.append(ratio)
        details.append({
            "i": i, "dz0_m": d, "outcome_01": a["outcome"],
            "outcome_005": b["outcome"],
            "wall_01": a["wall_time_s"], "wall_005": b["wall_time_s"],
            "ratio": ratio, "match": match,
        })
        print(f"  i={i} max|Δz0|={100*d:.3f} cm  "
              f"{a['outcome']} vs {b['outcome']}  "
              f"wall_ratio={ratio:.3f}", flush=True)
    max_dz = max(dz) if dz else float("nan")
    passed = (max_dz <= 0.01) and (n_match >= 5)
    dt_max_factor = float(np.nanmean(ratios)) if ratios else float("nan")
    chosen = 0.01 if passed else 0.005
    scale_factor = 1.0 if chosen == 0.01 else dt_max_factor
    print(f"  max|Δz0|={100*max_dz:.3f} cm  match={n_match}/6  "
          f"dt_max_factor={dt_max_factor:.4f}  chosen dt_max={chosen}  "
          f"PASS={passed}", flush=True)
    return {
        "pass": passed, "max_abs_dz0": max_dz, "n_outcome_match": n_match,
        "dt_max_factor": dt_max_factor, "dt_max": chosen,
        "estimate_dt_scale": scale_factor,
        "details": details,
    }


def _b0_t4(workers: int, N_t: int, dt_max: float) -> dict:
    print("=== B0 T4: persist / resume (N=1, one point, M=8) ===", flush=True)
    t4_dir = RESULTS / "b0_t4"
    csv_path = t4_dir / "sweep.csv"
    hdf5_dir = t4_dir / "ml_samples"
    if t4_dir.exists():
        import shutil
        shutil.rmtree(t4_dir)
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    def _specs(i0, i1):
        out = []
        for i in range(i0, i1):
            pt = SB.SweepPoint(N=1, sigma_w=0.15, x=0, i=i)
            out.append({
                "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
                "split": "main",
                "N_t": N_t, "t_end": SB.SWEEP_T_END,
                "output_dt": SB.SWEEP_OUTPUT_DT, "dt_max": dt_max,
                "write_hdf5": True, "hdf5_dir": str(hdf5_dir),
                "snapshot": False, "snapshot_dir": str(t4_dir / "snap"),
            })
        return out

    first = _run_specs(_specs(0, 4), workers)
    for r in first:
        _append_sweep_row(r, csv_path)
    print(f"  simulated kill after {len(first)} complete", flush=True)

    done = _load_done_keys(csv_path)
    pending = []
    for i in range(8):
        pt = SB.SweepPoint(N=1, sigma_w=0.15, x=0, i=i)
        if (pt.N, pt.sigma_w, pt.x, pt.i) not in done:
            pending.append(pt.i)
    assert pending == [4, 5, 6, 7], f"expected missing 4..7, got {pending}"

    second = _run_specs(_specs(4, 8), workers)
    for r in second:
        _append_sweep_row(r, csv_path)

    rows = _load_sweep_rows(csv_path)
    keys = [_row_done_key(r) for r in rows]
    n_dup = len(keys) - len(set(keys))
    h5s = sorted(hdf5_dir.glob("*.h5"))
    intact = 0
    sample_path = None
    for p in h5s:
        try:
            with h5py.File(p, "r") as f:
                assert "t" in f and "x" in f and "twist" in f
            intact += 1
            if sample_path is None:
                sample_path = str(p)
        except Exception as exc:
            print(f"  HDF5 corrupt {p.name}: {exc}", flush=True)

    passed = (len(rows) == 8 and n_dup == 0 and len(second) == 4
              and intact == 8)
    print(f"  rows={len(rows)} dupes={n_dup} restarted={len(second)} "
          f"hdf5_intact={intact}/8  PASS={passed}", flush=True)
    return {
        "pass": passed, "n_rows": len(rows), "n_dup": n_dup,
        "n_restarted": len(second), "hdf5_intact": intact,
        "sample_hdf5": sample_path,
    }


def _b0_t5(workers: int, N_t: int, dt_max: float) -> dict:
    print("=== B0 T5: sanity N=1, σ_w=0.30, x=0, M=40 ===", flush=True)
    specs = []
    for i in range(40):
        pt = SB.SweepPoint(N=1, sigma_w=0.30, x=0, i=i)
        specs.append({
            "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
            "split": "main",
            "N_t": N_t, "t_end": SB.SWEEP_T_END,
            "output_dt": SB.SWEEP_OUTPUT_DT, "dt_max": dt_max,
            "write_hdf5": False, "hdf5_dir": "",
            "snapshot": False, "snapshot_dir": "",
        })
    rows = _run_specs(specs, workers)
    phase = SB.aggregate_phase(rows)[0]
    print(f"  M={phase['M']} n_up={phase['n_up']} n_down={phase['n_down']} "
          f"n_timeout={phase['n_timeout']}", flush=True)
    print(f"  P={phase['P']:.4f}  [{phase['P_lo']:.4f}, {phase['P_hi']:.4f}]  "
          f"P_dd={phase['P_dd']:.4f} (expect 0.25 at x=0)", flush=True)
    return {
        "pass": True,
        "P": phase["P"], "P_lo": phase["P_lo"], "P_hi": phase["P_hi"],
        "P_dd": phase["P_dd"],
        "n_up": phase["n_up"], "n_down": phase["n_down"],
        "n_timeout": phase["n_timeout"], "M": phase["M"],
    }


def _b0_t6(sample_hdf5: str | None) -> dict:
    print("=== B0 T6: dump one ML HDF5 ===", flush=True)
    if not sample_hdf5 or not Path(sample_hdf5).exists():
        smoke = list((RESULTS / "ml_samples_smoke").glob("*.h5"))
        sample_hdf5 = str(smoke[0]) if smoke else None
    if not sample_hdf5:
        print("  FAIL: no ML HDF5 available", flush=True)
        return {"pass": False, "reason": "no hdf5"}
    lines = [f"file: {sample_hdf5}"]
    datasets = []

    def _walk(g, prefix=""):
        for name, obj in g.items():
            p = f"{prefix}/{name}" if prefix else f"/{name}"
            if isinstance(obj, h5py.Dataset):
                rec = {"path": p, "shape": list(obj.shape),
                       "dtype": str(obj.dtype)}
                datasets.append(rec)
                lines.append(f"  {p:20s}  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                attrs = dict(obj.attrs)
                if attrs:
                    lines.append(f"  {p}/  attrs={list(attrs)}")
                _walk(obj, p)

    with h5py.File(sample_hdf5, "r") as f:
        _walk(f)
        expected = {"/t", "/x", "/v", "/twist", "/u_air", "/charge",
                    "/topology/thread_id", "/topology/node_type",
                    "/topology/edges"}
        have = {d["path"] for d in datasets}
        missing = sorted(expected - have)
        no_theta = "/theta" not in have
        passed = (not missing) and no_theta
        lines.append(f"missing={missing or 'none'}  no_/theta={no_theta}  "
                     f"PASS={passed}")
    text = "\n".join(lines)
    print(text, flush=True)
    return {"pass": passed, "file": sample_hdf5, "datasets": datasets,
            "dump": text, "missing": missing}


def _lpt_wall(durs: list[float], workers: int) -> float:
    if workers <= 0 or not durs:
        return 0.0
    load = [0.0] * workers
    for d in sorted(durs, reverse=True):
        load[load.index(min(load))] += d
    return max(load)


def _b0_estimate(dt_scale: float) -> dict:
    """Scale stored cost-probe walls; do not re-probe.

    scale = 0.5 (N_t 100→50) × T3 estimate_dt_scale
    (1.0 if T3 kept dt_max=0.01; else mean(wall_005/wall_01)).
    """
    means = _cost_means()
    scale = 0.5 * float(dt_scale)
    pts = SB.iter_production_points()
    durs = [SB.cost_weight(p.N, means) * scale for p in pts]
    total_cpu_s = float(sum(durs))
    wall_32 = _lpt_wall(durs, 32)
    wall_2x32 = _lpt_wall(durs, 64)
    wall_3x32 = _lpt_wall(durs, 96)
    n_main = sum(1 for p in pts if p.split == "main")
    n_b2 = len(pts) - n_main
    rec = {
        "n_runs": len(pts),
        "n_b1": n_main,
        "n_b2": n_b2,
        "scale_Nt": 0.5,
        "scale_dt": float(dt_scale),
        "scale_total": scale,
        "cost_probe_mean_s": means,
        "total_cpu_h": total_cpu_s / 3600.0,
        "wall_32_workers_h": wall_32 / 3600.0,
        "wall_2x32_h": wall_2x32 / 3600.0,
        "wall_3x32_h": wall_3x32 / 3600.0,
    }
    rec["workers_are"] = (
        "vCPUs (one ProcessPool worker per requested slot). "
        "c2-standard-N has N vCPUs and N/2 physical cores. "
        "The cost probe used 16 workers on c2-standard-16 (16 vCPUs / 8 physical). "
        "The 32-worker B1 estimate is 32 vCPUs on c2-standard-32 (16 physical cores)."
    )
    rec["physical_cores_c2_32"] = 16
    rec["vcpus_c2_32"] = 32
    wall_16phys = _lpt_wall(durs, 16)
    rec["wall_16_physical_h"] = wall_16phys / 3600.0
    lines = [
        "Stage B B0 cost estimate (no re-probe)",
        f"B1+B2 runs: {len(pts)} (B1={n_main}, B2={n_b2})",
        f"scale = 0.5 (N_t 100→50) × {dt_scale:.6g} (T3 dt_max) = {scale:.6g}",
        f"total CPU  ≈ {rec['total_cpu_h']:.2f} h",
        f"wall @ 32 workers = 32 vCPUs (1× c2-standard-32) ≈ {rec['wall_32_workers_h']:.2f} h",
        f"wall @ 2×32 vCPUs                               ≈ {rec['wall_2x32_h']:.2f} h / machine",
        f"wall @ 3×32 vCPUs                               ≈ {rec['wall_3x32_h']:.2f} h / machine",
        f"wall @ 16 physical cores (if 1 worker/core)     ≈ {rec['wall_16_physical_h']:.2f} h",
        rec["workers_are"],
        "cost-probe means (N_t=100): "
        + ", ".join(f"N={n}:{means[n]:.1f}s" for n in sorted(means)),
        "",
        "Do not launch B1/B2 until confirmed.",
        "",
    ]
    text = "\n".join(lines)
    (RESULTS / "b0_estimate.txt").write_text(text)
    print(text, flush=True)
    return rec


def do_b0(workers: int, skip_prep: bool = False) -> int:
    """T1–T6 + scaled ETA. Writes b0_report.json/.txt, b0_config.json, estimate."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    prov = git_provenance()
    report = {"git_commit": prov["commit"], "git_dirty": prov["dirty"],
              "git_source": prov["source"], "workers": workers}
    if not skip_prep:
        try:
            do_tab_wc_m10(workers)
            report["tab_wc_m10"] = "ok"
        except Exception as exc:
            report["tab_wc_m10"] = f"error: {exc}"
            print(f"tab_wc m=10 failed: {exc}", flush=True)

    report["T1"] = _b0_t1()
    report["T2"] = _b0_t2(workers)
    report["T3"] = _b0_t3(workers)

    cfg = {
        "dt_max": report["T3"]["dt_max"],
        "dt_max_factor": report["T3"]["dt_max_factor"],
        "estimate_dt_scale": report["T3"]["estimate_dt_scale"],
        "t3_pass": report["T3"]["pass"],
        "t2_pass": report["T2"]["pass"],
    }
    B0_CONFIG.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {B0_CONFIG.name}: {cfg}", flush=True)

    dt_max = float(cfg["dt_max"])
    report["T4"] = _b0_t4(workers, N_t=SB.SWEEP_N_T, dt_max=dt_max)
    report["T5"] = _b0_t5(workers, N_t=SB.SWEEP_N_T, dt_max=dt_max)
    report["T6"] = _b0_t6(report["T4"].get("sample_hdf5"))
    report["estimate"] = _b0_estimate(cfg["estimate_dt_scale"])

    strict = ("T1", "T2", "T3", "T4", "T6")
    report["all_pass"] = all(report[k].get("pass") for k in strict)

    lines = [
        "Stage B B0 report",
        f"git={report['git_commit']}  dirty={report['git_dirty']}  "
        f"workers={workers}",
        "",
    ]
    for k in ("T1", "T2", "T3", "T4", "T5", "T6"):
        rec = report[k]
        flag = "PASS" if rec.get("pass") else "FAIL"
        extra = {kk: vv for kk, vv in rec.items()
                 if kk not in ("details", "datasets", "dump", "pass")}
        lines.append(f"{k}  {flag}  {extra}")
    lines.append("")
    lines.append("VERDICT: " + ("PASS" if report["all_pass"] else "FAIL"))
    lines.append("Stop here. Do not launch B1/B2 until confirmed.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (RESULTS / "b0_report.txt").write_text(text)
    with open(RESULTS / "b0_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(text, flush=True)
    (RESULTS / "B0_DONE").write_text(
        f"all_pass={report['all_pass']}\n", encoding="utf-8"
    )
    return 0 if report["all_pass"] else 1


def _grid_ks(N: int, sigma_w: float, x: int, i: int, N_t: int = 50,
             mesh_limited: bool = False) -> KinematicSimulation:
    """KS for a sweep key. Modes ignore N_t unless mesh_limited=True."""
    seed = SB.sweep_seed(N, sigma_w, x, i)
    return KinematicSimulation(
        sigma=sigma_w, ell=SB.SWEEP_ELL, U_h=SB.SWEEP_U_H,
        N_k=SB.SWEEP_N_K, seed=seed, L=0.5, N_t=N_t,
        lambda_=0.5, renormalize=True, mesh_limited=mesh_limited,
    )


def _compare_modes(a: KinematicSimulation, b: KinematicSimulation) -> dict:
    same = a.modes_equal(b)
    return {
        "equal": same,
        "k_max_a": a.k_max, "k_max_b": b.k_max,
        "k_min_a": a.k_min_used, "k_min_b": b.k_min_used,
        "max_abs_dk": float(np.max(np.abs(a.k - b.k))) if a.k.shape == b.k.shape
        else float("nan"),
        "max_abs_da": float(np.max(np.abs(a.a - b.a))) if a.a.shape == b.a.shape
        else float("nan"),
        "max_abs_db": float(np.max(np.abs(a.b - b.b))) if a.b.shape == b.b.shape
        else float("nan"),
        "max_abs_domega": float(np.max(np.abs(a.omega - b.omega)))
        if a.omega.shape == b.omega.shape else float("nan"),
    }


def _release_w_stats(N: int, sigma_w: float, x: int, M: int) -> dict:
    """Mean w and largest-scale w at the release point over M seeds."""
    pos = np.array([0.0, 0.0, SB.SWEEP_Z0])
    w, w_ls = [], []
    for i in range(M):
        ks = _grid_ks(N, sigma_w, x, i, N_t=50, mesh_limited=False)
        w.append(float(ks.u(pos, 0.0)[2]))
        w_ls.append(ks.largest_scale_w(pos, 0.0))
    return {
        "mean_w_release": float(np.mean(w)),
        "mean_largest_scale_w": float(np.mean(w_ls)),
        "std_w_release": float(np.std(w)),
        "n": M,
    }


def integrate_tracer(ks: KinematicSimulation, z0: float = SB.SWEEP_Z0,
                     h: float = SB.SWEEP_H, t_end: float = SB.SWEEP_T_END,
                     dt: float = 1e-3) -> dict:
    """Massless point: dx/dt = u(x,t). Stop at z=0, z=h, or t_end."""
    x = np.array([0.0, 0.0, float(z0)])
    t = 0.0
    while t < t_end - 0.5 * dt:
        def _u(xx, tt):
            return ks.u(xx, tt)
        k1 = _u(x, t)
        k2 = _u(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = _u(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = _u(x + dt * k3, t + dt)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        if x[2] >= h:
            return {"outcome": "rise", "exit_time": t, "z": float(x[2])}
        if x[2] <= 0.0:
            return {"outcome": "fall", "exit_time": t, "z": float(x[2])}
    return {"outcome": "timeout", "exit_time": t, "z": float(x[2])}


def _t7_one(i: int) -> dict:
    ks = _grid_ks(1, 0.30, 0, i, N_t=50, mesh_limited=False)
    rec = integrate_tracer(ks)
    rec["i"] = i
    rec["seed"] = ks.seed
    return rec


def _b0_field_fix_checks() -> dict:
    print("=== B0 field-fix: modes vs N_t, pre-fix T2 seed 0 ===", flush=True)
    seed0 = SB.sweep_seed(4, 0.30, 0, 0)
    old50 = _grid_ks(4, 0.30, 0, 0, N_t=50, mesh_limited=True)
    old100 = _grid_ks(4, 0.30, 0, 0, N_t=100, mesh_limited=True)
    pre = _compare_modes(old50, old100)
    print(f"  pre-fix T2 seed 0 (i=0, seed={seed0}): modes equal={pre['equal']}  "
          f"k_max 50={old50.k_max:.4f} 100={old100.k_max:.4f}  "
          f"max|Δa|={pre['max_abs_da']:.3e}", flush=True)

    new50 = _grid_ks(4, 0.30, 0, 0, N_t=50, mesh_limited=False)
    new100 = _grid_ks(4, 0.30, 0, 0, N_t=100, mesh_limited=False)
    post = _compare_modes(new50, new100)
    print(f"  post-fix same seed N_t=50 vs 100: equal={post['equal']}  "
          f"k_max={new50.k_max:.6f} (target {KS_K_MAX:.6f})", flush=True)

    # T1: N_t=100 already used k_max=KS_K_MAX; mesh-limited N_t=100 is identical
    t1_new = KinematicSimulation(sigma=studies.KSVALID_SIGMA, ell=studies.KSVALID_ELL,
                                 U_h=0.0, N_k=200, seed=0, L=0.5, N_t=100,
                                 renormalize=True, mesh_limited=False)
    t1_old = KinematicSimulation(sigma=studies.KSVALID_SIGMA, ell=studies.KSVALID_ELL,
                                 U_h=0.0, N_k=200, seed=0, L=0.5, N_t=100,
                                 renormalize=True, mesh_limited=True)
    t1 = _compare_modes(t1_new, t1_old)
    t1["k_max"] = t1_new.k_max
    t1["k_max_target"] = KS_K_MAX
    t1["unchanged"] = t1["equal"] and abs(t1_new.k_max - KS_K_MAX) < 1e-12
    print(f"  T1 N_t=100 vs mesh-limited N_t=100: equal={t1['equal']}  "
          f"unchanged={t1['unchanged']}", flush=True)
    return {
        "pre_fix_t2_seed0": {**pre, "seed": seed0,
                             "had_different_modes": not pre["equal"]},
        "post_fix_Nt50_vs_100": post,
        "T1_unchanged": t1,
        "pass_modes": bool(post["equal"]),
        "pass_t1": bool(t1["unchanged"]),
    }


def _b0_t7(M: int = 40) -> dict:
    print(f"=== B0 T7: massless tracer, same {M} fields as T5 ===", flush=True)
    rows = [_t7_one(i) for i in range(M)]
    n_up = sum(1 for r in rows if r["outcome"] == "rise")
    n_down = sum(1 for r in rows if r["outcome"] == "fall")
    n_timeout = sum(1 for r in rows if r["outcome"] == "timeout")
    P, P_lo, P_hi = SB.wilson_ci(n_up, n_up + n_down)
    wstats = _release_w_stats(1, 0.30, 0, M)
    print(f"  M={M} n_up={n_up} n_down={n_down} n_timeout={n_timeout}",
          flush=True)
    print(f"  P={P:.4f}  [{P_lo:.4f}, {P_hi:.4f}]", flush=True)
    print(f"  mean w(release)={wstats['mean_w_release']:.5f}  "
          f"mean largest-scale w={wstats['mean_largest_scale_w']:.5f}",
          flush=True)
    return {
        "pass": True, "P": P, "P_lo": P_lo, "P_hi": P_hi,
        "n_up": n_up, "n_down": n_down, "n_timeout": n_timeout, "M": M,
        **wstats,
    }


def do_b0_followup(workers: int) -> int:
    """Field-fix checks + T2 rerun + T5 rerun + T7. Does not touch T3 or R."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    prov = git_provenance()
    report = {
        "git_commit": prov["commit"], "git_dirty": prov["dirty"],
        "git_source": prov["source"], "workers": workers,
        "note": "B0 follow-up after k_max fix; T3 not rerun",
    }
    print(f"B0 follow-up  git={prov['commit']} dirty={prov['dirty']}  "
          f"workers={workers}", flush=True)

    report["field"] = _b0_field_fix_checks()
    report["T2"] = _b0_t2(workers)
    report["T5"] = _b0_t5(workers, N_t=SB.SWEEP_N_T, dt_max=0.01)
    report["T5"]["release"] = _release_w_stats(1, 0.30, 0, 40)
    print(f"  T5 mean w(release)={report['T5']['release']['mean_w_release']:.5f}  "
          f"mean largest-scale w="
          f"{report['T5']['release']['mean_largest_scale_w']:.5f}", flush=True)
    report["T7"] = _b0_t7(40)
    report["estimate"] = _b0_estimate(1.0)
    report["all_pass"] = bool(
        report["field"]["pass_modes"] and report["field"]["pass_t1"]
        and report["T2"].get("pass")
    )

    lines = [
        "Stage B B0 follow-up",
        f"git={report['git_commit']}  dirty={report['git_dirty']}  "
        f"source={report['git_source']}  workers={workers}",
        "",
        f"field pre-fix T2 seed 0 different modes: "
        f"{report['field']['pre_fix_t2_seed0']['had_different_modes']}  "
        f"{report['field']['pre_fix_t2_seed0']}",
        f"field post-fix N_t=50 vs 100 identical: "
        f"{report['field']['post_fix_Nt50_vs_100']['equal']}",
        f"T1 unchanged (N_t=100 already k_max=KS_K_MAX): "
        f"{report['field']['T1_unchanged']['unchanged']}",
        "",
    ]
    for k in ("T2", "T5", "T7"):
        rec = report[k]
        flag = "PASS" if rec.get("pass") else "FAIL"
        extra = {kk: vv for kk, vv in rec.items()
                 if kk not in ("details", "datasets", "dump", "pass")}
        lines.append(f"{k}  {flag}  {extra}")
    lines.append("")
    lines.append("VERDICT: " + ("PASS" if report["all_pass"] else "FAIL"))
    lines.append("T3 not rerun. Do not launch B1/B2 until confirmed.")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (RESULTS / "b0_followup.txt").write_text(text)
    with open(RESULTS / "b0_followup.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    # Keep the original T2 FAIL numbers; append follow-up into b0_report.json.
    main_path = RESULTS / "b0_report.json"
    if main_path.exists():
        try:
            prev = json.loads(main_path.read_text())
        except Exception:
            prev = {}
        prev["followup"] = report
        prev["git_commit"] = report["git_commit"]
        prev["git_dirty"] = report["git_dirty"]
        main_path.write_text(json.dumps(prev, indent=2, default=str))
    print(text, flush=True)
    (RESULTS / "B0_FOLLOWUP_DONE").write_text(
        f"all_pass={report['all_pass']}\n", encoding="utf-8"
    )
    return 0 if report["all_pass"] else 1


def do_calibrate(workers: int) -> None:
    """Two short production-setting runs per N; rescale vs cost-probe × 0.5."""
    print("=== Stage B machine calibration (2 runs / N) ===", flush=True)
    dt_max = SB.load_dt_max()
    specs = []
    for N in SB.SWEEP_N:
        for i in range(2):
            pt = SB.SweepPoint(N=N, sigma_w=0.15, x=0, i=i)
            specs.append({
                "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
                "N_t": SB.SWEEP_N_T, "t_end": 5.0,
                "output_dt": SB.SWEEP_OUTPUT_DT, "dt_max": dt_max,
                "write_hdf5": False, "hdf5_dir": "",
                "snapshot": False, "snapshot_dir": "",
            })
    rows = _run_specs(specs, workers)
    means = _cost_means()
    by_n: dict[int, list[float]] = {n: [] for n in SB.SWEEP_N}
    for r in rows:
        by_n[int(r["N"])].append(float(r["wall_time_s"]))
        print(f"  N={r['N']} i={r.get('i')} wall={r['wall_time_s']:.1f}s "
              f"{r['outcome']}", flush=True)
    factors = {}
    for n in SB.SWEEP_N:
        cal = float(np.mean(by_n[n])) if by_n[n] else float("nan")
        # probe is full t_end=60, N_t=100; this is t_end=5, N_t=50 — ratio
        # is not a duration match. Report raw walls and vs probe*0.5 as a
        # per-core speed hint only (same N, different t_end).
        factors[n] = {
            "cal_mean_s": cal,
            "probe_mean_s": means.get(n),
            "probe_scaled_Nt50": means.get(n, float("nan")) * 0.5,
        }
    out = {"dt_max": dt_max, "per_N": factors, "git": git_commit_hash()}
    (RESULTS / "calibrate.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ksvalid", action="store_true")
    ap.add_argument("--tab-wc-m10", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cost-probe", action="store_true")
    ap.add_argument("--b0", action="store_true",
                    help="B0 T1–T6 + scaled ETA; writes b0_report.*")
    ap.add_argument("--b0-followup", action="store_true",
                    help="field-fix checks + T2/T5 rerun + T7; do not touch R")
    ap.add_argument("--calibrate", action="store_true",
                    help="2 short runs per N on this machine (B1/B2 speed check)")
    ap.add_argument("--sweep", action="store_true",
                    help="B1+B2 production grid (resume-safe); wait for confirmation")
    ap.add_argument("--uniform-grid", action="store_true",
                    help="with --sweep: use uniform M × --N instead of B1+B2")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--M", type=int, default=SB.SWEEP_M_REALIZATIONS)
    ap.add_argument("--N", type=int, nargs="+", default=list(SB.SWEEP_N))
    ap.add_argument("--t-end", type=float, default=SB.SWEEP_T_END)
    ap.add_argument("--N-t", type=int, default=SB.SWEEP_N_T)
    ap.add_argument("--output-dt", type=float, default=SB.SWEEP_OUTPUT_DT)
    ap.add_argument("--dt-max", type=float, default=None)
    ap.add_argument("--no-hdf5", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-prep", action="store_true",
                    help="with --b0: skip tab_wc m=10 rerun")
    ns = ap.parse_args()

    flags = [ns.ksvalid, ns.tab_wc_m10, ns.smoke, ns.cost_probe, ns.sweep,
             ns.b0, ns.b0_followup, ns.calibrate]
    if not any(flags):
        ap.error("select --ksvalid / --tab-wc-m10 / --smoke / --cost-probe "
                 "/ --b0 / --b0-followup / --calibrate / --sweep")

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
    if ns.calibrate:
        do_calibrate(workers)
    rc = 0
    if ns.b0:
        rc = do_b0(workers, skip_prep=ns.skip_prep)
    if ns.b0_followup:
        rc = do_b0_followup(workers)
    if ns.sweep:
        dt_max = ns.dt_max if ns.dt_max is not None else SB.load_dt_max()
        do_sweep(workers, production=not ns.uniform_grid, M=ns.M,
                 N_values=ns.N, N_t=ns.N_t, t_end=ns.t_end,
                 output_dt=ns.output_dt, dt_max=dt_max,
                 write_hdf5=not ns.no_hdf5, shard=ns.shard,
                 num_shards=ns.num_shards, smoke=False)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
