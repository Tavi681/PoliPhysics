"""Relaxed-clamp release check (N in {4,8} x x in {0,1} x M=200, sigma_w=0.30).

Uses --clamp-relax-time (fixed still-air clamp) via run_sweep_job. Resume-safe.
Does not write ML HDF5. Outputs CSV under results/clamp_check/.

Usage (from Tema 1/):
  python scripts/clamp_check_run.py --probe --shard 0 --num-shards 2 --workers 4
  python scripts/clamp_check_run.py --run --shard 0 --num-shards 2 --workers 28 \\
      --clamp-relax-time 3.0
  python scripts/clamp_check_run.py --compare
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning import sweep as SB
from ballooning.io_hdf5 import git_provenance

OUT = ROOT / "results" / "clamp_check"
HEADER = [
    "N", "N_t", "sigma_w", "ell", "x", "Fbar_l", "seed", "i", "split",
    "outcome", "exit_time",
    "R_over_L_release", "R_over_L_mean", "R_over_L_std",
    "dmin_min_um", "newton_failures", "wall_time_s",
    "clamp_relax_time", "phase1_t_exit",
    "git_commit", "git_dirty",
]
COST = {4: SB.SWEEP_COST_WEIGHTS[4], 8: SB.SWEEP_COST_WEIGHTS[8]}


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        if isinstance(v, float) and math.isnan(v):
            return ""
        return f"{float(v):.9e}"
    return str(v)


def iter_clamp_points(M: int = 200) -> list[SB.SweepPoint]:
    out = []
    for N in (4, 8):
        for x in (0, 1):
            for i in range(M):
                out.append(SB.SweepPoint(N=N, sigma_w=0.30, x=x, i=i, split="main"))
    return out


def shard_assign(points: list[SB.SweepPoint], num_shards: int) -> list[int]:
    return SB.shard_assignment(points, num_shards, COST)


def _key(r: dict) -> tuple:
    return (int(r["N"]), float(r["sigma_w"]), int(float(r["x"])), int(r["i"]))


def load_done(csv_path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                done.add(_key(r))
            except Exception:
                continue
    return done


def csv_path_for(shard: int, num_shards: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    if num_shards <= 1:
        return OUT / "relaxed_release_sweep.csv"
    return OUT / f"relaxed_release_shard_{shard:02d}.csv"


def _row_from_job(row: dict) -> dict:
    return {h: row.get(h, "") for h in HEADER}


def run_jobs(specs: list[dict], workers: int, out_csv: Path) -> list[dict]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    rows_out = []
    t0 = time.perf_counter()
    n_fin = 0
    with ProcessPoolExecutor(max_workers=workers) as pool, \
            open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n",
                           extrasaction="ignore")
        if new:
            w.writeheader()
        futs = {pool.submit(SB.run_sweep_job, s): s for s in specs}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as exc:
                s = futs[fut]
                print(f"FAIL N={s['N']} x={s['x']} i={s['i']}: {exc!r}",
                      flush=True)
                continue
            rec = _row_from_job(row)
            w.writerow({h: _fmt(rec.get(h, "")) for h in HEADER})
            f.flush()
            rows_out.append(row)
            n_fin += 1
            if n_fin % 5 == 0 or n_fin == len(specs):
                elapsed = time.perf_counter() - t0
                rate = n_fin / elapsed if elapsed else 0
                eta = (len(specs) - n_fin) / rate if rate else float("inf")
                print(f"  progress {n_fin}/{len(specs)} "
                      f"elapsed={elapsed/3600:.2f}h ETA={eta/3600:.2f}h "
                      f"last N={row['N']} x={row['x']} i={row.get('i')} "
                      f"{row['outcome']} wall={float(row['wall_time_s']):.1f}s",
                      flush=True)
    return rows_out


def do_probe(shard: int, num_shards: int, workers: int,
             clamp_relax_time: float, n_probe: int = 4) -> dict:
    points = iter_clamp_points(M=200)
    assign = shard_assign(points, num_shards)
    mine = [pt for pt, s in zip(points, assign) if s == shard]
    # Spread probe across N if possible
    by_n: dict[int, list] = {}
    for pt in mine:
        by_n.setdefault(pt.N, []).append(pt)
    probe_pts = []
    for N in sorted(by_n):
        probe_pts.extend(by_n[N][: max(1, n_probe // max(1, len(by_n)))])
    probe_pts = probe_pts[:n_probe]
    dt_max = SB.load_dt_max()
    specs = [{
        "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
        "split": pt.split, "N_t": SB.SWEEP_N_T, "t_end": SB.SWEEP_T_END,
        "output_dt": SB.SWEEP_OUTPUT_DT, "dt_max": dt_max,
        "write_hdf5": False, "hdf5_dir": "",
        "snapshot": False, "snapshot_dir": "",
        "clamp_relax_time": clamp_relax_time,
    } for pt in probe_pts]
    print(f"PROBE shard={shard}/{num_shards} n={len(specs)} "
          f"clamp_relax_time={clamp_relax_time} workers={workers}", flush=True)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        for row in pool.map(SB.run_sweep_job, specs):
            rows.append(row)
            print(f"  probe N={row['N']} x={row['x']} i={row.get('i')} "
                  f"{row['outcome']} wall={float(row['wall_time_s']):.1f}s "
                  f"R_rel={float(row.get('R_over_L_release', float('nan'))):.4e} "
                  f"phase1={row.get('phase1_t_exit')}", flush=True)
    walls = [float(r["wall_time_s"]) for r in rows]
    mean_by_n = {}
    for r in rows:
        mean_by_n.setdefault(int(r["N"]), []).append(float(r["wall_time_s"]))
    mean_by_n = {n: float(np.mean(v)) for n, v in mean_by_n.items()}
    # project full shard cost
    rem = [pt for pt in mine]  # full shard incl. done — conservative
    # better: pending only
    done = load_done(csv_path_for(shard, num_shards))
    # also merge any combined file
    done |= load_done(OUT / "relaxed_release_sweep.csv")
    pending = [pt for pt in mine if (pt.N, pt.sigma_w, pt.x, pt.i) not in done]
    # scale from probe means; fall back to COST ratios
    def wall_est(N):
        if N in mean_by_n:
            return mean_by_n[N]
        # scale from available probe N
        if mean_by_n:
            ref_n = next(iter(mean_by_n))
            return mean_by_n[ref_n] * COST[N] / COST[ref_n]
        return COST[N] * 0.5  # N_t=50 heuristic vs probe at N_t=100
    cpu_s = sum(wall_est(pt.N) for pt in pending)
    wall_h = (cpu_s / max(workers, 1)) / 3600.0
    rec = {
        "shard": shard, "n_probe": len(rows), "walls_s": walls,
        "mean_by_n": mean_by_n, "n_pending": len(pending),
        "n_shard": len(mine), "proj_wall_h": wall_h,
        "probe_wall_s": time.perf_counter() - t0,
    }
    print(f"PROBE summary: mean_by_n={mean_by_n} pending={len(pending)} "
          f"proj_wall_h={wall_h:.2f} (workers={workers})", flush=True)
    return rec


def do_run(shard: int, num_shards: int, workers: int,
           clamp_relax_time: float, M: int = 200) -> None:
    prov = git_provenance()
    print(f"RUN shard={shard}/{num_shards} workers={workers} "
          f"clamp_relax_time={clamp_relax_time} "
          f"git={prov['commit'][:12]} dirty={prov['dirty']}", flush=True)
    if prov["dirty"]:
        raise SystemExit("ABORT: dirty tree")
    points = iter_clamp_points(M=M)
    assign = shard_assign(points, num_shards)
    mine = [pt for pt, s in zip(points, assign) if s == shard]
    out_csv = csv_path_for(shard, num_shards)
    done = load_done(out_csv) | load_done(OUT / "relaxed_release_sweep.csv")
    # also skip other shard files if present locally
    for p in OUT.glob("relaxed_release_shard_*.csv"):
        done |= load_done(p)
    pending = [pt for pt in mine if (pt.N, pt.sigma_w, pt.x, pt.i) not in done]
    print(f"  shard points={len(mine)} done={len(mine)-len(pending)} "
          f"pending={len(pending)} csv={out_csv.name}", flush=True)
    if not pending:
        print("  nothing pending", flush=True)
        return
    dt_max = SB.load_dt_max()
    specs = [{
        "N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
        "split": pt.split, "N_t": SB.SWEEP_N_T, "t_end": SB.SWEEP_T_END,
        "output_dt": SB.SWEEP_OUTPUT_DT, "dt_max": dt_max,
        "write_hdf5": False, "hdf5_dir": "",
        "snapshot": False, "snapshot_dir": "",
        "clamp_relax_time": clamp_relax_time,
    } for pt in pending]
    run_jobs(specs, workers, out_csv)
    (OUT / f"clamp_check_DONE_shard_{shard:02d}").write_text(
        f"finished\nshard={shard}\nnum_shards={num_shards}\n"
        f"git={prov['commit']}\ndirty={prov['dirty']}\n"
        f"clamp_relax_time={clamp_relax_time}\n",
        encoding="utf-8",
    )


def two_prop_pvalue(n1: int, N1: int, n2: int, N2: int) -> float:
    """Two-sided two-proportion z-test p-value (no continuity correction)."""
    if N1 <= 0 or N2 <= 0:
        return float("nan")
    p1, p2 = n1 / N1, n2 / N2
    p = (n1 + n2) / (N1 + N2)
    if p <= 0 or p >= 1:
        return 1.0 if abs(p1 - p2) < 1e-15 else 0.0
    se = math.sqrt(p * (1 - p) * (1 / N1 + 1 / N2))
    if se <= 0:
        return float("nan")
    z = abs(p1 - p2) / se
    # erfc for two-sided normal
    return float(math.erfc(z / math.sqrt(2.0)))


def do_compare() -> Path:
    """Build relaxed_vs_sweep.csv from all clamp_check CSVs + sweep.csv."""
    from collections import defaultdict

    rows_rel = []
    for p in [OUT / "relaxed_release_sweep.csv", *sorted(OUT.glob("relaxed_release_shard_*.csv"))]:
        if not p.exists():
            continue
        with open(p, newline="") as f:
            rows_rel.extend(csv.DictReader(f))
    # dedup by key (prefer later)
    by = {}
    for r in rows_rel:
        by[_key(r)] = r
    rows_rel = list(by.values())

    sweep_g = defaultdict(list)
    with open(ROOT / "results" / "sweep.csv", newline="") as f:
        for r in csv.DictReader(f):
            if str(r.get("valid", "true")).lower() not in ("true", "1", "yes"):
                continue
            key = (int(r["N"]), float(r["sigma_w"]), int(float(r["x"])))
            if key[0] in (4, 8) and abs(key[1] - 0.30) < 1e-12 and key[2] in (0, 1):
                sweep_g[key].append(r)

    rel_g = defaultdict(list)
    for r in rows_rel:
        rel_g[(int(r["N"]), float(r["sigma_w"]), int(float(r["x"])))].append(r)

    out_rows = []
    for key in sorted(set(rel_g) | set(sweep_g)):
        rg = rel_g.get(key, [])
        sg = sweep_g.get(key, [])
        r_up = sum(1 for r in rg if r["outcome"] == "rise")
        r_down = sum(1 for r in rg if r["outcome"] == "fall")
        r_to = sum(1 for r in rg if r["outcome"] == "timeout")
        s_up = sum(1 for r in sg if r["outcome"] == "rise")
        s_down = sum(1 for r in sg if r["outcome"] == "fall")
        s_to = sum(1 for r in sg if r["outcome"] == "timeout")
        Pr, Prlo, Prhi = SB.wilson_ci(r_up, r_up + r_down)
        Ps, Pslo, Pshi = SB.wilson_ci(s_up, s_up + s_down)
        # decided counts for two-prop test
        pval = two_prop_pvalue(r_up, r_up + r_down, s_up, s_up + s_down)
        R_rel = float(np.mean([float(r["R_over_L_mean"]) for r in rg])) if rg else float("nan")
        R_sw = float(np.mean([float(r["R_over_L_mean"]) for r in sg])) if sg else float("nan")
        out_rows.append({
            "N": key[0], "sigma_w": key[1], "x": key[2],
            "M_relaxed": len(rg), "M_sweep": len(sg),
            "P_sweep": Ps, "P_lo_sweep": Pslo, "P_hi_sweep": Pshi,
            "P_relaxed": Pr, "P_lo_relaxed": Prlo, "P_hi_relaxed": Prhi,
            "P_diff": (Pr - Ps) if (math.isfinite(Pr) and math.isfinite(Ps)) else float("nan"),
            "p_value_two_prop": pval,
            "R_over_L_mean_sweep": R_sw,
            "R_over_L_mean_relaxed": R_rel,
            "n_timeout_sweep": s_to, "n_timeout_relaxed": r_to,
            "n_up_sweep": s_up, "n_down_sweep": s_down,
            "n_up_relaxed": r_up, "n_down_relaxed": r_down,
        })
        print(f"N={key[0]} x={key[2]}: P_sw={Ps:.4f} P_rel={Pr:.4f} "
              f"diff={out_rows[-1]['P_diff']:.4f} p={pval:.3g} "
              f"R/L_sw={R_sw:.4e} R/L_rel={R_rel:.4e} "
              f"to_sw={s_to} to_rel={r_to}", flush=True)

    path = OUT / "relaxed_vs_sweep.csv"
    header = list(out_rows[0].keys()) if out_rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow({h: _fmt(r[h]) for h in header})
    print(f"wrote {path}", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--clamp-relax-time", type=float, default=3.0)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--max-proj-h", type=float, default=6.0,
                    help="abort probe with exit 2 if projected wall > this")
    ns = ap.parse_args()
    if not any([ns.probe, ns.run, ns.compare]):
        ap.error("select --probe / --run / --compare")
    if ns.probe:
        rec = do_probe(ns.shard, ns.num_shards, ns.workers, ns.clamp_relax_time)
        if rec["proj_wall_h"] > ns.max_proj_h:
            print(f"ABORT: projected {rec['proj_wall_h']:.2f}h > {ns.max_proj_h}h",
                  flush=True)
            return 2
        print("PROBE_OK", flush=True)
    if ns.run:
        do_run(ns.shard, ns.num_shards, ns.workers, ns.clamp_relax_time, M=ns.M)
    if ns.compare:
        do_compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
