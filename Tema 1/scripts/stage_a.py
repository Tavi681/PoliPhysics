"""Stage A artifacts: turbulence-generator validation, cheap studies, and pilot.

This script is ADDITIVE. It never deletes the baseline results/*.csv produced by
scripts/produce_results.py; it only writes the new Stage A files and merges a
``stage_a`` section into results/meta.json. Run produce_results.py first (for the
regression baseline), then this.

Outputs (results/):
  fig_ksvalid.csv       N_k, k, E_target, E_measured        (KS spectrum)
  fig_ksvalid_pdf.csv   N_k, w_bin_center, pdf, gaussian     (KS one-point PDF)
  tab_wc.csv            N, m_mg, q_nC, wc_numeric, wc_analytic, rel_err_pct
  relax.csv             t, dR_over_L, dshape_over_L          (lateral relaxation)
  pilot.csv             N, seed, outcome, ... cost columns    (alg:sweep pilot)
  ml_samples/pilot_N*.h5  float32 HDF5 samples (size reported in meta.json)

Usage:
  python scripts/stage_a.py --ksvalid --tab-wc --relax     # cheap studies
  python scripts/stage_a.py --pilot                        # pilot (heavier)
  python scripts/stage_a.py --all
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ballooning.io_hdf5 import git_commit_hash, write_trajectory
from ballooning.integrator import simulate
from ballooning import studies

RESULTS = ROOT / "results"
DEFAULT_CACHE = RESULTS / "equal_t_cache"


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.9e}"
    return str(v)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    print(f"wrote {path.name} ({len(rows)} rows)", flush=True)


def _load_meta() -> dict:
    p = RESULTS / "meta.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _save_meta(meta: dict) -> None:
    with open(RESULTS / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("updated results/meta.json (stage_a section)", flush=True)


def _stage_a_section(meta: dict) -> dict:
    return meta.setdefault("stage_a", {})


# ---------------------------------------------------------------------------
def do_ksvalid(meta: dict, n_k=(100, 200)) -> None:
    print("=== KS validation (fig_ksvalid, fig_ksvalid_pdf) ===", flush=True)
    res = studies.run_ks_validation(n_k)

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

    sec = _stage_a_section(meta)
    sec["ks_validation"] = {
        "sigma": res["sigma"], "ell": res["ell"],
        "n_seeds": studies.KSVALID_SEEDS,
        "line_points": studies.KSVALID_NPTS,
        "span_ell": studies.KSVALID_SPAN_ELL,
        "energy_fraction": {str(k): v for k, v in res["energy_fraction"].items()},
        "energy_fraction_note": (
            "fraction of (3/2)sigma^2 captured by the discrete modes; the deficit "
            "is the sub-mesh high-k tail beyond k_max=2pi/(5 dl) (dl=L/N_t), which "
            "k_min cannot recover for ell=1, N_t=100 -> ~0.964"),
    }


def do_tab_wc(meta: dict, workers: int) -> None:
    print("=== tab_wc (parallel) ===", flush=True)
    specs = studies.wc_specs()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(studies.run_wc_job, specs))
    wall = time.perf_counter() - t0
    out.sort(key=lambda r: (r["N"], r["m_mg"], r["q_nC"]))
    rows = []
    for r in out:
        rows.append([r["N"], r["m_mg"], r["q_nC"], r["wc_numeric"],
                     r["wc_analytic"], r["rel_err_pct"]])
        flag = " FOLD-BELOW-SPIDER" if r["fold_below_spider"] else ""
        print(f"  N={r['N']} m={r['m_mg']}mg q={r['q_nC']}nC "
              f"wc_num={r['wc_numeric']:.4e} wc_ana={r['wc_analytic']:.4e} "
              f"err={r['rel_err_pct']:.2f}% status={r['status']}{flag}", flush=True)
    _write_csv(RESULTS / "tab_wc.csv",
               ["N", "m_mg", "q_nC", "wc_numeric", "wc_analytic", "rel_err_pct"],
               rows)
    sec = _stage_a_section(meta)
    sec["tab_wc"] = {
        "wall_time_s": wall,
        "max_rel_err_pct": max(r["rel_err_pct"] for r in out),
        "fold_below_spider_cases": [
            {"N": r["N"], "m_mg": r["m_mg"], "q_nC": r["q_nC"]}
            for r in out if r["fold_below_spider"]],
    }


def do_relax(meta: dict, cache_dir: Path, dt: float | None) -> None:
    print("=== lateral relaxation (relax.csv) ===", flush=True)
    if dt is None:
        # read dt from the cache
        d = np.load(cache_dir / "w0.npz", allow_pickle=False)
        dt = float(d["dt"]) if "dt" in d.files else studies._select_fixed_dt()
    res = studies.run_lateral_relaxation(str(cache_dir), dt)
    rows = [[t, dr, ds] for t, dr, ds in
            zip(res["t"], res["dR_over_L"], res["dshape_over_L"])]
    _write_csv(RESULTS / "relax.csv", ["t", "dR_over_L", "dshape_over_L"], rows)
    sec = _stage_a_section(meta)
    sec["relaxation"] = {
        "tau_R_s": res["tau_R_s"],
        "tau_shape_s": res["tau_shape_s"],
        "t_s": res["t_s"],
        "fit_t_min_s": res["fit_t_min_s"],
        "source": "results/equal_t_cache transient (v0=0) runs",
    }
    print(f"  tau_R={res['tau_R_s']:.4f}s tau_shape={res['tau_shape_s']:.4f}s "
          f"t_s={res['t_s']:.4f}s", flush=True)


def do_pilot(meta: dict, workers: int, n_t: int = 100,
             t_end: float = studies.PILOT_T_END, smoke: bool = False) -> None:
    tag = " SMOKE" if smoke else ""
    print(f"=== pilot{tag} (alg:sweep, NOT production) t_end={t_end}s ===", flush=True)
    specs = studies.pilot_specs(t_end=t_end, N_t=n_t)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(studies.run_pilot_job, specs))
    wall = time.perf_counter() - t0
    out.sort(key=lambda r: (r["N"], r["seed"]))
    header = ["N", "seed", "outcome", "exit_time", "simulated_time",
              "wall_time_s", "wall_over_sim", "mean_dt", "min_dt",
              "newton_failures", "max_abs_u"]
    rows = [[r[k] for k in header] for r in out]
    for r in out:
        print(f"  N={r['N']} seed={r['seed']} {r['outcome']} "
              f"exit={r['exit_time']:.3f}s sim={r['simulated_time']:.3f}s "
              f"wall={r['wall_time_s']:.1f}s w/s={r['wall_over_sim']:.1f} "
              f"nf={r['newton_failures']} max|u|={r['max_abs_u']:.3f}", flush=True)
    _write_csv(RESULTS / "pilot.csv", header, rows)

    # --- ML HDF5 samples: float32 at output_dt=0.05, size per simulated second ---
    ml_dir = RESULTS / "ml_samples"
    ml_dir.mkdir(exist_ok=True)
    hdf5_sizes = {}
    # bytes/simulated-second is duration-independent, so a short sample suffices.
    ml_tend = min(5.0, t_end)
    for N in studies.PILOT_N:
        P = studies.pilot_params(N, seed=0, N_t=n_t, t_end=ml_tend)
        P.output_dt = 0.05
        traj = simulate(P)
        path = ml_dir / f"pilot_N{N}.h5"
        write_trajectory(str(path), P, traj, float32=True)
        size = path.stat().st_size
        sim = float(traj.t[-1]) if traj.t[-1] > 0 else float("nan")
        hdf5_sizes[str(N)] = {
            "file_bytes": size,
            "simulated_time_s": sim,
            "bytes_per_sim_second": size / sim if sim > 0 else None,
        }
        print(f"  ML N={N}: {size/1e6:.2f} MB over {sim:.2f}s "
              f"-> {size/sim/1e6:.3f} MB/s", flush=True)

    sec = _stage_a_section(meta)
    sec["pilot"] = {
        "wall_time_s": wall,
        "smoke": smoke,
        "t_end": t_end,
        "config": {
            "N": list(studies.PILOT_N), "seeds": list(studies.PILOT_SEEDS),
            "Fbar_l": studies.PILOT_FBAR, "sigma_w": studies.PILOT_SIGMA_W,
            "ell": studies.PILOT_ELL, "U_h": studies.PILOT_U_H,
            "N_k": studies.PILOT_N_K, "z0": studies.PILOT_Z0, "h": studies.PILOT_H,
            "t_end": studies.PILOT_T_END, "release_mode": "clamped",
        },
        "runs": out,
        "hdf5_float32": {"output_dt": 0.05, "sizes": hdf5_sizes},
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksvalid", action="store_true")
    ap.add_argument("--tab-wc", action="store_true")
    ap.add_argument("--relax", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--pilot-nt", type=int, default=100)
    ap.add_argument("--pilot-tend", type=float, default=studies.PILOT_T_END)
    ap.add_argument("--pilot-smoke", action="store_true",
                    help="short pilot to validate the pipeline (t_end=0.5, N_t=40)")
    ns = ap.parse_args()

    do = dict(ksvalid=ns.ksvalid or ns.all,
              tab_wc=ns.tab_wc or ns.all,
              relax=ns.relax or ns.all,
              pilot=ns.pilot or ns.all)
    if not any(do.values()):
        ap.error("select at least one of --ksvalid --tab-wc --relax --pilot --all")

    RESULTS.mkdir(parents=True, exist_ok=True)
    workers = ns.workers or (os.cpu_count() or 4)
    print(f"stage_a: workers={workers} git={git_commit_hash()[:10]}", flush=True)

    meta = _load_meta()
    _stage_a_section(meta)["git_commit"] = git_commit_hash()

    if do["ksvalid"]:
        do_ksvalid(meta)
        _save_meta(meta)
    if do["tab_wc"]:
        do_tab_wc(meta, workers)
        _save_meta(meta)
    if do["relax"]:
        do_relax(meta, Path(ns.cache_dir), ns.dt)
        _save_meta(meta)
    if do["pilot"]:
        if ns.pilot_smoke:
            do_pilot(meta, workers, n_t=40, t_end=0.5, smoke=True)
        else:
            do_pilot(meta, workers, ns.pilot_nt, ns.pilot_tend)
        _save_meta(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
