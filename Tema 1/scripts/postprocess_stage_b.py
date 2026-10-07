"""Stage B post-processing: HDF5 integrity, rebuild sweep.csv, phase + ML index.

Usage (from ``Tema 1/`` on a VM or Mac with ``results/ml_samples/*.h5``):

  python scripts/postprocess_stage_b.py
  python scripts/postprocess_stage_b.py --ml-dir /mnt/pd/tema1/new/results/ml_samples \\
      --results-dir /mnt/pd/tema1/new/results \\
      --legacy-csv results/sweep_shard_00.csv results/sweep_shard_01.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning import sweep as SB
from ballooning.io_hdf5 import ML_T6_DATASETS

RESULTS = ROOT / "results"
EXPECTED_COMMIT_PREFIX = "507637e"

SWEEP_HEADER = [
    "N", "N_t", "sigma_w", "ell", "x", "Fbar_l", "q_nC", "seed", "split",
    "outcome", "exit_time", "R_over_L_mean", "R_over_L_std", "theta_L_mean",
    "dmin_min_um", "entangled", "newton_failures", "wall_time_s", "hdf5_path",
    "valid",
]

PHASE_HEADER = [
    "N", "sigma_w", "x", "Fbar_l", "M", "n_up", "n_down", "n_timeout",
    "P", "P_lo", "P_hi", "P_dd", "mean_exit_time",
]

ML_HEADER = [
    "hdf5_path", "N", "sigma_w", "x", "Fbar_l", "split", "valid", "outcome",
    "n_frames",
]

EXCLUDED_GRID = (
    (4, 0.30, -4),
    (8, 0.15, -4),
    (8, 0.30, -2),
    (8, 0.30, -4),
)


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{float(v):.9e}"
    return str(v)


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({h: _fmt(r.get(h, "")) for h in header})
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def _row_key(r: dict) -> tuple:
    return (int(r["N"]), float(r["sigma_w"]), int(float(r["x"])), int(r["seed"]))


def _legacy_key(r: dict) -> tuple:
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


def check_hdf5(path: Path) -> dict:
    rec = {"path": str(path), "ok": False, "missing": [], "open_error": "",
           "git_commit": "", "git_dirty": None, "no_theta": True}
    try:
        with h5py.File(path, "r") as f:
            have = set()

            def walk(g, prefix=""):
                for name, obj in g.items():
                    p = f"{prefix}/{name}" if prefix else f"/{name}"
                    if isinstance(obj, h5py.Dataset):
                        have.add(p)
                    elif isinstance(obj, h5py.Group):
                        walk(obj, p)

            walk(f)
            rec["missing"] = sorted(ML_T6_DATASETS - have)
            rec["no_theta"] = "/theta" not in have
            if "params" in f:
                rec["git_commit"] = str(f["params"].attrs.get("git_commit", ""))
                rec["git_dirty"] = bool(f["params"].attrs.get("git_dirty", True))
            rec["ok"] = (not rec["missing"]) and rec["no_theta"]
            if rec["git_commit"] and not rec["git_commit"].startswith(
                    EXPECTED_COMMIT_PREFIX):
                rec["ok"] = False
                rec["commit_mismatch"] = True
            if rec["git_dirty"]:
                rec["ok"] = False
    except OSError as exc:
        rec["open_error"] = str(exc)
    return rec


def compare_legacy(rebuilt: list[dict], legacy_paths: list[Path]) -> dict:
    legacy: dict[tuple, dict] = {}
    for p in legacy_paths:
        if not p.exists():
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                legacy[_legacy_key(r)] = r

    rebuilt_by = {}
    for r in rebuilt:
        rebuilt_by[_legacy_key(r)] = r

    common = set(legacy) & set(rebuilt_by)
    float_cols = [
        "Fbar_l", "q_nC", "exit_time", "R_over_L_mean", "R_over_L_std",
        "theta_L_mean", "dmin_min_um",
    ]
    max_diff = {c: 0.0 for c in float_cols}
    int_cols = ["newton_failures"]
    max_diff_int = {c: 0 for c in int_cols}
    mismatches = 0
    for k in common:
        a, b = legacy[k], rebuilt_by[k]
        for c in float_cols:
            try:
                va = float(a[c]) if a.get(c, "") != "" else float("nan")
                vb = float(b[c]) if b.get(c, "") != "" else float("nan")
                if np.isfinite(va) and np.isfinite(vb):
                    max_diff[c] = max(max_diff[c], abs(va - vb))
                elif str(a.get(c, "")).strip() != str(b.get(c, "")).strip():
                    mismatches += 1
            except ValueError:
                mismatches += 1
        for c in int_cols:
            try:
                va = int(float(a[c])) if a.get(c, "") != "" else -999
                vb = int(b[c])
                max_diff_int[c] = max(max_diff_int[c], abs(va - vb))
            except ValueError:
                mismatches += 1
        if str(a.get("outcome", "")) != str(b.get("outcome", "")):
            mismatches += 1

    return {
        "legacy_rows": len(legacy),
        "rebuilt_rows": len(rebuilt_by),
        "matched_keys": len(common),
        "max_abs_diff": max_diff,
        "max_abs_diff_int": max_diff_int,
        "other_mismatches": mismatches,
        "only_legacy": len(set(legacy) - set(rebuilt_by)),
        "only_rebuilt": len(set(rebuilt_by) - set(legacy)),
    }


def _process_one_h5(path_str: str) -> dict:
    path = Path(path_str)
    out = {"check": check_hdf5(path), "row": None, "ml": None, "error": ""}
    try:
        row = SB.sweep_row_from_hdf5(path)
        out["row"] = row
        with h5py.File(path, "r") as f:
            n_frames = int(f["t"].shape[0])
        # Keep B1/B2 split labels; consumers should drop valid=False from main.
        out["ml"] = {
            "hdf5_path": row["hdf5_path"],
            "N": row["N"],
            "sigma_w": row["sigma_w"],
            "x": row["x"],
            "Fbar_l": row["Fbar_l"],
            "split": row["split"],
            "valid": row["valid"],
            "outcome": row["outcome"],
            "n_frames": n_frames,
        }
    except Exception as exc:
        out["error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-dir", type=Path, default=RESULTS / "ml_samples")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--legacy-csv", nargs="*", default=None,
                    help="Stale shard CSVs for comparison")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ns = ap.parse_args()

    ml_dir = ns.ml_dir
    out_dir = ns.results_dir
    h5_files = sorted(ml_dir.glob("*.h5"))
    print(f"ML HDF5 files: {len(h5_files)} in {ml_dir} workers={ns.workers}",
          flush=True)

    bad = []
    rows: list[dict] = []
    ml_rows: list[dict] = []
    paths = [str(p) for p in h5_files]
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        futs = {pool.submit(_process_one_h5, p): p for p in paths}
        done_n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            done_n += 1
            if done_n % 500 == 0:
                print(f"  processed {done_n}/{len(paths)}", flush=True)
            if not rec["check"]["ok"]:
                bad.append(rec["check"])
            if rec["error"]:
                bad.append({"path": futs[fut], "rebuild_error": rec["error"]})
                continue
            rows.append(rec["row"])
            ml_rows.append(rec["ml"])

    rows.sort(key=lambda r: (int(r["N"]), float(r["sigma_w"]), int(r["x"]),
                             int(r["seed"])))
    _write_csv(out_dir / "sweep.csv", SWEEP_HEADER, rows)

    valid_rows = [r for r in rows if r["valid"]]
    phase = SB.aggregate_phase(valid_rows)
    _write_csv(out_dir / "tab_phase.csv", PHASE_HEADER, phase)

    excl_rows = [r for r in rows
                 if (int(r["N"]), float(r["sigma_w"]), int(r["x"])) in EXCLUDED_GRID]
    excl_phase = SB.aggregate_phase(excl_rows)
    _write_csv(out_dir / "tab_phase_excluded.csv", PHASE_HEADER, excl_phase)

    _write_csv(out_dir / "ml_index.csv", ML_HEADER, ml_rows)

    expected = SB.production_point_lookup()
    have_keys = set()
    for path in h5_files:
        m = SB._H5_NAME.match(path.name)
        if m:
            have_keys.add((int(m.group("N")), float(m.group("sw")),
                           int(m.group("x")), int(m.group("i"))))
    missing = [expected[k] for k in expected if k not in have_keys]

    nf_by_grid: dict[tuple, int] = defaultdict(int)
    n_by_grid: dict[tuple, int] = defaultdict(int)
    for r in valid_rows:
        g = (int(r["N"]), float(r["sigma_w"]), int(r["x"]))
        nf_by_grid[g] += int(r["newton_failures"])
        n_by_grid[g] += 1

    legacy_paths = ns.legacy_csv
    if legacy_paths is None:
        legacy_paths = sorted(out_dir.glob("sweep_shard_*.csv"))
    legacy_paths = [Path(p) for p in legacy_paths]
    cmp = compare_legacy(rows, legacy_paths) if legacy_paths else {}

    ml_bytes = sum(p.stat().st_size for p in h5_files)

    report = {
        "n_hdf5": len(h5_files),
        "n_rebuilt_rows": len(rows),
        "n_valid_rows": len(valid_rows),
        "n_bad_hdf5": len(bad),
        "bad_hdf5_sample": bad[:20],
        "production_expected": len(expected),
        "missing_runs": [
            {"N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
             "split": pt.split, "Fbar_l": pt.Fbar_l, "valid": pt.Fbar_l > 0}
            for pt in missing
        ],
        "n_missing": len(missing),
        "excluded_grid": list(EXCLUDED_GRID),
        "newton_failures_valid_grid": {
            f"N={g[0]} sw={g[1]} x={g[2]}": nf_by_grid[g]
            for g in sorted(nf_by_grid)
        },
        "legacy_compare": cmp,
        "ml_samples_bytes": ml_bytes,
        "csv_lag_explanation": (
            "Workers write HDF5 inside run_sweep_job before returning; the parent "
            "appends sweep.csv only after ProcessPoolExecutor collects each future "
            "(as_completed, not in-order result()). When a job raised inside "
            "fut.result(), the old runner aborted the loop, so thousands of "
            "completed jobs had HDF5 on disk but no CSV row. Resume keys came "
            "from CSV only (_load_done_keys), not from existing .h5 files."
        ),
    }
    rep_path = out_dir / "postprocess_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({
        "n_hdf5": report["n_hdf5"],
        "n_bad": report["n_bad_hdf5"],
        "n_missing": report["n_missing"],
        "legacy_matched": cmp.get("matched_keys"),
        "max_diff": cmp.get("max_abs_diff"),
    }, indent=2), flush=True)

    if bad:
        print(f"WARNING: {len(bad)} HDF5 issues (see {rep_path})", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
