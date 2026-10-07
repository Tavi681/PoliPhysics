"""Merge shard post-process outputs into final Stage B tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning import sweep as SB

RESULTS = ROOT / "results"

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
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:.9e}"
    s = str(v)
    if s.lower() in ("true", "false"):
        return s.lower()
    return s


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({h: _fmt(r.get(h, "")) for h in header})
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def _key(r: dict) -> tuple:
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


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _truthy_valid(r: dict) -> bool:
    v = r.get("valid", "true")
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes")


def compare_legacy(rebuilt: list[dict], legacy_paths: list[Path]) -> dict:
    import numpy as np

    legacy: dict[tuple, dict] = {}
    for p in legacy_paths:
        if not p.exists():
            continue
        for r in load_csv(p):
            legacy[_key(r)] = r
    rebuilt_by = {_key(r): r for r in rebuilt}
    common = set(legacy) & set(rebuilt_by)
    float_cols = [
        "Fbar_l", "q_nC", "exit_time", "R_over_L_mean", "R_over_L_std",
        "theta_L_mean", "dmin_min_um",
    ]
    max_diff = {c: 0.0 for c in float_cols}
    for k in common:
        a, b = legacy[k], rebuilt_by[k]
        for c in float_cols:
            try:
                va = float(a[c]) if a.get(c, "") != "" else float("nan")
                vb = float(b[c]) if b.get(c, "") != "" else float("nan")
                if np.isfinite(va) and np.isfinite(vb):
                    max_diff[c] = max(max_diff[c], abs(va - vb))
            except ValueError:
                pass
    return {
        "legacy_rows": len(legacy),
        "rebuilt_rows": len(rebuilt_by),
        "matched_keys": len(common),
        "max_abs_diff": max_diff,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--legacy-csv", nargs="*", default=None)
    ns = ap.parse_args()

    rows: list[dict] = []
    ml: list[dict] = []
    seen = set()
    for d in ns.parts:
        d = Path(d)
        for r in load_csv(d / "sweep.csv"):
            k = _key(r)
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
        if (d / "ml_index.csv").exists():
            ml.extend(load_csv(d / "ml_index.csv"))

    rows.sort(key=lambda r: (int(r["N"]), float(r["sigma_w"]), int(float(r["x"])),
                             int(r["seed"])))
    out = ns.out
    _write_csv(out / "sweep.csv", SWEEP_HEADER, rows)

    valid_rows = [r for r in rows if _truthy_valid(r)]
    phase = SB.aggregate_phase([
        {**r, "N": int(r["N"]), "sigma_w": float(r["sigma_w"]),
         "x": int(float(r["x"])), "Fbar_l": float(r["Fbar_l"]),
         "outcome": r["outcome"], "exit_time": float(r["exit_time"])}
        for r in valid_rows
    ])
    _write_csv(out / "tab_phase.csv", PHASE_HEADER, phase)

    excl_rows = [
        {**r, "N": int(r["N"]), "sigma_w": float(r["sigma_w"]),
         "x": int(float(r["x"])), "Fbar_l": float(r["Fbar_l"]),
         "outcome": r["outcome"], "exit_time": float(r["exit_time"])}
        for r in rows
        if (int(r["N"]), float(r["sigma_w"]), int(float(r["x"]))) in EXCLUDED_GRID
    ]
    _write_csv(out / "tab_phase_excluded.csv", PHASE_HEADER,
               SB.aggregate_phase(excl_rows))

    # Prefer relative ml_samples path for Mac consumers
    for r in ml:
        p = Path(r["hdf5_path"])
        r["hdf5_path"] = f"ml_samples/{p.name}"
    ml.sort(key=lambda r: (int(r["N"]), float(r["sigma_w"]), int(float(r["x"])),
                           r["hdf5_path"]))
    _write_csv(out / "ml_index.csv", ML_HEADER, ml)

    expected = SB.production_point_lookup()
    have = set()
    for r in rows:
        have.add(_key(r))
    missing = [expected[k] for k in expected if k not in have]

    nf_by_grid: dict[tuple, int] = defaultdict(int)
    n_by_grid: dict[tuple, int] = defaultdict(int)
    for r in valid_rows:
        g = (int(r["N"]), float(r["sigma_w"]), int(float(r["x"])))
        nf_by_grid[g] += int(float(r["newton_failures"] or 0))
        n_by_grid[g] += 1

    missing_all_excluded = all(pt.Fbar_l <= 0 for pt in missing)
    missing_in_excl = all(
        (pt.N, pt.sigma_w, pt.x) in EXCLUDED_GRID for pt in missing
    )

    report = {
        "n_sweep_rows": len(rows),
        "n_valid": len(valid_rows),
        "n_ml_index": len(ml),
        "n_missing": len(missing),
        "missing_all_Fbar_le_0": missing_all_excluded,
        "missing_all_in_excluded_grid": missing_in_excl,
        "missing_runs": [
            {"N": pt.N, "sigma_w": pt.sigma_w, "x": pt.x, "i": pt.i,
             "Fbar_l": pt.Fbar_l, "split": pt.split}
            for pt in missing
        ],
        "newton_failures_valid_grid": {
            f"N={g[0]} sw={g[1]} x={g[2]} (M={n_by_grid[g]})": nf_by_grid[g]
            for g in sorted(nf_by_grid)
        },
        "csv_lag_explanation": (
            "Workers write HDF5 inside run_sweep_job before returning; the parent "
            "appends sweep.csv only after ProcessPoolExecutor collects each future "
            "(as_completed, not in-order fut.result()). When a job raised inside "
            "fut.result(), the old runner aborted the loop, so thousands of "
            "completed jobs had HDF5 on disk but no CSV row. Resume keys came "
            "from CSV only (_load_done_keys), not from existing .h5 files."
        ),
    }
    if ns.legacy_csv:
        report["legacy_compare"] = compare_legacy(
            rows, [Path(p) for p in ns.legacy_csv])
    with open(out / "postprocess_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({
        "n_sweep_rows": report["n_sweep_rows"],
        "n_missing": report["n_missing"],
        "missing_all_in_excluded_grid": missing_in_excl,
        "legacy": report.get("legacy_compare"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
