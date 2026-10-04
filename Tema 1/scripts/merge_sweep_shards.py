"""Concatenate Stage B sweep shards; reject duplicates; report missing runs.

Usage (from Tema 1/):
  python scripts/merge_sweep_shards.py
  python scripts/merge_sweep_shards.py --shards results/sweep_shard_*.csv
  python scripts/merge_sweep_shards.py --out results/sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning import sweep as SB

RESULTS = ROOT / "results"
HEADER = [
    "N", "N_t", "sigma_w", "ell", "x", "Fbar_l", "q_nC", "seed", "split",
    "outcome", "exit_time", "R_over_L_mean", "R_over_L_std", "theta_L_mean",
    "dmin_min_um", "entangled", "newton_failures", "wall_time_s", "hdf5_path",
]


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


def _prod_key(pt: SB.SweepPoint) -> tuple:
    return (pt.N, pt.sigma_w, pt.x, pt.i)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", default=None,
                    help="shard CSVs (default: results/sweep_shard_*.csv)")
    ap.add_argument("--out", type=Path, default=RESULTS / "sweep.csv")
    ns = ap.parse_args()

    if ns.shards:
        paths = [Path(p) for p in ns.shards]
    else:
        paths = sorted(RESULTS.glob("sweep_shard_*.csv"))
        if (RESULTS / "sweep.csv").exists() and not paths:
            paths = [RESULTS / "sweep.csv"]
    if not paths:
        print("no shard CSVs found", file=sys.stderr)
        return 2
    for p in paths:
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2

    rows: list[dict] = []
    seen: dict[tuple, str] = {}
    dupes = []
    for p in paths:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                k = _key(r)
                if k in seen:
                    dupes.append((k, seen[k], str(p)))
                    continue
                seen[k] = str(p)
                rows.append(r)

    expected = {_prod_key(pt): pt for pt in SB.iter_production_points()}
    missing = [pt for k, pt in expected.items() if k not in seen]

    ns.out.parent.mkdir(parents=True, exist_ok=True)
    with open(ns.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in HEADER})

    print(f"merged {len(paths)} shard(s) → {ns.out}")
    print(f"  rows written: {len(rows)}")
    print(f"  duplicates rejected: {len(dupes)}")
    print(f"  missing vs production grid: {len(missing)} / {len(expected)}")
    if dupes:
        print("  first duplicates:")
        for k, a, b in dupes[:10]:
            print(f"    {k} in {a} and {b}")
    if missing:
        print("  first missing:")
        for pt in missing[:10]:
            print(f"    N={pt.N} sigma_w={pt.sigma_w} x={pt.x} i={pt.i} "
                  f"split={pt.split}")
    return 1 if dupes or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
