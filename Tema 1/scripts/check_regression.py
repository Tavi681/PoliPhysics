"""Regression gate: compare results/*.csv against results/pre_stageA/*.csv.

Stage A must be additive: every baseline CSV produced before Stage A must be
reproduced by the post-Stage-A code. This compares each pre_stageA/*.csv with the
current results/*.csv column by column (numeric columns by max relative difference,
excluding any cpu/wall-time columns; text columns by exact equality) and writes the
verdict to results/regression.txt.

Usage:
  python scripts/produce_results.py            # regenerate results/*.csv first
  python scripts/check_regression.py           # then compare
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE = RESULTS / "pre_stageA"

TOL = 1e-12
# Columns excluded from the numeric comparison (timing is machine-dependent).
TIME_COLUMNS = {"cpu_s", "wall_s", "runtime_s", "cpu", "wall"}


def _read_csv(path: Path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def _to_float(s: str):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def compare_file(base: Path, cur: Path) -> dict:
    hb, rb = _read_csv(base)
    hc, rc = _read_csv(cur)
    if hb != hc:
        return {"ok": False, "reason": f"header mismatch: {hb} != {hc}"}
    if len(rb) != len(rc):
        return {"ok": False, "reason": f"row count {len(rb)} != {len(rc)}"}
    max_rel = 0.0
    worst = None
    text_mismatch = None
    for i, (row_b, row_c) in enumerate(zip(rb, rc)):
        for j, col in enumerate(hb):
            if col in TIME_COLUMNS:
                continue
            vb, vc = row_b[j], row_c[j]
            fb, fc = _to_float(vb), _to_float(vc)
            if fb is None or fc is None:
                if vb != vc and text_mismatch is None:
                    text_mismatch = f"row {i} col {col!r}: {vb!r} != {vc!r}"
                continue
            denom = abs(fb) if abs(fb) > 0 else 1.0
            rel = abs(fb - fc) / denom
            if rel > max_rel:
                max_rel = rel
                worst = f"row {i} col {col!r}: {fb!r} vs {fc!r} (rel {rel:.2e})"
    ok = (max_rel < TOL) and (text_mismatch is None)
    return {"ok": ok, "max_rel": max_rel, "worst": worst,
            "text_mismatch": text_mismatch}


def main() -> int:
    if not BASELINE.exists():
        print(f"no baseline at {BASELINE}", file=sys.stderr)
        return 2
    lines = [f"Regression check: results/*.csv vs pre_stageA/*.csv (tol {TOL:g})", ""]
    all_ok = True
    for base in sorted(BASELINE.glob("*.csv")):
        cur = RESULTS / base.name
        if not cur.exists():
            lines.append(f"[MISSING] {base.name}: not regenerated in results/")
            all_ok = False
            continue
        res = compare_file(base, cur)
        if not res["ok"]:
            all_ok = False
            reason = res.get("reason") or res.get("text_mismatch") or \
                f"max_rel={res.get('max_rel'):.2e} ({res.get('worst')})"
            lines.append(f"[FAIL]  {base.name}: {reason}")
        else:
            lines.append(f"[OK]    {base.name}: max_rel={res['max_rel']:.2e}")
    lines.append("")
    lines.append("VERDICT: " + ("PASS - all baseline CSVs reproduced within tol"
                                if all_ok else "FAIL - see above"))
    text = "\n".join(lines) + "\n"
    (RESULTS / "regression.txt").write_text(text)
    print(text)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
