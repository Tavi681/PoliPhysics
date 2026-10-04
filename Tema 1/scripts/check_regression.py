"""Regression gate: compare two directories of produce_results CSVs.

Default (Stage A gate):
    results/*.csv vs results/pre_stageA/*.csv, tol 1e-12.

Stage B R2 / R3 (via flags or run_regression_cloud.py):
    R2  old vs new on the same VM, PASS < 1e-12
    R3  cloud new vs baseline_mac/, PASS < 1e-6
    On fail print (file, row, col) and exit non-zero.

Usage:
  python scripts/check_regression.py
  python scripts/check_regression.py --baseline DIR --current DIR --tol 1e-6
  python scripts/check_regression.py --baseline DIR --current DIR --tol 1e-12 \\
      --out results/regression.txt --label R2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE = RESULTS / "pre_stageA"

DEFAULT_TOL = 1e-12
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


def compare_file(base: Path, cur: Path, tol: float = DEFAULT_TOL) -> dict:
    hb, rb = _read_csv(base)
    hc, rc = _read_csv(cur)
    if hb != hc:
        return {"ok": False, "reason": f"header mismatch: {hb} != {hc}",
                "file": base.name, "row": None, "col": None, "max_rel": None}
    if len(rb) != len(rc):
        return {"ok": False, "reason": f"row count {len(rb)} != {len(rc)}",
                "file": base.name, "row": None, "col": None, "max_rel": None}
    max_rel = 0.0
    worst = None
    worst_loc = (None, None)
    text_mismatch = None
    text_loc = (None, None)
    for i, (row_b, row_c) in enumerate(zip(rb, rc)):
        for j, col in enumerate(hb):
            if col in TIME_COLUMNS:
                continue
            vb, vc = row_b[j], row_c[j]
            fb, fc = _to_float(vb), _to_float(vc)
            if fb is None or fc is None:
                if vb != vc and text_mismatch is None:
                    text_mismatch = f"row {i} col {col!r}: {vb!r} != {vc!r}"
                    text_loc = (i, col)
                continue
            denom = abs(fb) if abs(fb) > 0 else 1.0
            rel = abs(fb - fc) / denom
            if rel > max_rel:
                max_rel = rel
                worst = f"row {i} col {col!r}: {fb!r} vs {fc!r} (rel {rel:.2e})"
                worst_loc = (i, col)
    ok = (max_rel < tol) and (text_mismatch is None)
    row, col = (text_loc if text_mismatch else worst_loc)
    return {
        "ok": ok, "max_rel": max_rel, "worst": worst,
        "text_mismatch": text_mismatch,
        "file": base.name, "row": row, "col": col,
        "reason": None if ok else (text_mismatch or worst
                                   or f"max_rel={max_rel:.2e}"),
    }


def compare_directories(baseline: Path, current: Path, tol: float):
    """Compare every baseline CSV to the same name in current.

    Returns ``(all_ok, lines, failures)`` where each failure includes
    ``(file, row, col)``.
    """
    lines = []
    failures = []
    all_ok = True
    bases = sorted(baseline.glob("*.csv"))
    if not bases:
        return False, [f"no CSVs in {baseline}"], [
            {"file": str(baseline), "row": None, "col": None,
             "reason": "no CSVs"}
        ]
    for base in bases:
        cur = current / base.name
        if not cur.exists():
            all_ok = False
            lines.append(f"[MISSING] {base.name}: not in {current}")
            failures.append({"file": base.name, "row": None, "col": None,
                             "reason": "missing in current"})
            continue
        res = compare_file(base, cur, tol=tol)
        if not res["ok"]:
            all_ok = False
            reason = res.get("reason") or res.get("text_mismatch") or \
                f"max_rel={res.get('max_rel'):.2e} ({res.get('worst')})"
            loc = f"  (file={res['file']}, row={res['row']}, col={res['col']})"
            lines.append(f"[FAIL]  {base.name}: {reason}{loc}")
            failures.append(res)
        else:
            lines.append(f"[OK]    {base.name}: max_rel={res['max_rel']:.2e}")
    return all_ok, lines, failures


def format_report(title: str, baseline: Path, current: Path, tol: float,
                  all_ok: bool, lines: list[str]) -> str:
    head = [
        title,
        f"baseline={baseline}",
        f"current ={current}",
        f"tol={tol:g}  (timing columns excluded: {sorted(TIME_COLUMNS)})",
        "",
    ]
    tail = [
        "",
        "VERDICT: " + ("PASS - all CSVs within tol"
                       if all_ok else "FAIL - see (file, row, col) above"),
        "",
    ]
    return "\n".join(head + lines + tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--current", type=Path, default=RESULTS)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--out", type=Path, default=RESULTS / "regression.txt")
    ap.add_argument("--label", default="Regression check")
    ns = ap.parse_args()

    if not ns.baseline.exists():
        print(f"no baseline at {ns.baseline}", file=sys.stderr)
        return 2
    all_ok, lines, failures = compare_directories(ns.baseline, ns.current, ns.tol)
    text = format_report(ns.label, ns.baseline, ns.current, ns.tol,
                         all_ok, lines)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(text)
    print(text)
    if not all_ok:
        print("FAIL locations (file, row, col):", file=sys.stderr)
        for f in failures:
            print(f"  ({f.get('file')}, {f.get('row')}, {f.get('col')})",
                  file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
