"""Cloud R1–R3: two checkouts, same venv, produce_results in parallel.

R1  old = 4ae88b1 (Before Stage A), new = uploaded HEAD tree.
    Run scripts/produce_results.py in each tree (separate output dirs / cores).
    produce_results.py is not modified; each tree writes its own results/.
R2  old vs new, max rel diff, timing cols excluded. PASS < 1e-12.
R3  new vs baseline_mac/. PASS < 1e-6. On fail print (file, row, col) and stop.

Usage (from the NEW Tema 1 tree, or with explicit dirs):
  python scripts/run_regression_cloud.py \\
      --old-dir /mnt/pd/tema1/old --new-dir /mnt/pd/tema1/new \\
      --baseline-mac /mnt/pd/tema1/new/baseline_mac
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from check_regression import compare_directories  # noqa: E402

OLD_COMMIT_DEFAULT = "4ae88b1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-dir", type=Path, required=True)
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--baseline-mac", type=Path, default=None)
    ap.add_argument("--workers-each", type=int, default=4)
    ap.add_argument("--skip-r1", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args()

    old = ns.old_dir.resolve()
    new = ns.new_dir.resolve()
    baseline = (ns.baseline_mac or (new / "baseline_mac")).resolve()
    out = ns.out or (new / "results" / "regression.txt")
    old_results = old / "results"
    new_results = new / "results"

    if not (old / "scripts" / "produce_results.py").exists():
        print(f"old produce_results missing: {old}", file=sys.stderr)
        return 2
    if not (new / "scripts" / "produce_results.py").exists():
        print(f"new produce_results missing: {new}", file=sys.stderr)
        return 2

    r1_ok = True
    if not ns.skip_r1:
        print(f"=== R1: produce_results.py old={old} new={new} "
              f"workers_each={ns.workers_each} ===", flush=True)
        t0 = time.perf_counter()
        log_old = new / "results"
        log_old.mkdir(parents=True, exist_ok=True)
        p_old = subprocess.Popen(
            [sys.executable, "-u", str(old / "scripts" / "produce_results.py"),
             "--workers", str(ns.workers_each)],
            cwd=str(old),
            stdout=open(new / "results" / "r1_old.log", "w"),
            stderr=subprocess.STDOUT,
        )
        p_new = subprocess.Popen(
            [sys.executable, "-u", str(new / "scripts" / "produce_results.py"),
             "--workers", str(ns.workers_each)],
            cwd=str(new),
            stdout=open(new / "results" / "r1_new.log", "w"),
            stderr=subprocess.STDOUT,
        )
        rc_old = p_old.wait()
        rc_new = p_new.wait()
        wall = time.perf_counter() - t0
        print(f"R1 done in {wall / 60:.1f} min  old_rc={rc_old} new_rc={rc_new}",
              flush=True)
        r1_ok = (rc_old == 0 and rc_new == 0)
        if not r1_ok:
            print("R1 FAIL: produce_results exited non-zero", flush=True)

    lines = [
        f"Stage B regression R1–R3  old_commit={OLD_COMMIT_DEFAULT}",
        f"old_dir={old}",
        f"new_dir={new}",
        f"R1 produce_results: {'PASS' if r1_ok else 'FAIL'}",
        "",
        "=== R2  old vs new (same VM), tol 1e-12 ===",
    ]
    r2_ok, r2_lines, r2_fail = compare_directories(old_results, new_results, 1e-12)
    lines.extend(r2_lines)
    lines.append("R2 VERDICT: " + ("PASS" if r2_ok else "FAIL"))
    lines.append("")
    lines.append("=== R3  cloud new vs baseline_mac, tol 1e-6 ===")
    if not baseline.exists():
        r3_ok = False
        r3_fail = [{"file": str(baseline), "row": None, "col": None,
                    "reason": "baseline_mac missing"}]
        lines.append(f"[MISSING] baseline_mac at {baseline}")
    else:
        r3_ok, r3_lines, r3_fail = compare_directories(baseline, new_results, 1e-6)
        lines.extend(r3_lines)
    lines.append("R3 VERDICT: " + ("PASS" if r3_ok else "FAIL"))
    lines.append("")
    all_ok = r1_ok and r2_ok and r3_ok
    lines.append("OVERALL: " + ("PASS" if all_ok else "FAIL"))
    if not r3_ok:
        lines.append("R3 FAIL locations (file, row, col):")
        for f in r3_fail:
            lines.append(f"  ({f.get('file')}, {f.get('row')}, {f.get('col')})"
                         f"  {f.get('reason', '')}")
        lines.append("Stop further work (do not launch B1/B2).")
    lines.append("")
    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, flush=True)
    (new_results / "R_DONE").write_text(
        f"r1={r1_ok}\nr2={r2_ok}\nr3={r3_ok}\noverall={all_ok}\n",
        encoding="utf-8",
    )
    if not r3_ok:
        print("R3 exceeded 1e-6. Locations:", file=sys.stderr)
        for f in r3_fail:
            print(f"  ({f.get('file')}, {f.get('row')}, {f.get('col')})",
                  file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
