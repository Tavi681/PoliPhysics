"""Assemble Table 1b from CV/FA eval_v3 per-run JSON and existing GNS rows.

Recomputes takeoff P on (split, N, sigma_w, x) for constant velocity and
follow-air. GNS |ΔP| is read from the existing by-x CSVs. Does not launch
rollouts.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ml.metrics.physics import wilson_interval
from ml.metrics.rollout_metrics import parse_x_from_run_id
from ml.train import _resolve

RUN_RE = re.compile(r"^(N\d+_sw[0-9.]+_x[+-]?\d+_i\d+)\.json$")
HEAD_STATUS = re.compile(r'"status": "(\w+)"')
HEAD_SIM = re.compile(r'"outcome_sim": "(\w+)"')
EXPECTED = {
    "test_interp": 590,
    "test_unseenN_interp": 150,
    "test_unseenN": 1850,
    "test_unseenSigma": 300,
}
MODEL_LABEL = {
    "CV": "constant_velocity",
    "FA": "follow_air",
    "gns_local": "gns_local",
    "gns_baseline": "gns_baseline_k1",
}


def head(path: Path) -> tuple[str, str]:
    raw = path.read_text()[:240]
    st = HEAD_STATUS.search(raw)
    sim = HEAD_SIM.search(raw)
    if not st or not sim:
        raise RuntimeError(f"missing status/outcome_sim in {path}")
    return st.group(1), sim.group(1)


def cell_key(run_id: str) -> tuple[int, float, int]:
    n_s, rest = run_id.split("_sw", 1)
    sw_s, _ = rest.split("_x", 1)
    return int(n_s[1:]), float(sw_s), parse_x_from_run_id(run_id)


def expected_counts(splits_path: Path) -> dict[str, dict[tuple[int, float, int], int]]:
    raw = json.loads(splits_path.read_text())
    splits = raw["splits"] if isinstance(raw, dict) and "splits" in raw else raw
    out = {}
    for sp, ids in splits.items():
        if sp not in EXPECTED:
            continue
        cells: dict[tuple[int, float, int], int] = defaultdict(int)
        for rid in ids:
            cells[cell_key(rid)] += 1
        out[sp] = dict(cells)
    return out


def recompute_model(
    model: str,
    per_dir: Path,
    out_csv: Path,
    expect: dict[str, dict[tuple[int, float, int], int]],
) -> dict[str, dict]:
    rows = []
    totals: dict[str, int] = defaultdict(int)
    got_cells: dict[str, dict[tuple[int, float, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for path in sorted(per_dir.rglob("*.json")):
        if path.name.endswith("_.gstmp"):
            continue
        m = RUN_RE.match(path.name)
        if not m:
            raise RuntimeError(f"unexpected name {path}")
        rid = m.group(1)
        split = path.parent.name
        status, outcome = head(path)
        n_nodes, sw, x = cell_key(rid)
        got_cells[split][(n_nodes, sw, x)] += 1
        totals[split] += 1
        rows.append({
            "model": model,
            "split": split,
            "run_id": rid,
            "N": n_nodes,
            "sigma_w": sw,
            "x": x,
            "status": status,
            "outcome_sim": outcome,
        })
    if not rows:
        raise RuntimeError(f"no per-run JSON under {per_dir}")
    frame = pd.DataFrame(rows)
    out_rows = []
    by_split: dict[str, list[float]] = defaultdict(list)
    for (split, n_nodes, sw, x), g in frame.groupby(
        ["split", "N", "sigma_w", "x"], sort=True
    ):
        n = int(len(g))
        ks = int((g["outcome_sim"] == "rise").sum())
        km = int((g["status"] == "rise").sum())
        lo_s, hi_s = wilson_interval(ks, n)
        lo_m, hi_m = wilson_interval(km, n)
        p_sim, p_surr = ks / n, km / n
        out_rows.append({
            "model": model,
            "split": split,
            "N": int(n_nodes),
            "sigma_w": float(sw),
            "x": int(x),
            "n_runs": n,
            "P_sim": p_sim,
            "P_surr": p_surr,
            "P_sim_wilson_lo": lo_s,
            "P_sim_wilson_hi": hi_s,
            "P_surr_wilson_lo": lo_m,
            "P_surr_wilson_hi": hi_m,
            "abs_delta_P": abs(p_surr - p_sim),
            "p_usable": True,
        })
        by_split[split].append(abs(p_surr - p_sim))
    mismatches = []
    for sp, n_exp in EXPECTED.items():
        if totals[sp] != n_exp:
            mismatches.append(f"{model} {sp} sum {totals[sp]} != {n_exp}")
        if got_cells[sp].keys() != expect[sp].keys() or any(
            got_cells[sp][k] != expect[sp][k] for k in expect[sp]
        ):
            mismatches.append(f"{model} {sp} cell counts differ")
    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    return {
        sp: {"mean": float(sum(v) / len(v)), "max": float(max(v)), "n_points": len(v)}
        for sp, v in by_split.items()
    }


def delta_from_csv(path: Path) -> dict[str, dict]:
    frame = pd.read_csv(path)
    return {
        sp: {
            "mean": float(g["abs_delta_P"].mean()),
            "max": float(g["abs_delta_P"].max()),
            "n_points": int(len(g)),
        }
        for sp, g in frame.groupby("split")
    }


def rms_mm(mse) -> float:
    if mse is None or not math.isfinite(float(mse)):
        return float("nan")
    return 1000.0 * math.sqrt(max(float(mse), 0.0))


def ttf_str(ttf: dict) -> str:
    return (
        f"{ttf.get('median', float('nan')):.3g} "
        f"({ttf.get('q25', float('nan')):.3g}–{ttf.get('q75', float('nan')):.3g})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--final-dir",
        type=Path,
        default=Path("results/ml/paper_report_pull/final_v2"),
    )
    ap.add_argument("--splits", type=Path, required=True)
    args = ap.parse_args()
    root = args.final_dir if args.final_dir.is_absolute() else _resolve(args.final_dir)
    splits_path = args.splits if args.splits.is_absolute() else _resolve(args.splits)
    pdir = root / "P"
    metrics = root / "metrics"
    expect = expected_counts(splits_path)

    old_cv_path = pdir / "tab_P_grid_CV_by_x.csv"
    old_cv = delta_from_csv(old_cv_path) if old_cv_path.exists() else {}
    delta_cv = recompute_model(
        "CV",
        root / "eval_v3" / "constant_velocity" / "per_run",
        pdir / "tab_P_grid_CV_v3_by_x.csv",
        expect,
    )
    delta_fa = recompute_model(
        "FA",
        root / "eval_v3" / "follow_air" / "per_run",
        pdir / "tab_P_grid_FA_v3_by_x.csv",
        expect,
    )
    delta_gns = {
        "gns_local": delta_from_csv(pdir / "tab_P_grid_gns_local_by_x.csv"),
        "gns_baseline": delta_from_csv(pdir / "tab_P_grid_gns_baseline_by_x.csv"),
    }
    delta = {
        "CV": delta_cv,
        "FA": delta_fa,
        "gns_local": delta_gns["gns_local"],
        "gns_baseline": delta_gns["gns_baseline"],
    }
    (root / "delta_P_by_split.json").write_text(
        json.dumps(
            {
                "old_CV": old_cv,
                "new_CV": delta_cv,
                "new_FA": delta_fa,
                "gns_local": delta_gns["gns_local"],
                "gns_baseline": delta_gns["gns_baseline"],
            },
            indent=2,
        )
        + "\n"
    )

    metric_files = {
        "CV": metrics / "CV_v3.json",
        "FA": metrics / "FA_v3.json",
        "gns_local": metrics / "gns_local.json",
        "gns_baseline": metrics / "gns_baseline.json",
    }
    rows = []
    for model in ("CV", "FA", "gns_local", "gns_baseline"):
        blob = json.loads(metric_files[model].read_text())
        label = MODEL_LABEL[model]
        for sp, m in blob["splits"].items():
            he = m.get("horizon_errors") or {}
            ttf = m.get("ttf") or {}
            phys = m.get("physical_fraction_at_times") or {}
            d_p = delta[model].get(sp, {})
            n_never = ttf.get("n_never_fail", 0)
            n = ttf.get("n_runs") or m.get("n_runs") or 1
            hw = m.get("surr_hardware")
            if model in ("gns_local", "gns_baseline") and hw == "cuda":
                hw = "NVIDIA L4 (cuda)"
            rows.append({
                "model": label,
                "split": sp,
                "one_step_mse": m.get("one_step_mse"),
                "rms_thread_0.2s_mm": rms_mm((he.get("0.2s") or {}).get("rel_mse_mean")),
                "rms_thread_1s_mm": rms_mm((he.get("1s") or {}).get("rel_mse_mean")),
                "rms_thread_5s_mm": rms_mm((he.get("5s") or {}).get("rel_mse_mean")),
                "rms_thread_10s_mm": rms_mm((he.get("10s") or {}).get("rel_mse_mean")),
                "rms_spider_0.2s_mm": rms_mm((he.get("0.2s") or {}).get("spider_abs_mse_mean")),
                "rms_spider_1s_mm": rms_mm((he.get("1s") or {}).get("spider_abs_mse_mean")),
                "rms_spider_5s_mm": rms_mm((he.get("5s") or {}).get("spider_abs_mse_mean")),
                "rms_spider_10s_mm": rms_mm((he.get("10s") or {}).get("spider_abs_mse_mean")),
                "ttf_median_q25_q75": ttf_str(ttf),
                "never_fail_fraction": n_never / max(int(n), 1),
                "phys_frac_1s": phys.get("1s", phys.get("1.0")),
                "phys_frac_5s": phys.get("5s", phys.get("5.0")),
                "abs_delta_P_mean": d_p.get("mean"),
                "abs_delta_P_max": d_p.get("max"),
                "delta_R_over_L_pct": (
                    100.0 * m["delta_R_over_L"]
                    if m.get("delta_R_over_L") is not None
                    and math.isfinite(m["delta_R_over_L"])
                    else float("nan")
                ),
                "speedup": m.get("speedup"),
                "speedup_scope": (
                    "per split, not per N (simulator wall time is not stored per run)"
                ),
                "sim_hardware": m.get("sim_hardware"),
                "surr_hardware": hw,
            })
    out_csv = root / "table1b_v2_final.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("OLD CV |dP|", json.dumps(old_cv))
    print("NEW CV |dP|", json.dumps(delta_cv))
    print("NEW FA |dP|", json.dumps(delta_fa))
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
