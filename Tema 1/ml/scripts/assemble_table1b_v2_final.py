"""Assemble Table 1b v2 final CSV + P-grid + summary from CV/FA/GNS metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml.train import _resolve


def _rms_mm(mse_m2: float) -> float:
    if mse_m2 is None or not np.isfinite(mse_m2):
        return float("nan")
    return 1000.0 * math.sqrt(max(float(mse_m2), 0.0))


def _ttf_str(ttf: dict) -> str:
    if not ttf:
        return ""
    med = ttf.get("median", float("nan"))
    q25 = ttf.get("q25", float("nan"))
    q75 = ttf.get("q75", float("nan"))
    return f"{med:.3g} ({q25:.3g}–{q75:.3g})"


def rows_from_metrics(model: str, metrics: dict) -> list[dict]:
    out = []
    for sp, m in (metrics.get("splits") or {}).items():
        he = m.get("horizon_errors") or {}
        phys = m.get("physical_fraction_at_times") or {}
        out.append({
            "model": model,
            "split": sp,
            "one_step": m.get("one_step_mse"),
            "rms_0.2s_mm": _rms_mm((he.get("0.2s") or {}).get("rel_mse_mean")),
            "rms_1s_mm": _rms_mm((he.get("1s") or {}).get("rel_mse_mean")),
            "rms_5s_mm": _rms_mm((he.get("5s") or {}).get("rel_mse_mean")),
            "rms_10s_mm": _rms_mm((he.get("10s") or {}).get("rel_mse_mean")),
            "rms_spider_0.2s_mm": _rms_mm((he.get("0.2s") or {}).get("spider_abs_mse_mean")),
            "rms_spider_1s_mm": _rms_mm((he.get("1s") or {}).get("spider_abs_mse_mean")),
            "rms_spider_5s_mm": _rms_mm((he.get("5s") or {}).get("spider_abs_mse_mean")),
            "rms_spider_10s_mm": _rms_mm((he.get("10s") or {}).get("spider_abs_mse_mean")),
            "ttf_median_q25_q75": _ttf_str(m.get("ttf") or {}),
            "phys_frac_1s": phys.get("1s", phys.get("1.0")),
            "phys_frac_5s": phys.get("5s", phys.get("5.0")),
            "abs_dP_mean": m.get("abs_delta_P"),
            "abs_dP_max": m.get("abs_delta_P_max"),
            "d_R_over_L_pct": (
                100.0 * m["delta_R_over_L"]
                if m.get("delta_R_over_L") is not None and np.isfinite(m["delta_R_over_L"])
                else float("nan")
            ),
            "speedup": m.get("speedup"),
            "surr_hardware": m.get("surr_hardware"),
            "sim_hardware": m.get("sim_hardware"),
        })
    return out


def _load_metrics(candidates: list[Path]) -> dict | None:
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pipeline-wall-s", type=float, default=float("nan"))
    ap.add_argument("--gpu-usd-per-hour", type=float, default=0.853624312)
    args = ap.parse_args()
    out = args.out_dir if args.out_dir.is_absolute() else _resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = _resolve(Path("results/ml"))

    sources = {
        "CV": [
            root / "_baseline_const_cpu_paper" / "metrics.json",
            root / "paper_report_pull" / "constant_velocity" / "runs" / "_baseline_const_cpu_paper" / "eval" / "metrics.json",
        ],
        "FA": [
            root / "_baseline_follow_air_cpu_paper" / "metrics.json",
            root / "paper_report_pull" / "follow_air" / "runs" / "_baseline_follow_air_cpu_paper" / "eval" / "metrics.json",
        ],
        "gns_local": [
            root / "gns_local_s42_88dc163f_1791760366" / "metrics.json",
            root / "runs" / "gns_local_s42_88dc163f_1791760366" / "metrics.json",
            root / "paper_report_pull" / "local" / "runs" / "gns_local_s42_88dc163f_1791760366" / "eval" / "metrics.json",
        ],
        "gns_baseline_k1": [
            root / "gns_baseline_s42_5716ee5b_1791828699" / "metrics.json",
            root / "runs" / "gns_baseline_s42_5716ee5b_1791828699" / "metrics.json",
            root / "paper_report_pull" / "baseline" / "runs" / "gns_baseline_s42_5716ee5b_1791828699" / "eval" / "metrics.json",
        ],
    }

    # Optional winning k ≠ 1
    win_path = root / "logs" / "table1b_final" / "winner_k.txt"
    win_ckpt = root / "logs" / "table1b_final" / "winner_ckpt.txt"
    if win_path.exists():
        wk = float(win_path.read_text().strip())
        if abs(wk - 1.0) > 1e-9 and win_ckpt.exists():
            ckpt = Path(win_ckpt.read_text().strip())
            rid = ckpt.parent.name
            sources[f"gns_baseline_k{wk:g}"] = [
                root / rid / "metrics.json",
                root / "runs" / rid / "metrics.json",
            ]

    all_rows = []
    p_frames = []
    used = {}
    for model, cands in sources.items():
        m = _load_metrics(cands)
        if m is None:
            used[model] = None
            continue
        used[model] = str(next(p for p in cands if p.exists()))
        all_rows.extend(rows_from_metrics(model, m))
        # P-grid if present beside metrics
        for c in cands:
            pg = c.parent / "tab_P_grid.csv"
            if pg.exists():
                df = pd.read_csv(pg)
                df.insert(0, "model", model)
                p_frames.append(df)
                break

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "table1b_v2_final.csv", index=False)
    if p_frames:
        pd.concat(p_frames, ignore_index=True).to_csv(out / "tab_P_grid_all_models.csv", index=False)
        for model, g in pd.concat(p_frames, ignore_index=True).groupby("model"):
            g.to_csv(out / f"tab_P_grid_{model}.csv", index=False)

    sel = {}
    sel_path = root / "logs" / "table1b_final" / "noise_k_selection_table.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text())

    wall = float(args.pipeline_wall_s)
    gpu_h = wall / 3600.0 if np.isfinite(wall) else float("nan")
    cost = gpu_h * float(args.gpu_usd_per_hour) if np.isfinite(gpu_h) else float("nan")
    lines = [
        "# Table 1b v2 final — summary",
        "",
        f"- pipeline_wall_s: {wall}",
        f"- gpu_hours: {gpu_h:.2f}",
        f"- approx_gpu_cost_usd (@${args.gpu_usd_per_hour}/h): {cost:.2f}",
        "",
        "## Metrics sources",
        "",
    ]
    for k, v in used.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## noise_k selection", ""]
    if sel:
        lines.append("```json")
        lines.append(json.dumps(sel.get("winner"), indent=2))
        lines.append("```")
        lines.append("")
        lines.append("| k | selected | TTF med | capped 5s | never-fail |")
        lines.append("|---|---|---|---|---|")
        for r in sel.get("rows", []):
            lines.append(
                f"| {r.get('k')} | `{r.get('selected')}` | {r.get('ttf_median')} | "
                f"{r.get('capped_5s_mse')} | {r.get('never_fail_frac')} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'table1b_v2_final.csv'}")
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
