"""Aggregate all results/ml/<run_id>/ into paper comparison tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.utils.git_info import repo_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=repo_root() / "results" / "ml",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs_dir = args.runs_dir
    out = args.out or (runs_dir / "tables")
    out.mkdir(parents=True, exist_ok=True)

    acc_rows = []
    cost_rows = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name == "tables":
            continue
        acc = d / "tab_accuracy.csv"
        cost = d / "tab_cost.csv"
        meta_p = d / "metrics.json"
        if acc.exists():
            df = pd.read_csv(acc)
            df["run_id"] = d.name
            acc_rows.append(df)
        if cost.exists():
            df = pd.read_csv(cost)
            df["run_id"] = d.name
            # Prefer evaluate.py metrics (duration-normalized Q*)
            if meta_p.exists():
                meta = json.loads(meta_p.read_text())
                df["C_data"] = meta.get("C_data", float("nan"))
                df["C_train"] = meta.get("C_train", float("nan"))
                df["Q*"] = meta.get("Q_star", float("nan"))
                if "t_sim_per_run" in meta:
                    df["t_sim_per_run"] = meta["t_sim_per_run"]
                if "t_surr_per_run" in meta:
                    df["t_surr_per_run"] = meta["t_surr_per_run"]
            cost_rows.append(df)

    if acc_rows:
        tab = pd.concat(acc_rows, ignore_index=True)
        tab.to_csv(out / "tab_accuracy_all.csv", index=False)
        print(tab.to_string(index=False))
    if cost_rows:
        tabc = pd.concat(cost_rows, ignore_index=True)
        tabc.to_csv(out / "tab_cost_all.csv", index=False)
        print(tabc.to_string(index=False))
    print(f"wrote tables → {out}")


if __name__ == "__main__":
    main()
