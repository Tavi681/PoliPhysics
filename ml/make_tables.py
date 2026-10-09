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
        default=repo_root() / "Tema 1" / "results" / "ml",
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
            # fill C_data from metrics if present
            if meta_p.exists():
                meta = json.loads(meta_p.read_text())
                # C_data = sum wall_time of training runs — optional field
                c_data = meta.get("C_data", float("nan"))
                c_train = meta.get("C_train", df["C_train"].iloc[0] if "C_train" in df else float("nan"))
                t_sim = meta.get("t_sim_mean", float("nan"))
                t_surr = df["Time per run"].iloc[0] if len(df) else float("nan")
                if np.isfinite(t_sim) and np.isfinite(t_surr) and (t_sim - t_surr) > 0:
                    qstar = (float(c_data) + float(c_train)) / (t_sim - t_surr) if np.isfinite(c_data) else float("nan")
                else:
                    qstar = float("nan")
                df["C_data"] = c_data
                df["C_train"] = c_train
                df["Q*"] = qstar
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
