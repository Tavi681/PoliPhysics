#!/usr/bin/env python
"""Build split manifests from ml_index/sweep + YAML config."""
from __future__ import annotations

import argparse
from pathlib import Path

from ml.data.preprocess import build_run_table
from ml.data.splits import build_splits, write_split_manifest
from ml.utils.config import load_yaml
from ml.utils.git_info import repo_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=repo_root() / "ml" / "configs" / "splits_default.yaml")
    ap.add_argument("--out", type=Path, default=repo_root() / "results" / "ml" / "splits")
    ap.add_argument("--runs-csv", type=Path, default=None, help="Optional pre-filtered run table")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.runs_csv:
        import pandas as pd
        df = pd.read_csv(args.runs_csv)
    else:
        df = build_run_table()
    splits = build_splits(df, cfg)
    man = write_split_manifest(splits, df, args.out)
    for k, v in splits.items():
        print(f"{k}: {len(v)}")
    print("splits_hash", man["splits_hash"])


if __name__ == "__main__":
    main()
