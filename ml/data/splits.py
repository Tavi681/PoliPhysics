"""Run-level splits (never by frame)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _assign_stratified(df: pd.DataFrame, frac: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    pick = pd.Series(False, index=df.index)
    for _, g in df.groupby(["N", "sigma_w", "x_grid"], sort=False):
        idx = g.index.to_numpy()
        n = len(idx)
        k = int(round(frac * n))
        # Only force a hold-out when the stratum is large enough; singletons
        # stay in the residual pool (train) so tiny local subsets remain usable.
        if frac > 0 and n >= 5 and k == 0:
            k = 1
        if frac < 1.0 and n > 1 and k >= n:
            k = n - 1
        if k <= 0:
            continue
        chosen = rng.choice(idx, size=min(k, n), replace=False)
        pick.loc[chosen] = True
    return pick


def build_splits(index: pd.DataFrame, split_cfg: dict[str, Any]) -> dict[str, list[str]]:
    seed = int(split_cfg.get("seed", 42))
    df = index.copy()
    assert bool(df["valid"].all())
    assigned: dict[str, str] = {}

    def assign(mask: pd.Series, name: str) -> None:
        for rid in df.loc[mask, "run_id"]:
            if rid in assigned:
                raise RuntimeError(f"{rid} already in {assigned[rid]}, cannot add to {name}")
            assigned[rid] = name

    cfg_uN = split_cfg["test_unseenN"]
    m = df["N"].isin(cfg_uN["N"])
    if cfg_uN.get("sources"):
        m &= df["split_source"].isin(cfg_uN["sources"])
    assign(m, "test_unseenN")

    cfg_us = split_cfg["test_unseenSigma"]
    m = df["sigma_w"].isin(cfg_us["sigma_w"]) & ~df["run_id"].isin(assigned)
    assign(m, "test_unseenSigma")

    pool_cfg = split_cfg["train_val_pool"]
    m_pool = (
        df["N"].isin(pool_cfg["N"])
        & df["sigma_w"].isin(pool_cfg["sigma_w"])
        & ~df["run_id"].isin(assigned)
    )
    if pool_cfg.get("sources"):
        m_pool &= df["split_source"].isin(pool_cfg["sources"])
    pool = df.loc[m_pool]

    test_frac = float(split_cfg["test_interp_frac"])
    val_frac = float(split_cfg["val_frac"])
    m_test = _assign_stratified(pool, test_frac, seed + 1)
    assign(m_pool & m_test.reindex(df.index, fill_value=False), "test_interp")

    rem = pool.loc[~m_test.reindex(pool.index, fill_value=False)]
    rem_frac = val_frac / max(1e-12, 1.0 - test_frac)
    m_val = _assign_stratified(rem, rem_frac, seed + 2)
    for rid in rem.loc[m_val.reindex(rem.index, fill_value=False), "run_id"]:
        assigned[rid] = "val"
    for rid in rem.loc[~m_val.reindex(rem.index, fill_value=False), "run_id"]:
        assigned[rid] = "train"

    out: dict[str, list[str]] = {
        k: [] for k in ["train", "val", "test_interp", "test_unseenN", "test_unseenSigma"]
    }
    for rid, sp in assigned.items():
        out.setdefault(sp, []).append(rid)
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for rid in ids:
            if rid in seen:
                raise AssertionError(f"leakage: {rid} in {seen[rid]} and {name}")
            seen[rid] = name


def splits_hash(splits: dict[str, list[str]]) -> str:
    payload = json.dumps({k: splits[k] for k in sorted(splits)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_split_manifest(
    splits: dict[str, list[str]],
    index: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert_no_leakage(splits)
    h = splits_hash(splits)
    manifest = {"splits_hash": h, "splits": splits}
    with open(out_dir / "splits.json", "w") as f:
        json.dump(manifest, f, indent=2)

    id2 = index.set_index("run_id")
    rows = []
    for sp, ids in splits.items():
        if not ids:
            continue
        sub = id2.loc[list(ids)].copy()
        if "q_nC" not in sub.columns:
            sub["q_nC"] = sub["q"] * 1e9 if sub["q"].abs().max() < 1e-3 else sub["q"]
        for (N, qn, sw), g in sub.groupby(["N", "q_nC", "sigma_w"], sort=True):
            rows.append({
                "Split": sp,
                "N": int(N),
                "q (nC)": float(qn),
                "sigma_w (m/s)": float(sw),
                "Runs": int(len(g)),
                "Frames": int(g["n_frames"].sum()),
            })
    pd.DataFrame(rows).to_csv(out_dir / "tab_mlsplit.csv", index=False)
    return manifest


def load_splits(path: str | Path) -> dict[str, list[str]]:
    with open(path) as f:
        m = json.load(f)
    return m["splits"] if "splits" in m else m
