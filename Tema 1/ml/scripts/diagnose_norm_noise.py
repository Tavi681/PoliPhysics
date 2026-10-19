"""Diagnose a_std / a_by_type / window scale consistency (CPU; safe offline)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.data.dataset import BallooningRunStore
from ml.data.frames import DT
from ml.data.noise_stats import a_std_by_type_matrix, resolve_noise
from ml.data.splits import load_splits
from ml.train import _resolve
from ml.utils.config import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(_resolve(args.config) if not args.config.is_absolute() else args.config)
    dcfg = cfg["data"]
    data_dir = Path(dcfg["data_dir"]) if Path(dcfg["data_dir"]).is_absolute() else _resolve(dcfg["data_dir"])
    splits = load_splits(
        Path(dcfg["splits"]) if Path(dcfg["splits"]).is_absolute() else _resolve(dcfg["splits"])
    )
    store = BallooningRunStore(data_dir, cache_shards=2, mmap=True)
    train_ids = list(splits["train"])
    if args.max_runs:
        train_ids = train_ids[: args.max_runs]

    C = int(dcfg.get("C", 5))
    sum_a = np.zeros(3)
    sum_a2 = np.zeros(3)
    n_a = 0
    type_sum = {t: np.zeros(3) for t in range(3)}
    type_sum2 = {t: np.zeros(3) for t in range(3)}
    type_n = {t: 0 for t in range(3)}
    max_dv = 0.0
    max_dv_where = None
    max_dv_ok = 0.0
    max_dv_ok_where = None

    for rid in train_ids:
        rec = store.get(rid)
        v = np.asarray(rec["v"], dtype=np.float64)
        nt = np.asarray(rec["node_type"], dtype=np.int64)
        ok = np.asarray(rec["window_ok"], dtype=bool)
        T = v.shape[0]
        for t in range(C, T - 1):
            dv = v[t + 1] - v[t]
            a = dv / DT
            flat = float(np.max(np.abs(dv)))
            if flat > max_dv:
                max_dv = flat
                max_dv_where = {"run_id": rid, "t": t, "window_ok": bool(ok[t]), "dv_max": flat}
            rows = a.reshape(-1, 3)
            if not ok[t]:
                continue
            if flat > max_dv_ok:
                max_dv_ok = flat
                max_dv_ok_where = {"run_id": rid, "t": t, "dv_max": flat}
            sum_a += rows.sum(0)
            sum_a2 += (rows ** 2).sum(0)
            n_a += rows.shape[0]
            for typ in range(3):
                m = nt == typ
                if not np.any(m):
                    continue
                r = a[m].reshape(-1, 3)
                type_sum[typ] += r.sum(0)
                type_sum2[typ] += (r ** 2).sum(0)
                type_n[typ] += r.shape[0]

    def finalize(s, s2, n):
        mean = s / max(n, 1)
        var = s2 / max(n, 1) - mean ** 2
        return mean.tolist(), np.sqrt(np.maximum(var, 0)).tolist(), int(n)

    mean_ok, std_ok, n_ok = finalize(sum_a, sum_a2, n_a)
    by_type = {}
    for typ, name in enumerate(("spider", "interior", "tip")):
        m, s, n = finalize(type_sum[typ], type_sum2[typ], type_n[typ])
        by_type[name] = {"a_mean": m, "a_std": s, "dv_std": (np.asarray(s) * DT).tolist(), "count": n}

    norm = store.norm["a"]
    a_std_norm = np.asarray(norm["std"], dtype=np.float64)
    a_type = a_std_by_type_matrix(store.norm)
    mode = str(dcfg.get("noise_mode", "per_type"))
    scalar, mat, comp = resolve_noise(
        noise_k=float(dcfg.get("noise_k", 1.0)),
        noise_std=None,
        noise_mode=mode,
        a_std=a_std_norm,
        a_std_by_type=a_type,
        dt=DT,
    )
    report = {
        "legacy_noise_stats_cause": (
            "Old noise_stats sampled linspace(0, T-2) including t<5 rest→air "
            "transients (window_ok=false) + per-run 200-frame cap overweighting "
            "short violent runs + mean-of-type-RMS. Factors ~3.8×2.3×1.1×0.63≈6.1 "
            "vs ||a_std||_rms*dt. Deleted from code path and data version."
        ),
        "convention": {
            "a_mean_a_std": "global per-component (target + decoder)",
            "a_by_type": "per-type per-component (noise scale only when noise_mode=per_type)",
            "noise_mode": mode,
        },
        "recomputed_window_ok": {
            "a_mean": mean_ok,
            "a_std": std_ok,
            "dv_rms": float(np.sqrt(np.mean((np.asarray(std_ok) * DT) ** 2))),
            "count": n_ok,
            "by_type": by_type,
        },
        "norm_stats_a": {
            "mean": np.asarray(norm["mean"]).tolist(),
            "std": a_std_norm.tolist(),
            "dv_rms_implied": float(np.sqrt(np.mean((a_std_norm * DT) ** 2))),
        },
        "norm_stats_a_by_type": (
            None
            if a_type is None
            else {n: a_type[i].tolist() for i, n in enumerate(("spider", "interior", "tip"))}
        ),
        "resolved_noise": {
            "scalar_rms": scalar,
            "per_type_comp": None if mat is None else mat.tolist(),
            "global_comp": None if comp is None else comp.tolist(),
        },
        "max_dv_any_frame": max_dv_where,
        "max_dv_window_ok": max_dv_ok_where,
        "noise_stats_present": (data_dir / "noise_stats.json").exists(),
    }
    out = Path(args.out) if Path(args.out).is_absolute() else _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
