"""Diagnose a_std / noise_stats / window_ok consistency (CPU; safe offline)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.data.dataset import BallooningRunStore
from ml.data.frames import DT
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

    # Accumulators: global + per type, only window_ok frames in [C, T-2]
    C = int(dcfg.get("C", 5))
    sum_a = np.zeros(3)
    sum_a2 = np.zeros(3)
    n_a = 0
    type_sum = {t: np.zeros(3) for t in range(3)}
    type_sum2 = {t: np.zeros(3) for t in range(3)}
    type_n = {t: 0 for t in range(3)}
    # Without window_ok (bad)
    sum_a_bad = np.zeros(3)
    sum_a2_bad = np.zeros(3)
    n_a_bad = 0
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
            # unfiltered
            rows = a.reshape(-1, 3)
            sum_a_bad += rows.sum(0)
            sum_a2_bad += (rows ** 2).sum(0)
            n_a_bad += rows.shape[0]
            if not ok[t]:
                continue
            if flat > max_dv_ok:
                max_dv_ok = flat
                max_dv_ok_where = {"run_id": rid, "t": t, "dv_max": flat}
            sum_a += rows.sum(0)
            sum_a2 += (rows ** 2).sum(0)
            n_a += rows.shape[0]
            for typ in (0, 1, 2):
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
    mean_bad, std_bad, n_bad = finalize(sum_a_bad, sum_a2_bad, n_a_bad)
    by_type = {}
    for typ, name in enumerate(("spider", "interior", "tip")):
        m, s, n = finalize(type_sum[typ], type_sum2[typ], type_n[typ])
        by_type[name] = {
            "a_mean": m,
            "a_std": s,
            "dv_std": (np.asarray(s) * DT).tolist(),
            "dv_rms": float(np.sqrt(np.mean((np.asarray(s) * DT) ** 2))),
            "count": n,
        }

    norm = store.norm["a"]
    noise_path = data_dir / "noise_stats.json"
    noise = json.loads(noise_path.read_text()) if noise_path.exists() else None

    a_std_norm = np.asarray(norm["std"], dtype=np.float64)
    report = {
        "convention": {
            "a_mean_a_std": "per-component global (all node types pooled); NOT per-type",
            "training_target": "normalize(a) with these a_mean/a_std",
            "noise_scale_fixed": "k * a_std * dt per component (matches normalizer)",
            "noise_scale_legacy_bug": (
                "k * noise_stats.by_type[type].rms — ~14x too large vs a_std*dt"
            ),
            "decoder": "pred_n * a_std + a_mean (same global per-component)",
        },
        "recomputed_window_ok": {
            "a_mean": mean_ok,
            "a_std": std_ok,
            "dv_std": (np.asarray(std_ok) * DT).tolist(),
            "dv_rms": float(np.sqrt(np.mean((np.asarray(std_ok) * DT) ** 2))),
            "count": n_ok,
            "by_type": by_type,
        },
        "recomputed_no_window_ok_filter": {
            "a_mean": mean_bad,
            "a_std": std_bad,
            "dv_rms": float(np.sqrt(np.mean((np.asarray(std_bad) * DT) ** 2))),
            "count": n_bad,
        },
        "norm_stats_a": {
            "mean": np.asarray(norm["mean"]).tolist(),
            "std": a_std_norm.tolist(),
            "dv_rms_implied": float(np.sqrt(np.mean((a_std_norm * DT) ** 2))),
        },
        "noise_stats": noise,
        "ratios": {
            "noise_overall_rms / (||a_std||_rms * dt)": (
                float(noise["overall_rms"]) / max(float(np.sqrt(np.mean((a_std_norm * DT) ** 2))), 1e-12)
                if noise else None
            ),
            "recomputed_dv_rms / norm_implied_dv_rms": (
                float(np.sqrt(np.mean((np.asarray(std_ok) * DT) ** 2)))
                / max(float(np.sqrt(np.mean((a_std_norm * DT) ** 2))), 1e-12)
            ),
        },
        "max_dv_any_frame": max_dv_where,
        "max_dv_window_ok": max_dv_ok_where,
        "expected_normalized_noise_var_if_matched": 1.0,
        "expected_normalized_noise_var_with_current_noise_stats": (
            float((noise["by_type"]["interior"]["rms"] / DT / float(np.mean(a_std_norm))) ** 2)
            if noise else None
        ),
    }
    out = Path(args.out) if Path(args.out).is_absolute() else _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
