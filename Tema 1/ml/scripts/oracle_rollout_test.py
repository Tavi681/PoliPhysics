"""Oracle GT-acceleration rollouts through the production evaluate path.

  PYTHONPATH=. python -m ml.scripts.oracle_rollout_test \\
    --data-dir "results/ml/processed/5c60fa669a719404" \\
    --run-ids-json "results/ml/paper_report_pull/diagnostics/gns_baseline_epoch06_val20/summary.json"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ml.data.dataset import BallooningRunStore
from ml.data.frames import DT
from ml.evaluate import full_rollout_surrogate
from ml.metrics.rollout_metrics import PHYS_EDGE_TOL, segment_rel_deviations, time_to_failure
from ml.models.oracle import OracleGroundTruthAccel
from ml.train import _resolve
from ml.utils.git_info import repo_root


def _roles(rec):
    edges = np.asarray(rec["edges"], dtype=np.int64)
    nt = rec["node_type"].astype(np.int64)
    adj = set()
    for i, j in edges:
        if i == 0:
            adj.add(int(j))
        if j == 0:
            adj.add(int(i))
    roles = []
    for n in range(len(nt)):
        if nt[n] == 0:
            roles.append("spider")
        elif n in adj:
            roles.append("spider_adj")
        elif nt[n] == 2:
            roles.append("tip")
        else:
            roles.append("interior")
    return np.array(roles, dtype=object)


def _max_abs_err_by_role(X_pred, X_gt, roles):
    d = np.abs(X_pred - X_gt)
    out = {}
    for name in ("spider", "spider_adj", "interior", "tip"):
        m = roles == name
        out[name] = float(np.max(d[:, m])) if m.any() else float("nan")
    out["all"] = float(np.max(d))
    return out


def run_one(store, rid, *, C, t_end, noise_std_by_type, noise_std_spider_adj, device):
    rec = store.get(rid)
    model = OracleGroundTruthAccel(
        rec, C=C, a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"], dt=DT,
    )
    model.noise_std_by_type = noise_std_by_type
    model.noise_std_spider_adj = noise_std_spider_adj
    model.reset()
    pred = full_rollout_surrogate(
        model, rec, store.norm, device,
        C=C, t_end=t_end, return_trajectory=True,
        node_features=["v_hist", "u", "u_next", "charge"],
        globals_fields=["qE", "m", "N"],
        edge_sets=["material", "interaction"],
        radius=0.05,
        normalize_features=True,
    )
    # Align GT absolute trajectory from release frame C
    t0 = C
    T_pred = pred["x"].shape[0]
    x_gt = rec["spider_abs"][t0 : t0 + T_pred, None, :] + rec["x_rel"][t0 : t0 + T_pred].astype(np.float64)
    T = min(T_pred, x_gt.shape[0])
    X = pred["x"][:T]
    G = x_gt[:T]
    roles = _roles(rec)
    max_err = _max_abs_err_by_role(X, G, roles)

    edges = np.asarray(rec["edges"], dtype=np.int64)
    L_seg = float(rec["globals"]["L"]) / max(int(rec["globals"]["N_t"]), 1)
    max_edge = []
    finite = []
    t_rel = pred["t"][:T] - float(rec["t"][t0])
    for k in range(T):
        if not np.isfinite(X[k]).all():
            max_edge.append(float("inf"))
            finite.append(False)
            continue
        finite.append(True)
        max_edge.append(float(np.max(segment_rel_deviations(X[k], edges, L_seg))))
    ttf = time_to_failure(t_rel, max_edge, finite)
    n_fail = sum(1 for m in max_edge if m >= PHYS_EDGE_TOL)
    gt_out = rec["globals"].get("outcome", "timeout")
    return {
        "run_id": rid,
        "status_surr": pred["status"],
        "status_sim": gt_out,
        "outcome_match": pred["status"] == gt_out,
        "n_edge_tol_failures": n_fail,
        "ttf_s": float(ttf) if np.isfinite(ttf) else float("inf"),
        "max_abs_err_m": max_err,
        "max_edge_rel_first5": max_edge[1:6],
        "T": T,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--run-ids-json", type=Path, required=True)
    ap.add_argument("--C", type=int, default=5)
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = args.data_dir if args.data_dir.is_absolute() else _resolve(args.data_dir)
    summary = json.loads(Path(args.run_ids_json if args.run_ids_json.is_absolute() else _resolve(args.run_ids_json)).read_text())
    run_ids = summary["run_ids"]
    store = BallooningRunStore(data_dir, cache_shards=4, mmap=True)
    device = torch.device(args.device)

    t0 = time.perf_counter()
    clean = [
        run_one(
            store, rid, C=args.C, t_end=args.t_end,
            noise_std_by_type=None, noise_std_spider_adj=None, device=device,
        )
        for rid in run_ids
    ]
    # noise std = sqrt(one-step MSE) in normalized units; node_type 0/1/2 + spider_adj override
    noise = np.array([np.sqrt(0.055), np.sqrt(0.013), np.sqrt(0.035)], dtype=np.float32)
    noise_adj = float(np.sqrt(0.017))
    noisy = [
        run_one(
            store, rid, C=args.C, t_end=args.t_end,
            noise_std_by_type=noise, noise_std_spider_adj=noise_adj, device=device,
        )
        for rid in run_ids
    ]

    report = {
        "data_dir": str(data_dir),
        "t_end": args.t_end,
        "C": args.C,
        "n_runs": len(run_ids),
        "wall_s": time.perf_counter() - t0,
        "oracle_clean": {
            "n_outcome_match": sum(1 for r in clean if r["outcome_match"]),
            "n_zero_edge_failures": sum(1 for r in clean if r["n_edge_tol_failures"] == 0),
            "max_abs_err_m_worst": max(r["max_abs_err_m"]["all"] for r in clean),
            "per_role_worst_m": {
                k: max(r["max_abs_err_m"][k] for r in clean)
                for k in ("spider", "spider_adj", "interior", "tip")
            },
            "runs": clean,
        },
        "oracle_noisy": {
            "noise_std_by_type_norm": noise.tolist(),
            "noise_std_spider_adj_norm": noise_adj,
            "ttf_median": float(np.median([r["ttf_s"] for r in noisy])),
            "ttf_min": float(np.min([r["ttf_s"] for r in noisy])),
            "ttf_max": float(np.max([r["ttf_s"] if np.isfinite(r["ttf_s"]) else 1e9 for r in noisy])),
            "runs": [
                {
                    "run_id": r["run_id"],
                    "ttf_s": r["ttf_s"],
                    "max_edge_rel_first5": r["max_edge_rel_first5"],
                    "n_edge_tol_failures": r["n_edge_tol_failures"],
                    "status_surr": r["status_surr"],
                }
                for r in noisy
            ],
        },
        "git": None,
    }
    try:
        from ml.utils.git_info import git_provenance
        report["git"] = git_provenance(repo_root())
    except Exception:
        pass

    out = args.out if args.out.is_absolute() else _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    summary = {
        "wall_s": report["wall_s"],
        "oracle_clean": {k: v for k, v in report["oracle_clean"].items() if k != "runs"},
        "oracle_noisy": {k: v for k, v in report["oracle_noisy"].items() if k != "runs"},
        "clean_sample": clean[0],
        "noisy_sample": report["oracle_noisy"]["runs"][0],
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
