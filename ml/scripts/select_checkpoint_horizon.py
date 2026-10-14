"""Long-horizon checkpoint selection on 32 stratified val runs.

Rolls each checkpoint open-loop (batched by n_nodes) until exit/take-off/t_end
(Algorithm 2: z<=0 fall, z>=h rise, else timeout). Reports spider-relative and
spider-absolute position MSE at 0.2 / 1 / 5 / 10 s.

Selection metric
----------------
Mean **capped** spider-relative MSE at 5 s over **all** runs (non-finite →
ERROR_CAP_REL_MSE). Physical fraction and TTF stats are reported separately and
do not gate the selection mean.

  PYTHONPATH=. python -m ml.scripts.select_checkpoint_horizon \\
    --config ml/configs/gns_local_gpu.yaml \\
    --checkpoints path/best.pt path/last.pt \\
    --device cuda --out Tema\\ 1/results/ml/logs/ckpt_select_local.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ml.data.dataset import BallooningDataset, BallooningRunStore
from ml.data.splits import load_splits
from ml.evaluate import full_rollout_batched
from ml.metrics.rollout_metrics import (
    ERROR_CAP_REL_MSE,
    PHYS_EDGE_TOL,
    aggregate_horizon_errors,
    aggregate_ttf,
    physical_fraction_at_times,
    segment_rel_deviations,
    time_to_failure,
)
from ml.models import build_model
from ml.models.baselines import ConstantVelocity, FollowAir
from ml.train import _device, _resolve, _set_seed
from ml.utils.config import load_yaml
from ml.validation import stratified_val_runs

HORIZONS_S = (0.2, 1.0, 5.0, 10.0)


def _load_model(cfg, store, ckpt_path, device, run_ids):
    dcfg = cfg["data"]
    edge_sets = list(dcfg.get("edge_sets", ["material"]))
    sample = BallooningDataset(
        store, run_ids[:1], C=dcfg.get("C", 5),
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets, radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True), noise_std=0.0,
    )[0]
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    if name in ("constant_velocity", "follow_air"):
        mcfg["a_mean"] = store.norm["a"]["mean"]
        mcfg["a_std"] = store.norm["a"]["std"]
        model = build_model(name, **mcfg)
    else:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        mcfg["node_dim"] = int(sample.x.shape[-1])
        mcfg.setdefault("edge_sets", edge_sets)
        edge_dims = {}
        for s in edge_sets:
            attr = getattr(sample, f"edge_attr_{s}", None)
            edge_dims[s] = int(attr.shape[-1]) if attr is not None and attr.ndim == 2 else 5
        mcfg["edge_dims"] = edge_dims
        model = build_model(name, **mcfg)
        model.load_state_dict(blob["model"], strict=False)
    return model.to(device).eval()


def _errors_at_horizons(pred, rec, horizons_s):
    """Spider-relative and spider-abs MSE at each horizon (or last frame if exited early)."""
    C = 5
    t0 = C
    x_rel_gt = rec["x_rel"].astype(np.float64)
    sp_gt = rec["spider_abs"].astype(np.float64)
    T_gt = x_rel_gt.shape[0]
    X = pred["x"]
    tt = pred["t"]
    out = {}
    for hs in horizons_s:
        t_target = float(rec["t"][t0]) + hs
        k = int(np.argmin(np.abs(tt - t_target)))
        t_gt = rec["t"].astype(np.float64)
        kg = int(np.argmin(np.abs(t_gt - t_target)))
        kg = min(max(kg, 0), T_gt - 1)
        sp_p = X[k, 0]
        x_rel_p = X[k] - sp_p
        rel = float(np.mean((x_rel_p - x_rel_gt[kg]) ** 2))
        abs_e = float(np.mean((sp_p - sp_gt[kg]) ** 2))
        out[f"{hs:g}s"] = {
            "rel_mse": rel,
            "spider_abs_mse": abs_e,
            "pred_t": float(tt[k]),
            "gt_t": float(t_gt[kg]),
            "exited_before": bool(tt[k] + 1e-9 < t_target),
        }
    return out


def _ttf_from_trajectory(pred, rec, C: int = 5) -> float:
    """Time-to-failure from release using max edge relative deviation / finite mask."""
    X = pred["x"]
    tt = np.asarray(pred["t"], dtype=np.float64)
    t_release = float(rec["t"][C])
    t_rel = tt - t_release
    edges = np.asarray(rec["edges"], dtype=np.int64)
    L_seg = float(rec["globals"]["L"]) / max(int(rec["globals"]["N_t"]), 1)
    max_edge_rel = np.empty(len(X), dtype=np.float64)
    finite_mask = np.empty(len(X), dtype=bool)
    for k in range(len(X)):
        finite = bool(np.isfinite(X[k]).all())
        finite_mask[k] = finite
        if not finite:
            max_edge_rel[k] = float("inf")
            continue
        devs = segment_rel_deviations(X[k], edges, L_seg)
        max_edge_rel[k] = float(np.max(devs)) if devs.size else 0.0
    return time_to_failure(t_rel, max_edge_rel, finite_mask)


def eval_checkpoint(model, store, run_ids, cfg, device, *, t_end: float):
    dcfg = cfg["data"]
    C = int(dcfg.get("C", 5))
    t0 = time.perf_counter()
    preds = full_rollout_batched(
        model, store, list(run_ids), device,
        C=C, t_end=t_end,
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=dcfg.get("edge_sets"),
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
    )
    per_run: list[dict] = []
    for rid, pred in preds.items():
        rec = store.get(rid)
        errs = _errors_at_horizons(pred, rec, HORIZONS_S)
        ttf = _ttf_from_trajectory(pred, rec, C=C)
        per_run.append({
            "run_id": rid,
            "horizon_errors": errs,
            "ttf_s": ttf,
        })
    wall = time.perf_counter() - t0
    horizons = aggregate_horizon_errors(per_run, HORIZONS_S)
    ttf_stats = aggregate_ttf(per_run)
    phys_at = physical_fraction_at_times(per_run, times=(1.0, 5.0, 10.0))
    # Full-rollout physical fraction: never failed (ttf = +inf).
    n_never = int(ttf_stats.get("n_never_fail", 0))
    n_tot = len(per_run)
    summary = {
        "n_runs": n_tot,
        "physical_fraction": n_never / max(n_tot, 1),
        "physical_fraction_at_times": phys_at,
        "ttf": ttf_stats,
        "wall_s": wall,
        "horizons": horizons,
        "error_cap_rel_mse": ERROR_CAP_REL_MSE,
        "physical_edge_tol": PHYS_EDGE_TOL,
        # Selection: mean capped 5s rel MSE over ALL runs (not physical-only).
        "select_5s_rel_mse": horizons.get("5s", {}).get("rel_mse_mean", float("nan")),
        "selection_metric": "mean_capped_5s_rel_mse_all_runs",
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true", help="also eval constant_velocity + follow_air")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--n-runs", type=int, default=32)
    ap.add_argument("--t-end", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_yaml(_resolve(args.config) if not args.config.is_absolute() else args.config)
    device = _device(args.device)
    _set_seed(args.seed)
    dcfg = cfg["data"]
    data_dir = Path(dcfg["data_dir"]) if Path(dcfg["data_dir"]).is_absolute() else _resolve(dcfg["data_dir"])
    splits = load_splits(
        Path(dcfg["splits"]) if Path(dcfg["splits"]).is_absolute() else _resolve(dcfg["splits"])
    )
    store = BallooningRunStore(data_dir, cache_shards=2, mmap=True)
    ids = stratified_val_runs(store, splits["val"], n_runs=args.n_runs, seed=args.seed)
    print(f"select on {len(ids)} val runs: {ids}", flush=True)

    rows = {}
    for ck in args.checkpoints:
        p = Path(ck) if Path(ck).is_absolute() else _resolve(ck)
        label = p.parent.name + "/" + p.name
        print(f"=== {label} ===", flush=True)
        model = _load_model(cfg, store, p, device, ids)
        rows[label] = eval_checkpoint(model, store, ids, cfg, device, t_end=args.t_end)
        rows[label]["path"] = str(p)
        print(json.dumps(rows[label], indent=2), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.baselines:
        for name, cls in (("constant_velocity", ConstantVelocity), ("follow_air", FollowAir)):
            print(f"=== {name} ===", flush=True)
            m = cls(a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"]).to(device).eval()
            cfg_b = json.loads(json.dumps(cfg))
            rows[name] = eval_checkpoint(m, store, ids, cfg_b, device, t_end=args.t_end)
            print(json.dumps(rows[name], indent=2), flush=True)

    cand = {k: v for k, v in rows.items() if k not in ("constant_velocity", "follow_air")}
    best_label = None
    best_score = float("inf")
    for k, v in cand.items():
        s = v.get("select_5s_rel_mse", float("nan"))
        if np.isfinite(s) and s < best_score:
            best_score = s
            best_label = k
    report = {
        "run_ids": ids,
        "horizons_s": list(HORIZONS_S),
        "physical_edge_tol": PHYS_EDGE_TOL,
        "error_cap_rel_mse": ERROR_CAP_REL_MSE,
        "selection_metric": "mean_capped_5s_rel_mse_all_runs",
        "selected": best_label,
        "selected_5s_rel_mse": best_score,
        "checkpoints": rows,
        "created_unix": time.time(),
    }
    out = Path(args.out) if Path(args.out).is_absolute() else _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"SELECTED {best_label} 5s_rel={best_score:.6e}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
