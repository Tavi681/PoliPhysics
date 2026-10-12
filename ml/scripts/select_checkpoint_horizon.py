"""Long-horizon checkpoint selection on 32 stratified val runs.

Rolls each checkpoint open-loop (batched by n_nodes) until exit/take-off/t_end
(Algorithm 2: z<=0 fall, z>=h rise, else timeout). Reports spider-relative and
spider-absolute position MSE at 0.2 / 1 / 5 / 10 s, plus the fraction of
rollouts that stay finite and physical (no NaN; max |Δℓ|/ℓ0 < 10%).

Selection metric: mean spider-relative MSE at 5 s among physical rollouts.

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
from ml.data.frames import DT
from ml.data.splits import load_splits
from ml.evaluate import full_rollout_surrogate
from ml.models import build_model
from ml.models.baselines import ConstantVelocity, FollowAir
from ml.train import _device, _resolve, _set_seed
from ml.utils.config import load_yaml
from ml.validation import stratified_val_runs

HORIZONS_S = (0.2, 1.0, 5.0, 10.0)
PHYS_EDGE_TOL = 0.10  # max |Δℓ|/ℓ0


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
    # pred["x"] is absolute positions (T_pred, n, 3); spider is node 0
    X = pred["x"]
    tt = pred["t"]
    out = {}
    for hs in horizons_s:
        # index in predicted trajectory nearest to t0_time + hs
        t_target = float(rec["t"][t0]) + hs
        k = int(np.argmin(np.abs(tt - t_target)))
        # corresponding GT frame
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


def _is_physical(pred, rec) -> bool:
    X = pred["x"]
    if not np.isfinite(X).all():
        return False
    if not np.isfinite(pred["v"]).all():
        return False
    # max relative segment-length deviation over trajectory
    edges = np.asarray(rec["edges"], dtype=np.int64)
    L_seg = float(rec["globals"]["L"]) / max(int(rec["globals"]["N_t"]), 1)
    i, j = edges[:, 0], edges[:, 1]
    lengths = np.linalg.norm(X[:, j] - X[:, i], axis=-1)  # (T, E)
    rel = np.abs(lengths - L_seg) / max(L_seg, 1e-12)
    return bool(np.max(rel) < PHYS_EDGE_TOL)


def eval_checkpoint(model, store, run_ids, cfg, device, *, t_end: float):
    dcfg = cfg["data"]
    C = int(dcfg.get("C", 5))
    per_h = {f"{hs:g}s": {"rel": [], "abs": []} for hs in HORIZONS_S}
    n_phys = 0
    n_tot = 0
    t0 = time.perf_counter()
    for rid in run_ids:
        rec = store.get(rid)
        T = int(rec["v"].shape[0])
        if C + int(HORIZONS_S[-1] / DT) >= T and T < C + 2:
            continue
        pred = full_rollout_surrogate(
            model, rec, store.norm, device,
            C=C, t_end=t_end,
            node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=dcfg.get("edge_sets"),
            radius=dcfg.get("radius", 0.1),
            normalize_features=dcfg.get("normalize_features", True),
        )
        n_tot += 1
        phys = _is_physical(pred, rec)
        if phys:
            n_phys += 1
        errs = _errors_at_horizons(pred, rec, HORIZONS_S)
        for hs in HORIZONS_S:
            key = f"{hs:g}s"
            if phys:
                per_h[key]["rel"].append(errs[key]["rel_mse"])
                per_h[key]["abs"].append(errs[key]["spider_abs_mse"])
    wall = time.perf_counter() - t0
    summary = {
        "n_runs": n_tot,
        "physical_fraction": n_phys / max(n_tot, 1),
        "wall_s": wall,
        "horizons": {},
    }
    for hs in HORIZONS_S:
        key = f"{hs:g}s"
        rel = per_h[key]["rel"]
        ab = per_h[key]["abs"]
        summary["horizons"][key] = {
            "rel_mse_mean": float(np.mean(rel)) if rel else float("nan"),
            "spider_abs_mse_mean": float(np.mean(ab)) if ab else float("nan"),
            "n_physical": len(rel),
        }
    summary["select_5s_rel_mse"] = summary["horizons"]["5s"]["rel_mse_mean"]
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
            # Temporarily patch cfg model name for feature expectations via evaluate path:
            cfg_b = json.loads(json.dumps(cfg))
            # use same node features as the GNS config being compared
            rows[name] = eval_checkpoint(m, store, ids, cfg_b, device, t_end=args.t_end)
            print(json.dumps(rows[name], indent=2), flush=True)

    # pick best among model checkpoints (exclude baselines) by 5s rel mse
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
        "selection_metric": "5s_rel_mse",
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
