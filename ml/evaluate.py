"""Evaluate a checkpoint on test splits."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.data.dataset import BallooningDataset, BallooningRunStore, make_loader
from ml.data.integrator import step_np, DT
from ml.data.normalize import denormalize
from ml.data.splits import load_splits
from ml.metrics.physics import (
    edge_length_drift,
    mean_vertical_spider_velocity,
    min_interthread_distance,
    opening_R_over_L,
    wilson_interval,
)
from ml.models import build_model
from ml.train import _device, _resolve, one_step_mse, rollout_mse
from ml.utils.config import load_yaml
from ml.utils.git_info import git_provenance, repo_root


def _settling_speed(N: int, m: float = 1e-6, L: float = 0.5, N_t: int = 50) -> float:
    """Import simulator settling_speed when possible."""
    try:
        import sys
        tema1 = str(repo_root() / "Tema 1")
        if tema1 not in sys.path:
            sys.path.insert(0, tema1)
        from ballooning.params import Params
        from ballooning.sweep import settling_speed
        return float(settling_speed(Params(N=N, N_t=N_t, L=L, m=m)))
    except Exception:
        g = 9.81
        # rough fallback
        return m * g / max(N, 1)


def _make_flow(seed: int, sigma_w: float, ell: float, U_h: float):
    import sys
    tema1 = str(repo_root() / "Tema 1")
    if tema1 not in sys.path:
        sys.path.insert(0, tema1)
    from ballooning.fields import KinematicSimulation
    return KinematicSimulation(
        sigma=sigma_w, ell=ell, U_h=U_h, N_k=200, seed=int(seed),
        lambda_=0.5, renormalize=True,
    )


def _interp_gt(t_src: np.ndarray, y_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear interpolate y_src[T,...] onto t_query (float64)."""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_query = np.asarray(t_query, dtype=np.float64)
    out = np.empty((len(t_query),) + y_src.shape[1:], dtype=np.float64)
    for i, tq in enumerate(t_query):
        if tq <= t_src[0]:
            out[i] = y_src[0]
        elif tq >= t_src[-1]:
            out[i] = y_src[-1]
        else:
            j = int(np.searchsorted(t_src, tq) - 1)
            j = max(0, min(j, len(t_src) - 2))
            a = (tq - t_src[j]) / (t_src[j + 1] - t_src[j])
            out[i] = (1 - a) * y_src[j] + a * y_src[j + 1]
    return out


@torch.no_grad()
def full_rollout_surrogate(
    model,
    rec: dict,
    norm: dict,
    device: torch.device,
    *,
    C: int = 5,
    t_end: float = 60.0,
    include_air_time: bool = True,
) -> dict:
    """Roll from release until z=0, z=h, or t_end. Air from KinematicSimulation."""
    g = rec["globals"]
    flow = _make_flow(g["seed"], g["sigma_w"], g["ell"], g["U_h"])
    h, z0 = float(g["h"]), float(g["z0"])
    a_mean = np.asarray(norm["a"]["mean"], dtype=np.float32)
    a_std = np.asarray(norm["a"]["std"], dtype=np.float32)

    # start after history warmup at frame C
    t0 = C
    x_rel = rec["x_rel"][t0].astype(np.float32).copy()
    spider = rec["spider_abs"][t0].astype(np.float64).copy()
    v = rec["v"][t0].astype(np.float32).copy()
    # velocity history buffer
    v_hist = rec["v"][t0 - C + 1 : t0 + 1].astype(np.float32).copy()

    from ml.data.dataset import BallooningDataset
    # dummy dataset for feature plumbing — we'll override fields
    store_stub = None

    xs_abs = [spider + x_rel.astype(np.float64)]
    vs = [v.copy()]
    ts = [float(rec["t"][t0])]
    air_time = 0.0
    wall0 = time.perf_counter()

    model.eval()
    tip_ids = np.where(rec["node_type"] == 2)[0]
    status = "timeout"
    exit_time = float("nan")
    t = float(rec["t"][t0])
    step = 0
    max_steps = int(t_end / DT) + 5

    while t < t_end and step < max_steps:
        x_abs = spider + x_rel.astype(np.float64)
        t_air0 = time.perf_counter()
        u = flow.u(x_abs, t).astype(np.float32)
        air_time += time.perf_counter() - t_air0

        # build features manually
        n = x_rel.shape[0]
        feat_parts = [
            v_hist.transpose(1, 0, 2).reshape(n, -1),
            u,
            rec["charge"][:, None].astype(np.float32),
            np.eye(3, dtype=np.float32)[rec["node_type"].astype(np.int64)],
        ]
        x_feat = torch.from_numpy(np.concatenate(feat_parts, axis=-1)).to(device)
        from ml.data.graph import material_edges
        e = material_edges(rec["edges"]).to(device)
        from torch_geometric.data import Data
        data = Data(
            x=x_feat,
            pos=torch.from_numpy(x_rel).to(device),
            edge_index=e,
            edge_index_material=e,
            node_type=torch.from_numpy(rec["node_type"].astype(np.int64)).to(device),
            thread_id=torch.from_numpy(rec["thread_id"].astype(np.int64)).to(device),
            v_t=torch.from_numpy(v).to(device),
            u_t=torch.from_numpy(u).to(device),
            num_nodes=n,
        )
        data.w_s = torch.tensor([_settling_speed(int(g["N"]), m=float(g["m"]), L=float(g["L"]), N_t=int(g["N_t"]))], device=device)

        pred_n = model(data)
        a = pred_n.detach().cpu().numpy().astype(np.float32) * a_std + a_mean
        x_rel, v = step_np(x_rel, v, a, dt=DT)
        # update spider abs: x_rel spider row is 0 by construction; track abs via integration
        spider = spider + DT * v[0].astype(np.float64)
        # re-zero relative to spider
        x_abs = np.vstack([spider[None, :], (spider + x_rel[1:].astype(np.float64))])
        # actually keep x_rel with spider at 0
        x_rel = (x_abs - spider[None, :]).astype(np.float32)
        v_hist = np.concatenate([v_hist[1:], v[None]], axis=0)
        t = t + DT
        step += 1
        xs_abs.append(x_abs.copy())
        vs.append(v.copy())
        ts.append(t)

        z = float(spider[2])
        if z >= h:
            status = "rise"
            exit_time = t
            break
        if z <= 0.0:
            status = "fall"
            exit_time = t
            break

    wall = time.perf_counter() - wall0
    X = np.stack(xs_abs)
    V = np.stack(vs)
    tt = np.asarray(ts)
    return {
        "status": status,
        "exit_time": exit_time,
        "t": tt,
        "x": X,
        "v": V,
        "wall_time_s": wall,
        "air_time_s": air_time,
        "V_mean": mean_vertical_spider_velocity(X, tt),
        "R_over_L": opening_R_over_L(X, tip_ids, L=float(g["L"])),
        "edge_drift": edge_length_drift(X, rec["edges"], L_seg=float(g["L"]) / int(g["N_t"])),
        "dmin": min_interthread_distance(X[-1], rec["thread_id"]),
    }


def evaluate_checkpoint(
    ckpt_path: Path,
    cfg: dict,
    device: torch.device,
    splits_to_eval: list[str] | None = None,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    data_dir = _resolve(cfg["data"]["data_dir"])
    store = BallooningRunStore(data_dir)
    splits = load_splits(_resolve(cfg["data"]["splits"]))
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    sample = BallooningDataset(store, splits.get("train") or next(iter(splits.values())), C=cfg["data"].get("C", 5))[0]
    if name in ("constant_velocity", "follow_air"):
        mcfg["a_mean"] = store.norm["a"]["mean"]
        mcfg["a_std"] = store.norm["a"]["std"]
    else:
        mcfg["node_dim"] = int(sample.x.shape[-1])
    model = build_model(name, **mcfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    prov = git_provenance()
    run_id = ckpt.get("meta", {}).get("run_id") or ckpt_path.parent.name
    out_dir = repo_root() / "Tema 1" / "results" / "ml" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = splits_to_eval or ["test_interp", "test_unseenN", "test_unseenSigma"]
    rows = []
    cost_rows = []
    metrics = {"git": prov, "run_id": run_id, "model": name, "splits": {}}

    for sp in split_names:
        ids = [r for r in splits.get(sp, []) if r in store.runs]
        if not ids:
            continue
        ds = BallooningDataset(
            store, ids, C=cfg["data"].get("C", 5),
            node_features=cfg["data"].get("node_features"),
            edge_sets=cfg["data"].get("edge_sets", ["material"]),
            noise_std=0.0, sample_weight="per_frame",
        )
        loader = make_loader(ds, batch_size=4, shuffle=False, device=str(device))
        osm = one_step_mse(model, loader, device, store.norm) if len(ds) else float("nan")
        rom = rollout_mse(
            model, store, ids, device, store.norm,
            K=int(cfg.get("eval", {}).get("rollout_K", 100)),
            C=cfg["data"].get("C", 5), max_runs=min(8, len(ids)),
        )

        # full rollouts on available runs in local subset
        outcomes_ok = 0
        exit_errs = []
        dV = []
        dR = []
        takeoff_s = []
        takeoff_m = []
        walls = []
        airs = []
        max_full = cfg.get("eval", {}).get("max_full_runs")
        ids_full = ids if not max_full else ids[: int(max_full)]
        for rid in ids_full:
            rec = store.get(rid)
            gt_out = rec["globals"]["outcome"]
            pred = full_rollout_surrogate(
                model, rec, store.norm, device,
                C=cfg["data"].get("C", 5),
                t_end=float(cfg.get("eval", {}).get("t_end", 60.0)),
            )
            walls.append(pred["wall_time_s"])
            airs.append(pred["air_time_s"])
            if pred["status"] == gt_out:
                outcomes_ok += 1
            if np.isfinite(pred["exit_time"]) and np.isfinite(rec["globals"]["exit_time"]):
                exit_errs.append(abs(pred["exit_time"] - rec["globals"]["exit_time"]))
            # GT metrics from stored trajectory
            x_abs = rec["spider_abs"][:, None, :] + rec["x_rel"].astype(np.float64)
            V_gt = mean_vertical_spider_velocity(x_abs, rec["t"].astype(np.float64))
            tips = np.where(rec["node_type"] == 2)[0]
            R_gt = opening_R_over_L(x_abs, tips, L=float(rec["globals"]["L"]))
            if abs(V_gt) > 1e-12:
                dV.append(abs(pred["V_mean"] - V_gt) / abs(V_gt))
            if abs(R_gt) > 1e-12:
                dR.append(abs(pred["R_over_L"] - R_gt) / abs(R_gt))
            takeoff_s.append(1 if gt_out == "rise" else 0)
            takeoff_m.append(1 if pred["status"] == "rise" else 0)

        n = len(ids_full)
        k_s, k_m = sum(takeoff_s), sum(takeoff_m)
        P_s, P_m = k_s / max(n, 1), k_m / max(n, 1)
        lo_s, hi_s = wilson_interval(k_s, n)
        lo_m, hi_m = wilson_interval(k_m, n)

        # speed-up vs simulator wall_time
        sim_times = [
            store.get(rid)["globals"]["wall_time_s"]
            for rid in ids_full
            if np.isfinite(store.get(rid)["globals"]["wall_time_s"])
        ]
        t_sim = float(np.nanmean(sim_times)) if sim_times else float("nan")
        t_surr = float(np.mean(walls)) if walls else float("nan")
        speedup = t_sim / t_surr if (np.isfinite(t_sim) and t_surr > 0) else float("nan")

        split_metrics = {
            "n_runs": n,
            "one_step_mse": osm,
            "rollout_mse": rom,
            "outcome_accuracy": outcomes_ok / max(n, 1),
            "exit_time_mae": float(np.mean(exit_errs)) if exit_errs else float("nan"),
            "delta_V_over_V": float(np.mean(dV)) if dV else float("nan"),
            "delta_R_over_L": float(np.mean(dR)) if dR else float("nan"),
            "P_sim": P_s,
            "P_surr": P_m,
            "P_sim_wilson": [lo_s, hi_s],
            "P_surr_wilson": [lo_m, hi_m],
            "abs_delta_P": abs(P_m - P_s),
            "time_per_run_s": t_surr,
            "air_time_per_run_s": float(np.mean(airs)) if airs else float("nan"),
            "time_sim_mean_s": t_sim,
            "speedup": speedup,
        }
        metrics["splits"][sp] = split_metrics
        rows.append({
            "Model": name,
            "Split": sp,
            "Rollout MSE": rom,
            "Delta V/V (%)": 100 * split_metrics["delta_V_over_V"] if np.isfinite(split_metrics["delta_V_over_V"]) else float("nan"),
            "Delta(R/L) (%)": 100 * split_metrics["delta_R_over_L"] if np.isfinite(split_metrics["delta_R_over_L"]) else float("nan"),
            "|Delta P|": split_metrics["abs_delta_P"],
            "Speed-up": speedup,
        })

    # cost table stub (C_data / C_train filled by make_tables when aggregating)
    cost_rows.append({
        "Method": name,
        "Hardware": str(device),
        "Time per run": metrics["splits"][split_names[0]]["time_per_run_s"] if metrics["splits"] else float("nan"),
        "Time per P (M)": float("nan"),
        "C_data": float("nan"),
        "C_train": ckpt.get("meta", {}).get("train_time_s", float("nan")),
        "Q*": float("nan"),
        "|Delta P|": float(np.mean([m["abs_delta_P"] for m in metrics["splits"].values()])) if metrics["splits"] else float("nan"),
        "air_time_per_run": float(np.mean([m.get("air_time_per_run_s", float("nan")) for m in metrics["splits"].values()])),
    })

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(rows).to_csv(out_dir / "tab_accuracy.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(out_dir / "tab_cost.csv", index=False)
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    args = ap.parse_args()
    cfg_path = args.config if args.config.is_absolute() else _resolve(args.config)
    if not cfg_path.exists():
        cfg_path = repo_root() / "ml" / args.config
    cfg = load_yaml(cfg_path)
    evaluate_checkpoint(Path(args.checkpoint), cfg, _device(args.device), args.splits)


if __name__ == "__main__":
    main()
