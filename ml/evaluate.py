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
from ml.data.splits import load_splits, split_display_name
from ml.metrics.physics import (
    edge_length_drift,
    mean_vertical_spider_velocity,
    min_interthread_distance,
    opening_R_over_L,
    wilson_interval,
)
from ml.metrics.speedup import (
    mean_speedup,
    q_star,
    scale_surr_to_full_length,
    sim_seconds,
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


def _norm_blocks(arr_c_n_3: np.ndarray, stats: dict | None) -> np.ndarray:
    from ml.data.normalize import normalize
    if stats is None:
        return arr_c_n_3.reshape(arr_c_n_3.shape[1], -1).astype(np.float32)
    blocks = [normalize(arr_c_n_3[k], stats["mean"], stats["std"]) for k in range(arr_c_n_3.shape[0])]
    return np.concatenate(blocks, axis=-1)


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
    node_features: list[str] | None = None,
    globals_fields: list[str] | None = None,
    edge_sets: list[str] | None = None,
    radius: float = 0.1,
    normalize_features: bool = True,
) -> dict:
    """Roll from release until z=0, z=h, or t_end.

    Air: u_t and (if requested) u_next from KinematicSimulation at the *current*
    predicted positions — same definition as stored u_next (no /u_air leak).
    """
    from torch_geometric.data import Data
    from ml.data.edge_features import (
        build_edge_sets_gns,
        interaction_edge_attr,
        material_edge_attr,
    )
    from ml.data.graph import build_edge_sets, material_edges
    from ml.data.ks_torch import TurbModeCache, ks_u_torch
    from ml.data.normalize import normalize

    g = rec["globals"]
    modes = TurbModeCache().get(g)
    h = float(g["h"])
    a_mean = np.asarray(norm["a"]["mean"], dtype=np.float32)
    a_std = np.asarray(norm["a"]["std"], dtype=np.float32)
    node_features = list(node_features or ["v_hist", "u", "charge", "node_type"])
    globals_fields = list(globals_fields or [])
    edge_sets = list(edge_sets or model.required_edges() or ["material"])

    t0 = C
    x_rel = rec["x_rel"][t0].astype(np.float32).copy()
    spider = rec["spider_abs"][t0].astype(np.float64).copy()
    v = rec["v"][t0].astype(np.float32).copy()
    v_hist = rec["v"][t0 - C + 1 : t0 + 1].astype(np.float32).copy()
    u_hist = rec["u"][t0 - C + 1 : t0 + 1].astype(np.float32).copy()

    xs_abs = [spider + x_rel.astype(np.float64)]
    vs = [v.copy()]
    ts = [float(rec["t"][t0])]
    air_time = 0.0
    model_time = 0.0
    integ_time = 0.0
    wall0 = time.perf_counter()

    model.eval()
    tip_ids = np.where(rec["node_type"] == 2)[0]
    status = "timeout"
    exit_time = float("nan")
    t = float(rec["t"][t0])
    step = 0
    max_steps = int(t_end / DT) + 5
    l0 = float(g["L"]) / max(int(g["N_t"]), 1)
    nt_np = rec["node_type"].astype(np.int64)
    tid_np = rec["thread_id"].astype(np.int64)
    charge_np = rec["charge"].astype(np.float32)

    while t < t_end and step < max_steps:
        x_abs = spider + x_rel.astype(np.float64)
        n = x_rel.shape[0]
        t_air0 = time.perf_counter()
        x_dev = torch.from_numpy(x_abs).to(device)
        u = ks_u_torch(x_dev, t, modes).detach().cpu().numpy().astype(np.float32)
        u_next = None
        if "u_next" in node_features:
            u_next = (
                ks_u_torch(x_dev, t + DT, modes).detach().cpu().numpy().astype(np.float32)
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        air_time += time.perf_counter() - t_air0

        feats = []
        for name in node_features:
            if name == "v_hist":
                feats.append(_norm_blocks(v_hist, norm.get("v") if normalize_features else None))
            elif name == "u":
                feats.append(
                    normalize(u, norm["u"]["mean"], norm["u"]["std"]) if normalize_features else u
                )
            elif name == "u_next":
                feats.append(
                    normalize(u_next, norm["u"]["mean"], norm["u"]["std"])
                    if normalize_features else u_next
                )
            elif name == "u_hist":
                uh = u_hist[:-1] if u_hist.shape[0] == C else u_hist
                feats.append(_norm_blocks(uh, norm.get("u") if normalize_features else None))
            elif name == "charge":
                feats.append(charge_np[:, None])
            elif name == "node_type":
                feats.append(np.eye(3, dtype=np.float32)[nt_np])
            elif name == "spider_z":
                feats.append(np.full((n, 1), float(spider[2]), dtype=np.float32))
            else:
                raise KeyError(name)
        for gf in globals_fields:
            if gf == "qE":
                val = float(g["q"]) * float(g["E"])
            elif gf == "m":
                val = float(g["m"])
            elif gf == "N":
                val = float(g["N"])
            elif gf == "sigma_w":
                val = float(g["sigma_w"])
            elif gf == "ell":
                val = float(g["ell"])
            elif gf == "spider_z":
                val = float(spider[2])
            else:
                raise KeyError(gf)
            feats.append(np.full((n, 1), val, dtype=np.float32))
        x_feat = torch.from_numpy(np.concatenate(feats, axis=-1).astype(np.float32)).to(device)

        pos = torch.from_numpy(x_rel).to(device)
        pos_abs = pos + torch.from_numpy(spider.astype(np.float32)).to(device)
        node_type = torch.from_numpy(nt_np).to(device)
        thread_id = torch.from_numpy(tid_np).to(device)
        charge_t = torch.from_numpy(charge_np).to(device)

        if "interaction" in edge_sets:
            edge_dict = build_edge_sets_gns(
                edge_sets, edges_material=rec["edges"], pos=pos_abs.cpu(),
                thread_id=thread_id.cpu(), node_type=node_type.cpu(), radius=radius,
            )
            inter_flags = edge_dict.pop("_interaction_flags")
        else:
            edge_dict = build_edge_sets(
                edge_sets, edges_material=rec["edges"], pos=pos_abs.cpu(),
                thread_id=thread_id.cpu(), node_type=node_type.cpu(), radius=radius,
            )
            inter_flags = torch.zeros(0)

        e_mat = edge_dict.get("material", material_edges(rec["edges"])).to(device)
        data = Data(
            x=x_feat, pos=pos, edge_index=e_mat, edge_index_material=e_mat,
            node_type=node_type, thread_id=thread_id, charge=charge_t,
            v_t=torch.from_numpy(v).to(device), u_t=torch.from_numpy(u).to(device),
            num_nodes=n,
        )
        for name, eidx in edge_dict.items():
            eidx_d = eidx.to(device)
            setattr(data, f"edge_index_{name}", eidx_d)
            if name == "material":
                data.edge_attr_material = material_edge_attr(pos, eidx_d, l0=l0)
            elif name == "interaction":
                data.edge_attr_interaction = interaction_edge_attr(
                    pos, eidx_d, charge_t, inter_flags.to(device)
                )

        t_m0 = time.perf_counter()
        pred_n = model(data)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        model_time += time.perf_counter() - t_m0
        a = pred_n.detach().cpu().numpy().astype(np.float32) * a_std + a_mean

        t_i0 = time.perf_counter()
        x_rel, v = step_np(x_rel, v, a, dt=DT)
        spider = spider + DT * v[0].astype(np.float64)
        x_abs = np.vstack([spider[None, :], (spider + x_rel[1:].astype(np.float64))])
        x_rel = (x_abs - spider[None, :]).astype(np.float32)
        v_hist = np.concatenate([v_hist[1:], v[None]], axis=0)
        u_hist = np.concatenate([u_hist[1:], u[None]], axis=0)
        integ_time += time.perf_counter() - t_i0

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
    sim_dur = float(tt[-1] - tt[0]) if len(tt) > 1 else 0.0
    return {
        "status": status,
        "exit_time": exit_time,
        "t": tt,
        "x": X,
        "v": V,
        "wall_time_s": wall,
        "air_time_s": air_time,
        "model_time_s": model_time,
        "integ_time_s": integ_time,
        "sim_duration_s": sim_dur,
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
    dcfg = cfg["data"]
    edge_sets = list(dcfg.get("edge_sets", ["material"]))
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    sample = BallooningDataset(
        store, splits.get("train") or next(iter(splits.values())),
        C=dcfg.get("C", 5),
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets,
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
    )[0]
    if name in ("constant_velocity", "follow_air"):
        mcfg["a_mean"] = store.norm["a"]["mean"]
        mcfg["a_std"] = store.norm["a"]["std"]
    else:
        mcfg["node_dim"] = int(sample.x.shape[-1])
        mcfg.setdefault("edge_sets", edge_sets)
        edge_dims = dict(mcfg.get("edge_dims") or {})
        for s in edge_sets:
            attr = getattr(sample, f"edge_attr_{s}", None)
            if attr is not None and attr.ndim == 2 and attr.shape[-1] > 0:
                edge_dims[s] = int(attr.shape[-1])
            else:
                edge_dims.setdefault(s, 5 if s == "material" else 7)
        mcfg["edge_dims"] = edge_dims
    model = build_model(name, **mcfg)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    prov = git_provenance()
    run_id = ckpt.get("meta", {}).get("run_id") or ckpt_path.parent.name
    out_dir = repo_root() / "Tema 1" / "results" / "ml" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = splits_to_eval or [
        "test_interp", "test_unseenN_interp", "test_unseenN", "test_unseenSigma",
    ]
    rows = []
    cost_rows = []
    metrics = {"git": prov, "run_id": run_id, "model": name, "splits": {}}

    eval_seed = int(cfg.get("eval", {}).get("seed", 42))
    one_step_n_frames = int(cfg.get("eval", {}).get("one_step_n_frames", 100_000))
    horizon_errors_s = list(cfg.get("eval", {}).get("horizon_errors_s", [0.2, 1.0, 5.0, 10.0]))

    for sp in split_names:
        ids = [r for r in splits.get(sp, []) if r in store.runs]
        if not ids:
            continue
        # One-step: fixed random sample of frames across ALL runs in the split
        # (same indices for every method when seed matches).
        ds = BallooningDataset(
            store, ids, C=dcfg.get("C", 5),
            node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=edge_sets,
            radius=dcfg.get("radius", 0.1),
            normalize_features=dcfg.get("normalize_features", True),
            noise_std=0.0, sample_weight="per_frame",
            seed=eval_seed,
        )
        n_frames = len(ds)
        rng = np.random.default_rng(eval_seed + hash(sp) % 10_000)
        k = min(one_step_n_frames, n_frames) if one_step_n_frames > 0 else n_frames
        frame_idx = sorted(rng.choice(n_frames, size=k, replace=False).tolist()) if n_frames else []
        from ml.validation import one_step_mse_sample
        osm = (
            one_step_mse_sample(
                model, ds, frame_idx, device,
                batch_size=int(cfg.get("eval", {}).get("one_step_batch", 16)),
            )
            if frame_idx else float("nan")
        )
        # Short fixed-K rollout diagnostic (separate from full-length Alg.2 rollouts).
        rom_max = cfg.get("eval", {}).get("rollout_mse_max_runs")
        rom_ids = ids if not rom_max else ids[: int(rom_max)]
        rom = rollout_mse(
            model, store, rom_ids, device, store.norm,
            K=int(cfg.get("eval", {}).get("rollout_K", 100)),
            C=dcfg.get("C", 5), max_runs=len(rom_ids),
            node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=edge_sets,
            radius=dcfg.get("radius", 0.1),
            normalize_features=dcfg.get("normalize_features", True),
        )

        # Full-length Alg.2 rollouts on ALL runs in the split (z<=0 fall, z>=h rise).
        outcomes_ok = 0
        exit_errs = []
        dV = []
        dR = []
        takeoff_s = []
        takeoff_m = []
        walls = []
        airs = []
        models_t = []
        integs_t = []
        surr_durs = []
        sim_walls = []
        sim_durs = []
        delta_P_by_grid: dict[tuple, list] = {}
        horizon_rel = {f"{hs:g}s": [] for hs in horizon_errors_s}
        horizon_abs = {f"{hs:g}s": [] for hs in horizon_errors_s}
        t_end_cfg = float(cfg.get("eval", {}).get("t_end", 60.0))
        max_full = cfg.get("eval", {}).get("max_full_runs")
        ids_full = ids if not max_full else ids[: int(max_full)]
        C = int(dcfg.get("C", 5))
        for rid in ids_full:
            rec = store.get(rid)
            gt_out = rec["globals"]["outcome"]
            pred = full_rollout_surrogate(
                model, rec, store.norm, device,
                C=dcfg.get("C", 5),
                t_end=t_end_cfg,
                node_features=dcfg.get("node_features"),
                globals_fields=dcfg.get("globals_fields"),
                edge_sets=edge_sets,
                radius=dcfg.get("radius", 0.1),
                normalize_features=dcfg.get("normalize_features", True),
            )
            walls.append(pred["wall_time_s"])
            airs.append(pred["air_time_s"])
            models_t.append(pred["model_time_s"])
            integs_t.append(pred["integ_time_s"])
            surr_durs.append(pred["sim_duration_s"])
            sw = rec["globals"]["wall_time_s"]
            sd = sim_seconds(rec["globals"].get("exit_time"), t_end=60.0)
            sim_walls.append(float(sw) if np.isfinite(sw) else float("nan"))
            sim_durs.append(sd)
            if pred["status"] == gt_out:
                outcomes_ok += 1
            if np.isfinite(pred["exit_time"]) and np.isfinite(rec["globals"]["exit_time"]):
                exit_errs.append(abs(pred["exit_time"] - rec["globals"]["exit_time"]))
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
            # P grid: (N, q, sigma_w)
            gkey = (
                int(rec["globals"]["N"]),
                round(float(rec["globals"]["q"]), 6),
                round(float(rec["globals"]["sigma_w"]), 4),
            )
            delta_P_by_grid.setdefault(gkey, []).append(
                (1 if pred["status"] == "rise" else 0) - (1 if gt_out == "rise" else 0)
            )
            # Horizon position errors (spider-rel / spider-abs)
            t_gt = rec["t"].astype(np.float64)
            X = pred["x"]
            tt = pred["t"]
            t_release = float(rec["t"][C])
            for hs in horizon_errors_s:
                t_target = t_release + float(hs)
                k = int(np.argmin(np.abs(tt - t_target)))
                kg = int(np.argmin(np.abs(t_gt - t_target)))
                kg = min(max(kg, 0), x_abs.shape[0] - 1)
                sp_p = X[k, 0]
                x_rel_p = X[k] - sp_p
                key = f"{hs:g}s"
                horizon_rel[key].append(float(np.mean((x_rel_p - rec["x_rel"][kg]) ** 2)))
                horizon_abs[key].append(float(np.mean((sp_p - rec["spider_abs"][kg]) ** 2)))

        n = len(ids_full)
        k_s, k_m = sum(takeoff_s), sum(takeoff_m)
        P_s, P_m = k_s / max(n, 1), k_m / max(n, 1)
        lo_s, hi_s = wilson_interval(k_s, n)
        lo_m, hi_m = wilson_interval(k_m, n)
        grid_abs = [abs(float(np.mean(v))) for v in delta_P_by_grid.values() if v]
        abs_delta_P_mean = float(np.mean(grid_abs)) if grid_abs else float("nan")
        abs_delta_P_max = float(np.max(grid_abs)) if grid_abs else float("nan")
        horizon_report = {
            hs: {
                "rel_mse_mean": float(np.mean(horizon_rel[hs])) if horizon_rel[hs] else float("nan"),
                "spider_abs_mse_mean": float(np.mean(horizon_abs[hs])) if horizon_abs[hs] else float("nan"),
            }
            for hs in horizon_rel
        }

        # Duration-normalized speed-up; only runs with finite simulator wall_time_s
        speedup = mean_speedup(sim_walls, sim_durs, walls, surr_durs)
        # Full-length equivalent times for Q* (scale surrogate to simulator duration)
        full_sim_walls = []
        full_surr_walls = []
        for sw, sd, uw, ud in zip(sim_walls, sim_durs, walls, surr_durs):
            if not np.isfinite(sw):
                continue
            full_sim_walls.append(sw)
            full_surr_walls.append(scale_surr_to_full_length(uw, ud, sd))
        t_sim = float(np.mean(full_sim_walls)) if full_sim_walls else float("nan")
        t_surr_full = float(np.mean(full_surr_walls)) if full_surr_walls else float("nan")
        t_surr = float(np.mean(walls)) if walls else float("nan")

        split_metrics = {
            "n_runs": n,
            "one_step_mse": osm,
            "one_step_n_frames": len(frame_idx),
            "one_step_seed": eval_seed,
            # Short-K diagnostic rollout (not the paper full-length metric).
            "rollout_mse_shortK": rom,
            "rollout_mse_units": "m^2_rel_pos",
            "rollout_K": int(cfg.get("eval", {}).get("rollout_K", 100)),
            "horizon_errors": horizon_report,
            "outcome_accuracy": outcomes_ok / max(n, 1),
            "exit_time_mae": float(np.mean(exit_errs)) if exit_errs else float("nan"),
            "delta_V_over_V": float(np.mean(dV)) if dV else float("nan"),
            "delta_R_over_L": float(np.mean(dR)) if dR else float("nan"),
            "P_sim": P_s,
            "P_surr": P_m,
            "P_sim_wilson": [lo_s, hi_s],
            "P_surr_wilson": [lo_m, hi_m],
            "abs_delta_P": abs_delta_P_mean,  # mean |ΔP| over (N,q,σ_w) grid
            "abs_delta_P_max": abs_delta_P_max,
            "abs_delta_P_pooled": abs(P_m - P_s),
            "n_grid_points": len(grid_abs),
            "time_per_run_s": t_surr,
            "time_per_run_full_s": t_surr_full,
            "air_time_per_run_s": float(np.mean(airs)) if airs else float("nan"),
            "model_time_per_run_s": float(np.mean(models_t)) if models_t else float("nan"),
            "integ_time_per_run_s": float(np.mean(integs_t)) if integs_t else float("nan"),
            "time_sim_mean_s": t_sim,
            "sim_hardware": "c2-standard-30",
            "surr_hardware": str(device),
            "n_speedup_runs": len(full_sim_walls),
            "speedup": speedup,
            "t_end": t_end_cfg,
        }
        metrics["splits"][sp] = split_metrics
        rows.append({
            "Model": name,
            "Split": split_display_name(sp),
            "SplitKey": sp,
            "One-step MSE": osm,
            "Rollout MSE (5s)": horizon_report.get("5s", {}).get("rel_mse_mean", float("nan")),
            "Delta V/V (%)": 100 * split_metrics["delta_V_over_V"] if np.isfinite(split_metrics["delta_V_over_V"]) else float("nan"),
            "Delta(R/L) (%)": 100 * split_metrics["delta_R_over_L"] if np.isfinite(split_metrics["delta_R_over_L"]) else float("nan"),
            "|Delta P| mean": abs_delta_P_mean,
            "|Delta P| max": abs_delta_P_max,
            "Speed-up": speedup,
        })

    # C_data = sum of simulator wall_time for training runs (NaN-safe)
    splits_all = load_splits(_resolve(cfg["data"]["splits"]))
    C_data = 0.0
    n_cdata = 0
    for rid in splits_all.get("train", []):
        if rid not in store.runs:
            continue
        wt = store.get(rid)["globals"]["wall_time_s"]
        if np.isfinite(wt):
            C_data += float(wt)
            n_cdata += 1
    C_train = float(ckpt.get("meta", {}).get("train_time_s", float("nan")))
    # train_time_s is written to run_meta.json after the last ckpt save
    for meta_path in (
        Path(ckpt_path).resolve().parent / "run_meta.json",
        repo_root() / "Tema 1" / "results" / "ml" / "runs" / run_id / "run_meta.json",
    ):
        if np.isfinite(C_train):
            break
        if meta_path.exists():
            try:
                C_train = float(json.loads(meta_path.read_text()).get("train_time_s", float("nan")))
            except Exception:
                pass
    # Zero-parameter baselines / missing meta → treat train cost as 0
    if not np.isfinite(C_train):
        C_train = 0.0
    # Aggregate full-length times across splits for Q*
    # (surrogate wall scaled to each run's simulator simulated duration)
    t_sims = [m["time_sim_mean_s"] for m in metrics["splits"].values() if np.isfinite(m.get("time_sim_mean_s", float("nan")))]
    t_surrs = [m["time_per_run_full_s"] for m in metrics["splits"].values() if np.isfinite(m.get("time_per_run_full_s", float("nan")))]
    t_sim_q = float(np.mean(t_sims)) if t_sims else float("nan")
    t_surr_q = float(np.mean(t_surrs)) if t_surrs else float("nan")
    metrics["C_data"] = C_data if n_cdata else float("nan")
    metrics["C_train"] = C_train
    metrics["t_sim_per_run"] = t_sim_q
    metrics["t_surr_per_run"] = t_surr_q
    metrics["Q_star"] = q_star(C_data if n_cdata else float("nan"), C_train, t_sim_q, t_surr_q)
    metrics["sim_hardware"] = "c2-standard-30"
    metrics["surr_hardware"] = str(device)
    metrics["speedup_definition"] = (
        "(sim wall/sim-s) / (surr wall/sim-s); surr includes model + "
        "KinematicSimulation u-eval + integration; sim walls only when wall_time_s finite"
    )

    first = next(iter(metrics["splits"].values()), {})
    cost_rows.append({
        "Method": name,
        "Hardware": f"surr={device}; sim=c2-standard-30",
        "Time per run": first.get("time_per_run_full_s", float("nan")),
        "Time per P (M)": float("nan"),
        "C_data": metrics["C_data"],
        "C_train": C_train,
        "Q*": metrics["Q_star"],
        "|Delta P|": float(np.mean([m["abs_delta_P"] for m in metrics["splits"].values()])) if metrics["splits"] else float("nan"),
        "model_time_per_run": first.get("model_time_per_run_s", float("nan")),
        "air_time_per_run": first.get("air_time_per_run_s", float("nan")),
        "integ_time_per_run": first.get("integ_time_per_run_s", float("nan")),
        "t_sim_per_run": t_sim_q,
        "t_surr_per_run": t_surr_q,
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
