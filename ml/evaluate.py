"""Evaluate a checkpoint on test splits."""
from __future__ import annotations

import argparse
import json
import math
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
from ml.metrics.rollout_metrics import (
    PHYS_EDGE_TOL,
    SCHEMA_VERSION,
    aggregate_P_grid,
    aggregate_error_vs_time,
    aggregate_horizon_errors,
    aggregate_ttf,
    default_error_time_grid,
    delta_R_over_L_physical_only,
    physical_fraction_at_times,
    segment_rel_deviations,
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


def _json_safe(obj):
    """Convert numpy scalars for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _eval_per_run_dir(out_dir: Path, cfg: dict) -> Path:
    ecfg = cfg.get("eval", {})
    pr = ecfg.get("per_run_dir")
    if pr:
        p = Path(str(pr))
        return p if p.is_absolute() else Path(_resolve(p))
    return out_dir / "per_run"


def _per_run_json_path(per_run_dir: Path, split: str, run_id: str) -> Path:
    return per_run_dir / split / f"{run_id}.json"


def _time_sample_targets(
    t_release: float,
    times_s: list[float],
    t_gt: np.ndarray,
) -> dict[str, tuple[float, int]]:
    """Map label -> (absolute target time, GT frame index)."""
    out: dict[str, tuple[float, int]] = {}
    for tg in times_s:
        label = f"{tg:g}"
        t_target = t_release + float(tg)
        kg = int(np.argmin(np.abs(t_gt - t_target)))
        kg = min(max(kg, 0), len(t_gt) - 1)
        out[label] = (t_target, kg)
    return out


def _update_time_mse_samples(
    samples: dict[str, dict],
    label: str,
    t_target: float,
    kg: int,
    t: float,
    x_abs: np.ndarray,
    x_rel_gt: np.ndarray,
    sp_gt: np.ndarray,
) -> None:
    d = abs(t - t_target)
    slot = samples[label]
    if d >= slot["dist"]:
        return
    slot["dist"] = d
    sp_p = x_abs[0]
    x_rel_p = x_abs - sp_p[None, :]
    slot["rel_mse"] = float(np.mean((x_rel_p - x_rel_gt[kg]) ** 2))
    slot["spider_abs_mse"] = float(np.mean((sp_p - sp_gt[kg]) ** 2))
    slot["pred_t"] = float(t)


def _samples_to_error_dict(samples: dict[str, dict]) -> dict[str, dict]:
    return {
        label: {
            "rel_mse": slot.get("rel_mse", float("nan")),
            "spider_abs_mse": slot.get("spider_abs_mse", float("nan")),
            "pred_t": slot.get("pred_t", float("nan")),
        }
        for label, slot in samples.items()
    }


def _p_grid_key(rec: dict) -> tuple[int, float, float]:
    g = rec["globals"]
    return (
        int(g["N"]),
        round(float(g["q"]), 6),
        round(float(g["sigma_w"]), 4),
    )


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
    return_trajectory: bool = True,
    horizon_errors_s: list[float] | None = None,
    error_vs_time_grid_s: list[float] | None = None,
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

    store_traj = return_trajectory
    xs_abs: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    ts: list[float] = []
    if store_traj:
        xs_abs = [spider + x_rel.astype(np.float64)]
        vs = [v.copy()]
        ts = [float(rec["t"][t0])]
    t_release = float(rec["t"][t0])
    z0_spider = float(spider[2])
    t_gt_arr = rec["t"].astype(np.float64)
    x_rel_gt = rec["x_rel"].astype(np.float64)
    sp_gt = rec["spider_abs"].astype(np.float64)
    horizon_samples: dict[str, dict] = {}
    grid_samples: dict[str, dict] = {}
    if not store_traj:
        for label, (t_target, kg) in _time_sample_targets(
            t_release, list(horizon_errors_s or []), t_gt_arr
        ).items():
            horizon_samples[label] = {
                "dist": float("inf"), "kg": kg, "t_target": t_target,
            }
        for label, (t_target, kg) in _time_sample_targets(
            t_release, list(error_vs_time_grid_s or []), t_gt_arr
        ).items():
            grid_samples[label] = {
                "dist": float("inf"), "kg": kg, "t_target": t_target,
            }
    r_over_l_sum = 0.0
    r_over_l_n = 0
    max_edge_rel = 0.0
    ei, ej = rec["edges"][:, 0], rec["edges"][:, 1]
    edges_arr = np.asarray(rec["edges"], dtype=np.int64)
    # Compact streaming series (1 float/step; no full X/V retained).
    t_series: list[float] = []
    R_over_L_series: list[float] = []
    max_edge_rel_series: list[float] = []
    ttf_s = float("inf")
    fail_node_i: int | None = None
    fail_node_j: int | None = None
    failed = False
    r_over_l_phys_sum = 0.0
    r_over_l_phys_n = 0
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
    L_thread = float(g["L"])
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
        # Integrate relative positions with absolute velocities, then re-center so
        # node 0 stays at the origin. Without ``x_rel -= x_rel[0]``, every step
        # injects an extra ``dt * v_spider`` into all thread nodes (absolute-space
        # double-count of the spider advance). Matches batched paths below /
        # validation.rollout_mse_batched.
        x_rel, v = step_np(x_rel, v, a, dt=DT)
        spider = spider + DT * v[0].astype(np.float64)
        x_rel = (x_rel - x_rel[0:1]).astype(np.float32)
        x_abs = spider[None, :] + x_rel.astype(np.float64)
        v_hist = np.concatenate([v_hist[1:], v[None]], axis=0)
        u_hist = np.concatenate([u_hist[1:], u[None]], axis=0)
        integ_time += time.perf_counter() - t_i0

        t = t + DT
        step += 1
        if store_traj:
            xs_abs.append(x_abs.copy())
            vs.append(v.copy())
            ts.append(t)
        else:
            t_rel = float(t - t_release)
            tips_xy = x_abs[tip_ids, :2] - x_abs[0, :2]
            r_ol = float(np.mean(np.linalg.norm(tips_xy, axis=-1)) / L_thread)
            r_over_l_sum += r_ol
            r_over_l_n += 1
            finite_step = bool(np.isfinite(x_abs).all() and np.isfinite(v).all())
            if finite_step:
                devs = segment_rel_deviations(x_abs, edges_arr, l0)
                mer = float(np.max(devs)) if devs.size else 0.0
                worst = int(np.argmax(devs)) if devs.size else -1
            else:
                mer = float("inf")
                worst = -1
            if np.isfinite(mer):
                max_edge_rel = max(max_edge_rel, mer)
            t_series.append(t_rel)
            R_over_L_series.append(r_ol)
            max_edge_rel_series.append(mer if np.isfinite(mer) else float("inf"))
            if not failed:
                if (not finite_step) or mer >= PHYS_EDGE_TOL:
                    ttf_s = t_rel
                    failed = True
                    if worst >= 0 and finite_step:
                        fail_node_i = int(ei[worst])
                        fail_node_j = int(ej[worst])
                else:
                    r_over_l_phys_sum += r_ol
                    r_over_l_phys_n += 1
            for label, slot in horizon_samples.items():
                _update_time_mse_samples(
                    horizon_samples, label, slot["t_target"], slot["kg"],
                    t, x_abs, x_rel_gt, sp_gt,
                )
            for label, slot in grid_samples.items():
                _update_time_mse_samples(
                    grid_samples, label, slot["t_target"], slot["kg"],
                    t, x_abs, x_rel_gt, sp_gt,
                )

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
    r_thread = float(g.get("r", 300e-9))
    entangle_factor = float(g.get("entangle_contact_factor", 2.0))
    if store_traj:
        X = np.stack(xs_abs)
        V = np.stack(vs)
        tt = np.asarray(ts)
        sim_dur = float(tt[-1] - tt[0]) if len(tt) > 1 else 0.0
        dmin = min_interthread_distance(X[-1], rec["thread_id"])
        return {
            "status": status,
            "exit_time": exit_time,
            "entangled": bool(np.isfinite(dmin) and dmin < entangle_factor * r_thread),
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
            "edge_drift": edge_length_drift(X, rec["edges"], L_seg=l0),
            "dmin": dmin,
        }
    dmin = min_interthread_distance(x_abs, rec["thread_id"])
    sim_dur = float(t - t_release) if t > t_release else 0.0
    v_mean = (
        float((spider[2] - z0_spider) / (t - t_release))
        if t > t_release + 1e-12 else float("nan")
    )
    r_over_l = r_over_l_sum / max(r_over_l_n, 1)
    r_over_l_phys = (
        r_over_l_phys_sum / r_over_l_phys_n if r_over_l_phys_n > 0 else float("nan")
    )
    horizon_out = {
        f"{lbl}s": err
        for lbl, err in _samples_to_error_dict(horizon_samples).items()
    }
    grid_out = _samples_to_error_dict(grid_samples)
    gt_outcome = g.get("outcome")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "exit_time": exit_time,
        "outcome_sim": gt_outcome,
        "entangled": bool(np.isfinite(dmin) and dmin < entangle_factor * r_thread),
        "wall_time_s": wall,
        "air_time_s": air_time,
        "model_time_s": model_time,
        "integ_time_s": integ_time,
        "sim_duration_s": sim_dur,
        "V_mean": v_mean,
        "R_over_L": r_over_l,
        "R_over_L_physical": r_over_l_phys,
        "edge_drift": float("nan"),
        "max_edge_rel_drift": max_edge_rel,
        "dmin": dmin,
        "horizon_errors": horizon_out,
        "error_vs_time": grid_out,
        "t_release": t_release,
        "t_final": float(t),
        "ttf_s": ttf_s,
        "fail_node_i": fail_node_i,
        "fail_node_j": fail_node_j,
        "t_series": t_series,
        "R_over_L_series": R_over_L_series,
        "max_edge_rel_series": max_edge_rel_series,
        "N": int(g["N"]),
        "q": float(g["q"]),
        "sigma_w": float(g["sigma_w"]),
    }


@torch.no_grad()
def full_rollout_streaming(
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
    horizon_errors_s: list[float] | None = None,
    error_vs_time_grid_s: list[float] | None = None,
) -> dict:
    """Full Alg.2 rollout retaining only compact online metrics (OOM-safe)."""
    _ = include_air_time  # same air path as surrogate; kept for API parity
    return full_rollout_surrogate(
        model, rec, norm, device,
        C=C, t_end=t_end, include_air_time=True,
        node_features=node_features,
        globals_fields=globals_fields,
        edge_sets=edge_sets,
        radius=radius,
        normalize_features=normalize_features,
        return_trajectory=False,
        horizon_errors_s=horizon_errors_s,
        error_vs_time_grid_s=error_vs_time_grid_s,
    )


@torch.no_grad()
def full_rollout_batched(
    model,
    store: BallooningRunStore,
    run_ids: list[str],
    device: torch.device,
    *,
    C: int = 5,
    t_end: float = 60.0,
    node_features: list[str] | None = None,
    globals_fields: list[str] | None = None,
    edge_sets: list[str] | None = None,
    radius: float = 0.1,
    normalize_features: bool = True,
    batch_runs: int | None = None,
) -> dict[str, dict]:
    """Full-length Alg.2 rollouts batched by n_nodes (same exit criteria as single).

    Groups runs with equal node count, rolls open-loop until each exits
    (z>=h rise, z<=0 fall) or t_end. Returns ``{run_id: result_dict}`` with the
    same keys as :func:`full_rollout_surrogate`.
    """
    from torch_geometric.data import Batch, Data
    from ml.data.edge_features import (
        build_edge_sets_gns,
        interaction_edge_attr,
        material_edge_attr,
    )
    from ml.data.graph import build_edge_sets, material_edges
    from ml.data.integrator import step_torch
    from ml.data.ks_torch import TurbModeCache, ks_u_torch
    from ml.validation import _norm_blocks_torch

    model.eval()
    node_features = list(node_features or ["v_hist", "u", "charge", "node_type"])
    globals_fields = list(globals_fields or [])
    edge_sets = list(edge_sets or model.required_edges() or ["material"])
    need_u = True
    need_unext = "u_next" in node_features
    mode_cache = TurbModeCache()
    norm = store.norm
    a_mean = torch.as_tensor(norm["a"]["mean"], device=device, dtype=torch.float32)
    a_std = torch.as_tensor(norm["a"]["std"], device=device, dtype=torch.float32)
    u_mean = norm.get("u", {}).get("mean")
    u_std = norm.get("u", {}).get("std")
    v_mean = norm.get("v", {}).get("mean")
    v_std = norm.get("v", {}).get("std")
    Bmax = int(batch_runs) if batch_runs else (4 if device.type == "cpu" else 16)
    max_steps = int(t_end / DT) + 5

    by_n: dict[int, list[str]] = {}
    for rid in run_ids:
        if rid not in store:
            continue
        rec = store.get(rid)
        if int(rec["v"].shape[0]) < C + 2:
            continue
        by_n.setdefault(int(rec["v"].shape[1]), []).append(rid)

    out: dict[str, dict] = {}
    for n_nodes, rids in by_n.items():
        for start in range(0, len(rids), Bmax):
            batch_ids = rids[start : start + Bmax]
            recs = [store.get(rid) for rid in batch_ids]
            B = len(recs)
            modes = [mode_cache.get(r["globals"]) for r in recs]
            h = torch.tensor(
                [float(r["globals"]["h"]) for r in recs], device=device, dtype=torch.float32
            )
            t0i = C
            x = torch.stack([
                torch.from_numpy(r["x_rel"][t0i].astype(np.float32)) for r in recs
            ]).to(device)
            spider = torch.stack([
                torch.from_numpy(r["spider_abs"][t0i].astype(np.float32)) for r in recs
            ]).to(device)
            v = torch.stack([
                torch.from_numpy(r["v"][t0i].astype(np.float32)) for r in recs
            ]).to(device)
            v_hist = torch.stack([
                torch.from_numpy(r["v"][t0i - C + 1 : t0i + 1].astype(np.float32))
                for r in recs
            ]).to(device)
            t_vals = torch.tensor(
                [float(np.asarray(r["t"][t0i])) for r in recs],
                device=device, dtype=torch.float32,
            )
            nt = torch.stack([
                torch.from_numpy(r["node_type"].astype(np.int64)) for r in recs
            ]).to(device)
            tid = torch.stack([
                torch.from_numpy(r["thread_id"].astype(np.int64)) for r in recs
            ]).to(device)
            charge = torch.stack([
                torch.from_numpy(r["charge"].astype(np.float32)) for r in recs
            ]).to(device)
            l0 = torch.tensor(
                [float(r["globals"]["L"]) / max(int(r["globals"]["N_t"]), 1) for r in recs],
                device=device, dtype=torch.float32,
            )
            edges_mat = [np.asarray(r["edges"], dtype=np.int64) for r in recs]
            tip_ids = [np.where(r["node_type"] == 2)[0] for r in recs]

            active = torch.ones(B, dtype=torch.bool, device=device)
            status = ["timeout"] * B
            exit_time = [float("nan")] * B
            xs_abs = [[] for _ in range(B)]
            vs_hist = [[] for _ in range(B)]
            ts_hist = [[] for _ in range(B)]
            for b in range(B):
                sp0 = spider[b].detach().cpu().numpy().astype(np.float64)
                xr0 = x[b].detach().cpu().numpy().astype(np.float64)
                xs_abs[b].append(sp0 + xr0)
                vs_hist[b].append(v[b].detach().cpu().numpy().astype(np.float32))
                ts_hist[b].append(float(t_vals[b].item()))

            air_time = 0.0
            model_time = 0.0
            integ_time = 0.0
            wall0 = time.perf_counter()
            step = 0
            while step < max_steps and bool(active.any()):
                # Soft t_end: deactivate runs that already reached t_end
                active = active & (t_vals < float(t_end) - 1e-12)
                if not bool(active.any()):
                    break
                pos_abs = x + spider.unsqueeze(1)
                u_t = None
                u_next = None
                if need_u or need_unext:
                    t_air0 = time.perf_counter()
                    if need_u:
                        u_t = ks_u_torch(pos_abs, t_vals, modes)
                    if need_unext:
                        u_next = ks_u_torch(pos_abs, t_vals + float(DT), modes)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elif device.type == "mps":
                        torch.mps.synchronize()
                    air_time += time.perf_counter() - t_air0

                graphs = []
                active_idx = [b for b in range(B) if bool(active[b].item())]
                for b in active_idx:
                    feats = []
                    for name in node_features:
                        if name == "v_hist":
                            feats.append(
                                _norm_blocks_torch(
                                    v_hist[b], v_mean, v_std, do_norm=normalize_features
                                )
                            )
                        elif name == "u":
                            assert u_t is not None
                            ub = u_t[b]
                            if normalize_features and u_mean is not None:
                                ub = (ub - torch.as_tensor(u_mean, device=device)) / torch.as_tensor(
                                    u_std, device=device
                                ).clamp_min(1e-8)
                            feats.append(ub)
                        elif name == "u_next":
                            assert u_next is not None
                            ub = u_next[b]
                            if normalize_features and u_mean is not None:
                                ub = (ub - torch.as_tensor(u_mean, device=device)) / torch.as_tensor(
                                    u_std, device=device
                                ).clamp_min(1e-8)
                            feats.append(ub)
                        elif name == "charge":
                            feats.append(charge[b][:, None])
                        elif name == "node_type":
                            feats.append(
                                torch.nn.functional.one_hot(nt[b], num_classes=3).float()
                            )
                        else:
                            raise KeyError(name)
                    g = recs[b]["globals"]
                    for gf in globals_fields:
                        if gf == "qE":
                            val = float(g["q"]) * float(g["E"])
                        elif gf == "m":
                            val = float(g["m"])
                        elif gf == "N":
                            val = float(g["N"])
                        else:
                            raise KeyError(gf)
                        feats.append(
                            torch.full((n_nodes, 1), val, device=device, dtype=torch.float32)
                        )
                    x_feat = torch.cat(feats, dim=-1)
                    pos_b = x[b]
                    pos_abs_b = pos_abs[b]
                    if "interaction" in edge_sets:
                        edge_dict = build_edge_sets_gns(
                            edge_sets,
                            edges_material=edges_mat[b],
                            pos=pos_abs_b.detach().cpu(),
                            thread_id=tid[b].detach().cpu(),
                            node_type=nt[b].detach().cpu(),
                            radius=radius,
                        )
                        inter_flags = edge_dict.pop("_interaction_flags")
                    else:
                        edge_dict = build_edge_sets(
                            edge_sets,
                            edges_material=edges_mat[b],
                            pos=pos_abs_b.detach().cpu(),
                            thread_id=tid[b].detach().cpu(),
                            node_type=nt[b].detach().cpu(),
                            radius=radius,
                        )
                        inter_flags = torch.zeros(0)
                    e_mat = edge_dict.get("material", material_edges(edges_mat[b])).to(device)
                    data = Data(
                        x=x_feat,
                        pos=pos_b,
                        edge_index=e_mat,
                        edge_index_material=e_mat,
                        node_type=nt[b],
                        thread_id=tid[b],
                        charge=charge[b],
                        v_t=v[b],
                        u_t=u_t[b] if u_t is not None else torch.zeros_like(v[b]),
                        num_nodes=n_nodes,
                    )
                    for name, eidx in edge_dict.items():
                        eidx_d = eidx.to(device)
                        setattr(data, f"edge_index_{name}", eidx_d)
                        if name == "material":
                            data.edge_attr_material = material_edge_attr(
                                pos_b, eidx_d, l0=float(l0[b])
                            )
                        elif name == "interaction":
                            data.edge_attr_interaction = interaction_edge_attr(
                                pos_b, eidx_d, charge[b], inter_flags.to(device)
                            )
                    graphs.append(data)

                t_m0 = time.perf_counter()
                batch = Batch.from_data_list(graphs)
                pred = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                model_time += time.perf_counter() - t_m0
                a = pred * a_std + a_mean
                ptr = batch.ptr

                t_i0 = time.perf_counter()
                for bi, b in enumerate(active_idx):
                    ab = a[ptr[bi] : ptr[bi + 1]]
                    xb, vb = step_torch(x[b], v[b], ab, dt=DT)
                    spider[b] = spider[b] + float(DT) * vb[0]
                    x[b] = xb - xb[0:1]
                    v[b] = vb
                    v_hist[b] = torch.cat([v_hist[b, 1:], v[b].unsqueeze(0)], dim=0)
                    t_vals[b] = t_vals[b] + float(DT)
                    sp_np = spider[b].detach().cpu().numpy().astype(np.float64)
                    xr_np = x[b].detach().cpu().numpy().astype(np.float64)
                    xs_abs[b].append(sp_np + xr_np)
                    vs_hist[b].append(v[b].detach().cpu().numpy().astype(np.float32))
                    ts_hist[b].append(float(t_vals[b].item()))
                    z = float(spider[b, 2].item())
                    if z >= float(h[b].item()):
                        status[b] = "rise"
                        exit_time[b] = float(t_vals[b].item())
                        active[b] = False
                    elif z <= 0.0:
                        status[b] = "fall"
                        exit_time[b] = float(t_vals[b].item())
                        active[b] = False
                integ_time += time.perf_counter() - t_i0
                step += 1

            wall = time.perf_counter() - wall0
            # Equal share of batch wall (throughput accounting for speed-up).
            share = 1.0 / max(B, 1)
            for b, rid in enumerate(batch_ids):
                X = np.stack(xs_abs[b])
                V = np.stack(vs_hist[b])
                tt = np.asarray(ts_hist[b], dtype=np.float64)
                sim_dur = float(tt[-1] - tt[0]) if len(tt) > 1 else 0.0
                g = recs[b]["globals"]
                dmin = min_interthread_distance(X[-1], recs[b]["thread_id"])
                r_thread = float(g.get("r", 300e-9))
                entangle_factor = float(g.get("entangle_contact_factor", 2.0))
                out[rid] = {
                    "status": status[b],
                    "exit_time": exit_time[b],
                    "entangled": bool(np.isfinite(dmin) and dmin < entangle_factor * r_thread),
                    "t": tt,
                    "x": X,
                    "v": V,
                    "wall_time_s": wall * share,
                    "air_time_s": air_time * share,
                    "model_time_s": model_time * share,
                    "integ_time_s": integ_time * share,
                    "sim_duration_s": sim_dur,
                    "V_mean": mean_vertical_spider_velocity(X, tt),
                    "R_over_L": opening_R_over_L(X, tip_ids[b], L=float(g["L"])),
                    "edge_drift": edge_length_drift(
                        X, recs[b]["edges"], L_seg=float(g["L"]) / int(g["N_t"])
                    ),
                    "dmin": dmin,
                }
    return out


def _append_horizon_errors(
    pred: dict,
    rec: dict,
    C: int,
    horizon_errors_s: list[float],
    horizon_rel: dict[str, list],
    horizon_abs: dict[str, list],
) -> None:
    if "horizon_errors" in pred:
        for hs in horizon_errors_s:
            key = f"{hs:g}s"
            he = pred["horizon_errors"].get(key, {})
            horizon_rel[key].append(float(he.get("rel_mse", float("nan"))))
            horizon_abs[key].append(float(he.get("spider_abs_mse", float("nan"))))
        return
    t_gt = rec["t"].astype(np.float64)
    X = pred["x"]
    tt = pred["t"]
    t_release = float(rec["t"][C])
    x_abs_gt = rec["spider_abs"][:, None, :] + rec["x_rel"].astype(np.float64)
    for hs in horizon_errors_s:
        t_target = t_release + float(hs)
        k = int(np.argmin(np.abs(tt - t_target)))
        kg = int(np.argmin(np.abs(t_gt - t_target)))
        kg = min(max(kg, 0), x_abs_gt.shape[0] - 1)
        sp_p = X[k, 0]
        x_rel_p = X[k] - sp_p
        key = f"{hs:g}s"
        horizon_rel[key].append(float(np.mean((x_rel_p - rec["x_rel"][kg]) ** 2)))
        horizon_abs[key].append(float(np.mean((sp_p - rec["spider_abs"][kg]) ** 2)))


def _compact_pred_for_disk(pred: dict) -> dict:
    """Drop full trajectories; keep compact schema-v2 series (t / R/L / max_edge)."""
    drop = {"x", "v", "t"}
    return _json_safe({k: v for k, v in pred.items() if k not in drop})


def _per_run_schema_ok(pred: dict | None) -> bool:
    """Resume only schema_version >= 2 with ttf_s and R_over_L_series."""
    if not isinstance(pred, dict):
        return False
    if int(pred.get("schema_version", 0) or 0) < SCHEMA_VERSION:
        return False
    if "ttf_s" not in pred or "R_over_L_series" not in pred:
        return False
    return True


def _load_or_run_full_rollout(
    *,
    rid: str,
    sp: str,
    model,
    store: BallooningRunStore,
    device: torch.device,
    cfg: dict,
    dcfg: dict,
    edge_sets: list[str],
    per_run_dir: Path,
    stream_full: bool,
    C: int,
    t_end_cfg: float,
    horizon_errors_s: list[float],
    error_vs_time_grid_s: list[float],
) -> dict | None:
    path = _per_run_json_path(per_run_dir, sp, rid)
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            if _per_run_schema_ok(cached):
                return cached
        except Exception:
            pass
    if rid not in store:
        return None
    rec = store.get(rid)
    if int(rec["v"].shape[0]) < C + 2:
        return None
    if stream_full:
        pred = full_rollout_streaming(
            model, rec, store.norm, device,
            C=C,
            t_end=t_end_cfg,
            node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=edge_sets,
            radius=dcfg.get("radius", 0.1),
            normalize_features=dcfg.get("normalize_features", True),
            horizon_errors_s=horizon_errors_s,
            error_vs_time_grid_s=error_vs_time_grid_s,
        )
    else:
        batch_runs = cfg.get("eval", {}).get("full_rollout_batch_runs")
        batched = full_rollout_batched(
            model, store, [rid], device,
            C=C,
            t_end=t_end_cfg,
            node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=edge_sets,
            radius=dcfg.get("radius", 0.1),
            normalize_features=dcfg.get("normalize_features", True),
            batch_runs=int(batch_runs) if batch_runs else 1,
        )
        pred = batched.get(rid)
        if pred is None:
            return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_compact_pred_for_disk(pred), f, indent=2)
    return pred


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

    ecfg = cfg.get("eval", {})
    eval_seed = int(ecfg.get("seed", 42))
    one_step_n_frames = int(ecfg.get("one_step_n_frames", 100_000))
    horizon_errors_s = list(ecfg.get("horizon_errors_s", [0.2, 1.0, 5.0, 10.0]))
    t_end_cfg = float(ecfg.get("t_end", 60.0))
    raw_grid = ecfg.get("error_vs_time_grid_s", None)
    if raw_grid is None:
        error_vs_time_grid_s = default_error_time_grid(t_end_cfg)
    else:
        error_vs_time_grid_s = list(raw_grid)
    stream_full = bool(ecfg.get("stream_full_rollouts", True))
    per_run_dir = _eval_per_run_dir(out_dir, cfg)
    per_run_dir.mkdir(parents=True, exist_ok=True)
    p_grid_rows_all: list[dict] = []
    evt_rows_all: list[dict] = []

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
        max_full = ecfg.get("max_full_runs")
        ids_full = ids if not max_full else ids[: int(max_full)]
        C = int(dcfg.get("C", 5))
        per_run_acc: list[dict] = []
        gt_R_phys: list[float] = []
        for rid in ids_full:
            pred = _load_or_run_full_rollout(
                rid=rid,
                sp=sp,
                model=model,
                store=store,
                device=device,
                cfg=cfg,
                dcfg=dcfg,
                edge_sets=edge_sets,
                per_run_dir=per_run_dir,
                stream_full=stream_full,
                C=C,
                t_end_cfg=t_end_cfg,
                horizon_errors_s=horizon_errors_s,
                error_vs_time_grid_s=error_vs_time_grid_s,
            )
            if pred is None:
                continue
            rec = store.get(rid)
            gt_out = rec["globals"]["outcome"]
            # Enrich for shared aggregators (schema v2 already stores most of these).
            if "horizon_errors" not in pred and "x" in pred:
                _he_rel: dict[str, list] = {f"{hs:g}s": [] for hs in horizon_errors_s}
                _he_abs: dict[str, list] = {f"{hs:g}s": [] for hs in horizon_errors_s}
                _append_horizon_errors(
                    pred, rec, C, horizon_errors_s, _he_rel, _he_abs,
                )
                pred = dict(pred)
                pred["horizon_errors"] = {
                    k: {
                        "rel_mse": _he_rel[k][0] if _he_rel[k] else float("nan"),
                        "spider_abs_mse": _he_abs[k][0] if _he_abs[k] else float("nan"),
                    }
                    for k in _he_rel
                }
            run_row = dict(pred)
            run_row.setdefault("status", pred.get("status"))
            run_row.setdefault("outcome", gt_out)
            run_row.setdefault("status_sim", gt_out)
            run_row.setdefault("N", int(rec["globals"]["N"]))
            run_row.setdefault("q", float(rec["globals"]["q"]))
            run_row.setdefault("sigma_w", float(rec["globals"]["sigma_w"]))
            run_row.setdefault("ttf_s", float("nan"))
            per_run_acc.append(run_row)
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
            # GT R/L over physical window matching surrogate TTF when series exist.
            ttf = float(pred.get("ttf_s", float("inf")))
            t_gt = rec["t"].astype(np.float64)
            t_rel_gt = t_gt - float(rec["t"][C])
            L_th = float(rec["globals"]["L"])
            if np.isfinite(ttf) or math.isinf(ttf):
                mask = t_rel_gt >= 0
                if np.isfinite(ttf):
                    mask = mask & (t_rel_gt <= ttf + 1e-12)
                if np.any(mask):
                    R_phys_gt = opening_R_over_L(
                        x_abs[mask], tips, L=L_th,
                    )
                else:
                    R_phys_gt = R_gt
            else:
                R_phys_gt = R_gt
            gt_R_phys.append(float(R_phys_gt))
            if abs(V_gt) > 1e-12:
                dV.append(abs(pred["V_mean"] - V_gt) / abs(V_gt))
            takeoff_s.append(1 if gt_out == "rise" else 0)
            takeoff_m.append(1 if pred["status"] == "rise" else 0)
            gkey = (
                int(rec["globals"]["N"]),
                round(float(rec["globals"]["q"]), 6),
                round(float(rec["globals"]["sigma_w"]), 4),
            )
            delta_P_by_grid.setdefault(gkey, []).append(
                (1 if pred["status"] == "rise" else 0) - (1 if gt_out == "rise" else 0)
            )

        for row in aggregate_P_grid(per_run_acc):
            p_grid_rows_all.append({
                "Model": name,
                "Split": split_display_name(sp),
                "SplitKey": sp,
                **row,
            })
        for row in aggregate_error_vs_time(per_run_acc, error_vs_time_grid_s):
            evt_rows_all.append({
                "Model": name,
                "Split": split_display_name(sp),
                "SplitKey": sp,
                **row,
            })

        n = len(takeoff_s)
        k_s, k_m = sum(takeoff_s), sum(takeoff_m)
        P_s, P_m = k_s / max(n, 1), k_m / max(n, 1)
        lo_s, hi_s = wilson_interval(k_s, n)
        lo_m, hi_m = wilson_interval(k_m, n)
        grid_abs = [abs(float(np.mean(v))) for v in delta_P_by_grid.values() if v]
        abs_delta_P_mean = float(np.mean(grid_abs)) if grid_abs else float("nan")
        abs_delta_P_max = float(np.max(grid_abs)) if grid_abs else float("nan")
        horizon_report = aggregate_horizon_errors(per_run_acc, horizon_errors_s)
        ttf_report = aggregate_ttf(per_run_acc)
        phys_at = physical_fraction_at_times(per_run_acc, times=(1.0, 5.0, 10.0))
        dR_phys = delta_R_over_L_physical_only(per_run_acc, gt_R_phys)

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
            "ttf": ttf_report,
            "physical_fraction_at_times": phys_at,
            "outcome_accuracy": outcomes_ok / max(n, 1),
            "exit_time_mae": float(np.mean(exit_errs)) if exit_errs else float("nan"),
            "delta_V_over_V": float(np.mean(dV)) if dV else float("nan"),
            "delta_R_over_L": float(dR_phys["delta_R_over_L_mean"]),
            "delta_R_over_L_n_runs_used": int(dR_phys["n_runs_used"]),
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
    metrics["eval_stream_full_rollouts"] = stream_full
    metrics["per_run_dir"] = str(per_run_dir)

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
    if p_grid_rows_all:
        pd.DataFrame(p_grid_rows_all).to_csv(out_dir / "tab_P_grid.csv", index=False)
    if evt_rows_all:
        pd.DataFrame(evt_rows_all).to_csv(out_dir / "tab_error_vs_time.csv", index=False)
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
