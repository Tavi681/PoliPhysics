"""Cheap tiered validation: sampled one-step + stratified batched rollouts."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from ml.data.dataset import BallooningDataset, BallooningRunStore
from ml.data.edge_features import (
    build_edge_sets_gns,
    interaction_edge_attr,
    material_edge_attr,
)
from ml.data.frames import DT
from ml.data.graph import build_edge_sets, material_edges
from ml.data.integrator import step_torch
from ml.data.ks_torch import TurbModeCache, ks_u_torch
from ml.data.normalize import normalize


def stratified_val_runs(
    store: BallooningRunStore,
    run_ids: Sequence[str],
    *,
    n_runs: int = 32,
    seed: int = 0,
) -> list[str]:
    """Fixed stratified subset by (N, sigma_w)."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[str]] = {}
    for rid in run_ids:
        if rid not in store:
            continue
        # Prefer manifest metadata to avoid loading heavy arrays.
        meta = None
        for r in store.manifest["records"]:
            if r["run_id"] == rid:
                meta = r
                break
        if meta is None:
            g = store.get(rid)["globals"]
            key = (int(g["N"]), round(float(g["sigma_w"]), 4))
        else:
            key = (int(meta["N"]), round(float(meta["sigma_w"]), 4))
        buckets.setdefault(key, []).append(rid)
    keys = sorted(buckets.keys())
    if not keys:
        return []
    chosen: list[str] = []
    # Round-robin across strata for balance.
    ptr = {k: 0 for k in keys}
    for k in keys:
        rng.shuffle(buckets[k])
    while len(chosen) < n_runs:
        progressed = False
        for k in keys:
            i = ptr[k]
            if i < len(buckets[k]):
                chosen.append(buckets[k][i])
                ptr[k] = i + 1
                progressed = True
                if len(chosen) >= n_runs:
                    break
        if not progressed:
            break
    return chosen


def fixed_val_frame_indices(
    dataset: BallooningDataset,
    *,
    n_frames: int = 256,
    seed: int = 0,
) -> list[int]:
    """Fixed random sample of dataset indices (reproducible)."""
    n = len(dataset)
    if n == 0:
        return []
    rng = np.random.default_rng(seed)
    k = min(int(n_frames), n)
    return sorted(rng.choice(n, size=k, replace=False).tolist())


@torch.no_grad()
def one_step_mse_sample(
    model,
    dataset: BallooningDataset,
    indices: Sequence[int],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> float:
    """One-step MSE on a fixed index sample (GPU/MPS only; no full val loader)."""
    model.eval()
    if not indices:
        return float("nan")
    total = 0.0
    n = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        graphs = [dataset[i] for i in chunk]
        batch = Batch.from_data_list(graphs).to(device)
        pred = model(batch)
        err = (pred - batch.y).pow(2).mean()
        total += float(err) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def one_step_mse_by_type(
    model,
    dataset: BallooningDataset,
    indices: Sequence[int],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> dict[str, float]:
    """Overall + per node_type (spider/interior/tip) one-step MSE on fixed indices."""
    model.eval()
    names = ("spider", "interior", "tip")
    sum_err = {n: 0.0 for n in names}
    sum_err["overall"] = 0.0
    count = {n: 0 for n in names}
    count["overall"] = 0
    if not indices:
        return {k: float("nan") for k in ("overall",) + names}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        graphs = [dataset[i] for i in chunk]
        batch = Batch.from_data_list(graphs).to(device)
        pred = model(batch)
        err = (pred - batch.y).pow(2)  # (N, 3)
        per_node = err.mean(dim=-1)
        sum_err["overall"] += float(per_node.sum())
        count["overall"] += int(per_node.numel())
        nt = batch.node_type
        for typ, name in enumerate(names):
            m = nt == typ
            if not bool(m.any()):
                continue
            sum_err[name] += float(per_node[m].sum())
            count[name] += int(m.sum().item())
    return {k: (sum_err[k] / max(count[k], 1)) for k in ("overall",) + names}


def _norm_blocks_torch(v_hist: torch.Tensor, mean, std, *, do_norm: bool) -> torch.Tensor:
    """v_hist (C,n,3) → (n, 3C)."""
    C, n, _ = v_hist.shape
    if not do_norm or mean is None:
        return v_hist.reshape(n, -1)
    mean_t = torch.as_tensor(mean, device=v_hist.device, dtype=v_hist.dtype)
    std_t = torch.as_tensor(std, device=v_hist.device, dtype=v_hist.dtype)
    blocks = [(v_hist[k] - mean_t) / std_t.clamp_min(1e-8) for k in range(C)]
    return torch.cat(blocks, dim=-1)


@torch.no_grad()
def rollout_mse_batched(
    model,
    store: BallooningRunStore,
    run_ids: Sequence[str],
    device: torch.device,
    norm: dict,
    *,
    K: int = 20,
    C: int = 5,
    node_features=None,
    globals_fields=None,
    edge_sets=None,
    radius: float = 0.1,
    normalize_features: bool = True,
    mode_cache: TurbModeCache | None = None,
) -> tuple[float, dict]:
    """Batched open-loop rollout MSE with device KS for u / u_next.

    Returns (mse, timing_dict) where timing includes air_time_s (counted).
    """
    model.eval()
    node_features = list(node_features or ["v_hist", "u", "charge"])
    globals_fields = list(globals_fields or [])
    edge_sets = list(edge_sets or model.required_edges() or ["material"])
    need_u = True  # always evaluate u for baselines (follow_air) + feature builders
    need_unext = "u_next" in node_features
    mode_cache = mode_cache or TurbModeCache()
    a_mean = torch.as_tensor(norm["a"]["mean"], device=device, dtype=torch.float32)
    a_std = torch.as_tensor(norm["a"]["std"], device=device, dtype=torch.float32)
    u_mean = norm.get("u", {}).get("mean")
    u_std = norm.get("u", {}).get("std")
    v_mean = norm.get("v", {}).get("mean")
    v_std = norm.get("v", {}).get("std")

    errs: list[float] = []
    air_time = 0.0
    model_time = 0.0
    graph_time = 0.0
    n_steps = 0

    # Group runs by n_nodes for true batching
    by_n: dict[int, list[str]] = {}
    for rid in run_ids:
        if rid not in store:
            continue
        rec = store.get(rid)
        T = int(rec["v"].shape[0])
        if C + K >= T:
            continue
        n = int(rec["v"].shape[1])
        by_n.setdefault(n, []).append(rid)

    for n_nodes, rids in by_n.items():
        # Process in mini-batches of runs (memory)
        Bmax = 8 if device.type == "cpu" else 16
        for start in range(0, len(rids), Bmax):
            batch_ids = rids[start : start + Bmax]
            recs = [store.get(rid) for rid in batch_ids]
            B = len(recs)
            modes = [mode_cache.get(r["globals"]) for r in recs]
            t0 = C
            x = torch.stack([
                torch.from_numpy(r["x_rel"][t0].astype(np.float32)) for r in recs
            ]).to(device)  # (B,n,3)
            spider = torch.stack([
                torch.from_numpy(r["spider_abs"][t0].astype(np.float32)) for r in recs
            ]).to(device)
            v = torch.stack([
                torch.from_numpy(r["v"][t0].astype(np.float32)) for r in recs
            ]).to(device)
            v_hist = torch.stack([
                torch.from_numpy(r["v"][t0 - C + 1 : t0 + 1].astype(np.float32))
                for r in recs
            ]).to(device)  # (B,C,n,3)
            t_vals = torch.tensor(
                [float(np.asarray(r["t"][t0])) for r in recs],
                device=device,
                dtype=torch.float32,
            )
            x_gt_seq = torch.stack([
                torch.from_numpy(r["x_rel"][t0 + 1 : t0 + K + 1].astype(np.float32))
                for r in recs
            ]).to(device)  # (B,K,n,3)

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
                device=device,
                dtype=torch.float32,
            )
            edges_mat = [np.asarray(r["edges"], dtype=np.int64) for r in recs]

            for k in range(K):
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

                t_graph0 = time.perf_counter()
                graphs = []
                for b in range(B):
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
                batch = Batch.from_data_list(graphs)
                graph_time += time.perf_counter() - t_graph0

                t_m0 = time.perf_counter()
                pred = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                model_time += time.perf_counter() - t_m0
                a = pred * a_std + a_mean
                ptr = batch.ptr
                for b in range(B):
                    ab = a[ptr[b] : ptr[b + 1]]
                    xb, vb = step_torch(x[b], v[b], ab, dt=DT)
                    spider[b] = spider[b] + float(DT) * vb[0]
                    # Keep spider-relative frame with node 0 at origin (evaluate.py).
                    x[b] = xb - xb[0:1]
                    v[b] = vb
                    errs.append(float((x[b] - x_gt_seq[b, k]).pow(2).mean()))
                v_hist = torch.cat([v_hist[:, 1:], v.unsqueeze(1)], dim=1)
                t_vals = t_vals + float(DT)
                n_steps += B

    timing = {
        "air_time_s": air_time,
        "model_time_s": model_time,
        "graph_time_s": graph_time,
        "n_steps": n_steps,
        "n_runs": sum(len(v) for v in by_n.values()),
    }
    return (float(np.mean(errs)) if errs else float("nan")), timing
