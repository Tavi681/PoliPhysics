"""Per-step rollout errors + leakage check (relative vs spider-abs).

Confirms open-loop rollout: GT only for the first C frames (init at t0=C);
later states from model predictions; air from ks_torch (not stored u).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from ml.data.splits import load_splits
from ml.models import build_model
from ml.models.baselines import ConstantVelocity, FollowAir
from ml.train import _device, _resolve, _set_seed
from ml.utils.config import load_yaml
from ml.validation import stratified_val_runs, _norm_blocks_torch


@torch.no_grad()
def rollout_per_step(
    model,
    store,
    run_ids,
    device,
    norm,
    *,
    K,
    C,
    node_features,
    globals_fields,
    edge_sets,
    radius,
    normalize_features,
):
    model.eval()
    node_features = list(node_features or ["v_hist", "u", "charge"])
    globals_fields = list(globals_fields or [])
    edge_sets = list(edge_sets or ["material"])
    need_unext = "u_next" in node_features
    mode_cache = TurbModeCache()
    a_mean = torch.as_tensor(norm["a"]["mean"], device=device, dtype=torch.float32)
    a_std = torch.as_tensor(norm["a"]["std"], device=device, dtype=torch.float32)
    u_mean = norm.get("u", {}).get("mean")
    u_std = norm.get("u", {}).get("std")
    v_mean = norm.get("v", {}).get("mean")
    v_std = norm.get("v", {}).get("std")

    rel = np.zeros(K, dtype=np.float64)
    spider_abs = np.zeros(K, dtype=np.float64)
    n_used = 0
    for rid in run_ids:
        rec = store.get(rid)
        T = int(rec["v"].shape[0])
        if C + K >= T:
            continue
        t0 = C
        # Leakage check: init state is GT only.
        x = torch.from_numpy(rec["x_rel"][t0].astype(np.float32)).to(device)
        spider = torch.from_numpy(rec["spider_abs"][t0].astype(np.float32)).to(device)
        v = torch.from_numpy(rec["v"][t0].astype(np.float32)).to(device)
        v_hist = torch.from_numpy(rec["v"][t0 - C + 1 : t0 + 1].astype(np.float32)).to(device)
        t_val = float(np.asarray(rec["t"][t0]))
        nt = torch.from_numpy(rec["node_type"].astype(np.int64)).to(device)
        tid = torch.from_numpy(rec["thread_id"].astype(np.int64)).to(device)
        charge = torch.from_numpy(rec["charge"].astype(np.float32)).to(device)
        edges = np.asarray(rec["edges"], dtype=np.int64)
        g = rec["globals"]
        l0 = float(g["L"]) / max(int(g["N_t"]), 1)
        modes = mode_cache.get(g)
        n = x.shape[0]
        x_gt = torch.from_numpy(rec["x_rel"][t0 + 1 : t0 + K + 1].astype(np.float32)).to(device)
        sp_gt = torch.from_numpy(rec["spider_abs"][t0 + 1 : t0 + K + 1].astype(np.float32)).to(device)

        for k in range(K):
            pos_abs = x + spider
            # Air from ks_torch on *predicted* absolute positions — not stored u.
            u_t = ks_u_torch(pos_abs.unsqueeze(0), torch.tensor([t_val], device=device), [modes])[0]
            u_next = None
            if need_unext:
                u_next = ks_u_torch(
                    pos_abs.unsqueeze(0),
                    torch.tensor([t_val + DT], device=device),
                    [modes],
                )[0]
            feats = []
            for name in node_features:
                if name == "v_hist":
                    feats.append(
                        _norm_blocks_torch(v_hist, v_mean, v_std, do_norm=normalize_features)
                    )
                elif name == "u":
                    ub = u_t
                    if normalize_features and u_mean is not None:
                        ub = (ub - torch.as_tensor(u_mean, device=device)) / torch.as_tensor(
                            u_std, device=device
                        ).clamp_min(1e-8)
                    feats.append(ub)
                elif name == "u_next":
                    ub = u_next
                    if normalize_features and u_mean is not None:
                        ub = (ub - torch.as_tensor(u_mean, device=device)) / torch.as_tensor(
                            u_std, device=device
                        ).clamp_min(1e-8)
                    feats.append(ub)
                elif name == "charge":
                    feats.append(charge[:, None])
                elif name == "node_type":
                    feats.append(torch.nn.functional.one_hot(nt, num_classes=3).float())
                else:
                    raise KeyError(name)
            for gf in globals_fields:
                if gf == "qE":
                    val = float(g["q"]) * float(g["E"])
                elif gf == "m":
                    val = float(g["m"])
                elif gf == "N":
                    val = float(g["N"])
                else:
                    raise KeyError(gf)
                feats.append(torch.full((n, 1), val, device=device, dtype=torch.float32))
            x_feat = torch.cat(feats, dim=-1)
            if "interaction" in edge_sets:
                edge_dict = build_edge_sets_gns(
                    edge_sets, edges_material=edges, pos=pos_abs.cpu(),
                    thread_id=tid.cpu(), node_type=nt.cpu(), radius=radius,
                )
                inter_flags = edge_dict.pop("_interaction_flags")
            else:
                edge_dict = build_edge_sets(
                    edge_sets, edges_material=edges, pos=pos_abs.cpu(),
                    thread_id=tid.cpu(), node_type=nt.cpu(), radius=radius,
                )
                inter_flags = torch.zeros(0)
            e_mat = edge_dict.get("material", material_edges(edges)).to(device)
            data = Data(
                x=x_feat, pos=x, edge_index=e_mat, edge_index_material=e_mat,
                node_type=nt, thread_id=tid, charge=charge, v_t=v, u_t=u_t, num_nodes=n,
            )
            for name, eidx in edge_dict.items():
                eidx_d = eidx.to(device)
                setattr(data, f"edge_index_{name}", eidx_d)
                if name == "material":
                    data.edge_attr_material = material_edge_attr(x, eidx_d, l0=l0)
                elif name == "interaction":
                    data.edge_attr_interaction = interaction_edge_attr(
                        x, eidx_d, charge, inter_flags.to(device)
                    )
            pred = model(Batch.from_data_list([data]))
            a = pred * a_std + a_mean
            x, v = step_torch(x, v, a, dt=DT)
            spider = spider + float(DT) * v[0]
            x = x - x[0:1]
            v_hist = torch.cat([v_hist[1:], v.unsqueeze(0)], dim=0)
            t_val += DT
            rel[k] += float((x - x_gt[k]).pow(2).mean())
            spider_abs[k] += float((spider - sp_gt[k]).pow(2).mean())
        n_used += 1

    if n_used == 0:
        return {"error": "no runs"}
    return {
        "n_runs": n_used,
        "horizon_steps": K,
        "horizon_seconds": K * DT,
        "init": f"GT frames [t0-C+1 .. t0] with t0=C={C}; no GT state after t0",
        "air_field": "ks_torch on predicted absolute positions",
        "per_step_rel_mse": (rel / n_used).tolist(),
        "per_step_spider_abs_mse": (spider_abs / n_used).tolist(),
        "mean_rel_mse": float(rel.mean() / n_used),
        "mean_spider_abs_mse": float(spider_abs.mean() / n_used),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-runs", type=int, default=8)
    args = ap.parse_args()
    cfg = load_yaml(_resolve(args.config) if not args.config.is_absolute() else args.config)
    device = _device(args.device)
    _set_seed(42)
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    data_dir = Path(dcfg["data_dir"]) if Path(dcfg["data_dir"]).is_absolute() else _resolve(dcfg["data_dir"])
    splits = load_splits(
        Path(dcfg["splits"]) if Path(dcfg["splits"]).is_absolute() else _resolve(dcfg["splits"])
    )
    store = BallooningRunStore(data_dir, cache_shards=2, mmap=True)
    ids = stratified_val_runs(store, splits["val"], n_runs=args.n_runs, seed=42)
    K = int(tcfg.get("rollout_K", 20))
    C = int(dcfg.get("C", 5))
    kw = dict(
        K=K, C=C,
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=dcfg.get("edge_sets", ["material"]),
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
    )
    out = {
        "leakage": (
            "Rollout uses GT only for initial C-frame velocity history and state at t0=C; "
            "all later x/v/spider from model+integrator; u/u_next from ks_torch."
        ),
        "run_ids": ids,
    }
    cv = ConstantVelocity(a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"]).to(device)
    fa = FollowAir(a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"]).to(device)
    out["constant_velocity"] = rollout_per_step(cv, store, ids, device, store.norm, **kw)
    out["follow_air"] = rollout_per_step(fa, store, ids, device, store.norm, **kw)
    if args.checkpoint:
        ckpt = torch.load(
            _resolve(args.checkpoint) if not Path(args.checkpoint).is_absolute() else args.checkpoint,
            map_location="cpu", weights_only=False,
        )
        sample = BallooningDataset(
            store, ids[:1], C=C, node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=dcfg.get("edge_sets"), radius=dcfg.get("radius", 0.1),
            noise_std=0.0,
        )[0]
        mcfg = dict(cfg["model"])
        name = mcfg.pop("name")
        mcfg["node_dim"] = int(sample.x.shape[-1])
        mcfg.setdefault("edge_sets", dcfg.get("edge_sets"))
        edge_dims = {}
        for s in dcfg.get("edge_sets", ["material"]):
            attr = getattr(sample, f"edge_attr_{s}", None)
            edge_dims[s] = int(attr.shape[-1]) if attr is not None and attr.ndim == 2 else 5
        mcfg["edge_dims"] = edge_dims
        model = build_model(name, **mcfg)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        out["gns"] = rollout_per_step(model, store, ids, device, store.norm, **kw)
    path = _resolve(args.out) if not Path(args.out).is_absolute() else Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
