"""Failure diagnostics for selected gns_baseline checkpoints.

Rolls stratified val runs with full trajectories, tracks segment-length drift
by node-adjacency class, first edge leaving [0.9, 1.1] * l0, spider altitude
(surrogate vs simulator), and one-step acceleration error just before failure.

Example (CPU VM, full data):

  PYTHONPATH=. python -m ml.scripts.diagnose_baseline_failure \\
    --config ml/configs/gns_baseline_gpu.yaml \\
    --checkpoint "Tema 1/results/ml/runs/gns_baseline_s42_5716ee5b_1791828699/epoch_04.pt" \\
    --device cpu --n-runs 20 --t-end 60

Local smoke (local_subset):

  PYTHONPATH=. python -m ml.scripts.diagnose_baseline_failure \\
    --config ml/configs/gns_baseline.yaml \\
    --checkpoint "Tema 1/results/ml/runs/gns_baseline_s42_409336f7_1791728467/best.pt" \\
    --device cpu --n-runs 2 --t-end 5
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

from ml.data.dataset import BallooningDataset, BallooningRunStore
from ml.data.splits import load_splits
from ml.evaluate import full_rollout_surrogate
from ml.metrics.rollout_metrics import PHYS_EDGE_TOL, segment_rel_deviations, time_to_failure
from ml.models import build_model
from ml.train import _device, _resolve, _set_seed
from ml.utils.config import load_yaml
from ml.validation import stratified_val_runs

EDGE_BUCKETS = ("spider_adj", "interior", "tip")
NODE_TYPES = ("spider", "spider_adj", "interior", "tip")


def _load_model(cfg, store, ckpt_path: Path, device, run_ids):
    dcfg = cfg["data"]
    edge_sets = list(dcfg.get("edge_sets", ["material"]))
    sample = BallooningDataset(
        store,
        run_ids[:1],
        C=dcfg.get("C", 5),
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets,
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
        noise_std=0.0,
    )[0]
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
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


def _spider_adjacent(edges: np.ndarray) -> set[int]:
    adj: set[int] = set()
    for i, j in edges:
        if int(i) == 0:
            adj.add(int(j))
        if int(j) == 0:
            adj.add(int(i))
    return adj


def _node_role(node_id: int, node_type: np.ndarray, adj: set[int]) -> str:
    if node_id == 0:
        return "spider"
    if node_id in adj:
        return "spider_adj"
    if int(node_type[node_id]) == 2:
        return "tip"
    return "interior"


def _edge_bucket(roles: list[str], i: int, j: int) -> str:
    if i == 0 or j == 0:
        return "spider_adj"
    ri, rj = roles[i], roles[j]
    if "tip" in (ri, rj):
        return "tip"
    if "spider_adj" in (ri, rj):
        return "spider_adj"
    return "interior"


def _edge_bucket_masks(edges: np.ndarray, node_type: np.ndarray) -> dict[str, np.ndarray]:
    adj = _spider_adjacent(edges)
    n = int(node_type.shape[0])
    roles = [_node_role(k, node_type, adj) for k in range(n)]
    masks = {b: np.zeros(len(edges), dtype=bool) for b in EDGE_BUCKETS}
    for e, (i, j) in enumerate(edges):
        masks[_edge_bucket(roles, int(i), int(j))][e] = True
    return masks


def _series_by_bucket(X: np.ndarray, edges: np.ndarray, l0: float, bucket_masks: dict[str, np.ndarray]):
    """Mean max relative deviation per bucket at each trajectory frame."""
    T = X.shape[0]
    out: dict[str, np.ndarray] = {}
    for b in EDGE_BUCKETS:
        m = bucket_masks[b]
        series = np.full(T, np.nan, dtype=np.float64)
        if not bool(m.any()):
            out[b] = series
            continue
        e_sub = edges[m]
        for k in range(T):
            if not np.isfinite(X[k]).all():
                series[k] = float("inf")
                continue
            devs = segment_rel_deviations(X[k], e_sub, l0)
            series[k] = float(np.max(devs)) if devs.size else 0.0
        out[b] = series
    return out


def _first_fail_edge(
    X: np.ndarray,
    edges: np.ndarray,
    l0: float,
    t_rel: np.ndarray,
) -> dict:
    ei, ej = edges[:, 0], edges[:, 1]
    for k in range(X.shape[0]):
        if not np.isfinite(X[k]).all():
            return {
                "t_rel_s": float(t_rel[k]) if k < len(t_rel) else float("nan"),
                "frame": k,
                "edge_i": None,
                "edge_j": None,
                "rel_dev": float("inf"),
                "reason": "non_finite",
            }
        devs = segment_rel_deviations(X[k], edges, l0)
        if devs.size == 0:
            continue
        worst = int(np.argmax(devs))
        if float(devs[worst]) >= PHYS_EDGE_TOL:
            return {
                "t_rel_s": float(t_rel[k]),
                "frame": k,
                "edge_i": int(ei[worst]),
                "edge_j": int(ej[worst]),
                "rel_dev": float(devs[worst]),
                "reason": "edge_tol",
            }
    return {
        "t_rel_s": float("inf"),
        "frame": None,
        "edge_i": None,
        "edge_j": None,
        "rel_dev": float("nan"),
        "reason": "never",
    }


def _gt_spider_z_series(rec: dict, C: int) -> tuple[np.ndarray, np.ndarray]:
    t = rec["t"].astype(np.float64)
    t_rel = t - float(t[C])
    z = rec["spider_abs"][:, 2].astype(np.float64)
    return t_rel, z


def _nearest_gt_index(t_arr: np.ndarray, t_query: float) -> int:
    return int(np.argmin(np.abs(t_arr - t_query)))


@torch.no_grad()
def _one_step_at_t(
    model,
    dataset: BallooningDataset,
    rid: str,
    t_frame: int,
    device,
    norm: dict,
) -> dict[str, float]:
    idx = None
    for i, (r, t) in enumerate(dataset.index):
        if r == rid and t == t_frame:
            idx = i
            break
    if idx is None:
        return {k: float("nan") for k in ("overall", "spider", "interior", "tip", "spider_adj")}

    data = dataset[idx].to(device)
    pred = model(Batch.from_data_list([data]))
    a_mean = torch.as_tensor(norm["a"]["mean"], device=device, dtype=pred.dtype)
    a_std = torch.as_tensor(norm["a"]["std"], device=device, dtype=pred.dtype)
    pred_a = pred * a_std + a_mean
    tgt = data.y * a_std + a_mean if dataset.normalize_target else data.y
    err = (pred_a - tgt).pow(2).mean(dim=-1)
    nt = data.node_type
    rec = dataset.store.get(rid)
    adj = _spider_adjacent(np.asarray(rec["edges"], dtype=np.int64))
    nt_np = rec["node_type"]
    out: dict[str, float] = {"overall": float(err.mean())}
    for typ, name in enumerate(("spider", "interior", "tip")):
        m = nt == typ
        out[name] = float(err[m].mean()) if bool(m.any()) else float("nan")
    m_adj = torch.tensor(
        [_node_role(i, nt_np, adj) == "spider_adj" for i in range(nt.numel())],
        device=device,
        dtype=torch.bool,
    )
    out["spider_adj"] = float(err[m_adj].mean()) if bool(m_adj.any()) else float("nan")
    return out


def _analyze_run(
    model,
    store,
    rid: str,
    cfg: dict,
    device,
    *,
    t_end: float,
    dataset: BallooningDataset,
) -> dict:
    dcfg = cfg["data"]
    C = int(dcfg.get("C", 5))
    rec = store.get(rid)
    g = rec["globals"]
    pred = full_rollout_surrogate(
        model,
        rec,
        store.norm,
        device,
        C=C,
        t_end=t_end,
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=dcfg.get("edge_sets"),
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
        return_trajectory=True,
    )
    edges = np.asarray(rec["edges"], dtype=np.int64)
    l0 = float(g["L"]) / max(int(g["N_t"]), 1)
    nt = rec["node_type"].astype(np.int64)
    bucket_masks = _edge_bucket_masks(edges, nt)

    X = pred["x"]
    tt = np.asarray(pred["t"], dtype=np.float64)
    t_release = float(rec["t"][C])
    t_rel = tt - t_release

    bucket_series = _series_by_bucket(X, edges, l0, bucket_masks)
    max_edge_rel = np.empty(len(X), dtype=np.float64)
    finite = np.empty(len(X), dtype=bool)
    for k in range(len(X)):
        finite[k] = bool(np.isfinite(X[k]).all())
        if not finite[k]:
            max_edge_rel[k] = float("inf")
        else:
            devs = segment_rel_deviations(X[k], edges, l0)
            max_edge_rel[k] = float(np.max(devs)) if devs.size else 0.0
    ttf = time_to_failure(t_rel, max_edge_rel, finite)
    first_fail = _first_fail_edge(X, edges, l0, t_rel)

    gt_t_rel, gt_z = _gt_spider_z_series(rec, C)
    surr_z = X[:, 0, 2]

    pre_t = float("nan")
    pre_frame = None
    one_step: dict[str, float] = {}
    if first_fail["frame"] is not None and first_fail["frame"] > 0:
        pre_frame = int(first_fail["frame"]) - 1
        pre_t = float(t_rel[pre_frame])
        t_gt = rec["t"].astype(np.float64)
        t_query = t_release + pre_t
        t_frame = _nearest_gt_index(t_gt, t_query)
        t_frame = max(C, min(t_frame, len(t_gt) - 2))
        one_step = _one_step_at_t(model, dataset, rid, t_frame, device, store.norm)
    elif first_fail["reason"] == "never" and len(t_rel) > 1:
        pre_frame = len(t_rel) - 2
        pre_t = float(t_rel[pre_frame])
        t_gt = rec["t"].astype(np.float64)
        t_frame = _nearest_gt_index(t_gt, t_release + pre_t)
        t_frame = max(C, min(t_frame, len(t_gt) - 2))
        one_step = _one_step_at_t(model, dataset, rid, t_frame, device, store.norm)

    return {
        "run_id": rid,
        "N": int(g["N"]),
        "q": float(g["q"]),
        "sigma_w": float(g["sigma_w"]),
        "outcome_sim": str(g.get("outcome", "")),
        "status_surr": str(pred.get("status", "")),
        "exit_time_surr": float(pred.get("exit_time", float("nan"))),
        "ttf_s": ttf,
        "first_fail_edge": first_fail,
        "pre_fail_t_rel_s": pre_t,
        "pre_fail_frame": pre_frame,
        "one_step_pre_fail_mse": one_step,
        "t_rel_s": t_rel.tolist(),
        "edge_dev_spider_adj": bucket_series["spider_adj"].tolist(),
        "edge_dev_interior": bucket_series["interior"].tolist(),
        "edge_dev_tip": bucket_series["tip"].tolist(),
        "spider_z_surr": surr_z.tolist(),
        "spider_z_gt_t_rel": gt_t_rel.tolist(),
        "spider_z_gt": gt_z.tolist(),
    }


def _plot_edge_dev_by_n(runs: list[dict], out_dir: Path) -> list[str]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for r in runs:
        by_n[int(r["N"])].append(r)
    paths: list[str] = []
    for N, rs in sorted(by_n.items()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for bucket, key, color in (
            ("spider_adj", "edge_dev_spider_adj", "C0"),
            ("interior", "edge_dev_interior", "C1"),
            ("tip", "edge_dev_tip", "C2"),
        ):
            grid: dict[float, list[float]] = defaultdict(list)
            for r in rs:
                for t, v in zip(r["t_rel_s"], r[key]):
                    if np.isfinite(v):
                        grid[round(float(t), 3)].append(float(v))
            if not grid:
                continue
            xs = sorted(grid.keys())
            ys = [float(np.mean(grid[t])) for t in xs]
            ax.plot(xs, ys, label=bucket, color=color)
        ax.axhline(PHYS_EDGE_TOL, color="k", ls="--", lw=0.8, label=f"tol={PHYS_EDGE_TOL}")
        ax.set_xlabel("time since release (s)")
        ax.set_ylabel("mean max |Δℓ|/ℓ₀ in bucket")
        ax.set_title(f"Edge length drift (N={N}, n={len(rs)} runs)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"edge_dev_vs_time_N{N}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _plot_spider_z(runs: list[dict], out_dir: Path, max_panels: int = 6) -> list[str]:
    paths: list[str] = []
    for r in runs[:max_panels]:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(r["t_rel_s"], r["spider_z_surr"], label="surrogate", color="C3")
        ax.plot(r["spider_z_gt_t_rel"], r["spider_z_gt"], label="simulator", color="C0", alpha=0.8)
        ax.set_xlabel("time since release (s)")
        ax.set_ylabel("spider z (m)")
        ax.set_title(
            f"{r['run_id']} N={r['N']} sim={r['outcome_sim']} surr={r['status_surr']}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = r["run_id"].replace("/", "_")
        p = out_dir / f"spider_z_{safe}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _plot_onestep_prefail(runs: list[dict], out_dir: Path) -> str | None:
    buckets = ("spider", "spider_adj", "interior", "tip")
    sums = {b: 0.0 for b in buckets}
    counts = {b: 0 for b in buckets}
    for r in runs:
        m = r.get("one_step_pre_fail_mse") or {}
        for b in buckets:
            v = m.get(b, float("nan"))
            if np.isfinite(v):
                sums[b] += v
                counts[b] += 1
    if not any(counts.values()):
        return None
    means = [sums[b] / max(counts[b], 1) for b in buckets]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(buckets, means, color=["C0", "C1", "C2", "C3"])
    ax.set_ylabel("one-step accel MSE (m²/s⁴)")
    ax.set_title("One-step error at frame before first edge failure")
    fig.tight_layout()
    p = out_dir / "one_step_pre_fail_by_type.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def _markdown_summary(summary: dict) -> str:
    lines = [
        "# gns_baseline failure diagnostics",
        "",
        f"- checkpoint: `{summary.get('checkpoint', '')}`",
        f"- n_runs: {summary.get('n_runs', 0)}",
        f"- physical_edge_tol: {summary.get('physical_edge_tol', PHYS_EDGE_TOL)}",
        "",
        "## Outcomes (simulator vs surrogate)",
        "",
        "| run_id | N | sim | surr | ttf_s | first_fail_t | edge (i,j) |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for r in summary.get("runs", []):
        ff = r.get("first_fail_edge") or {}
        lines.append(
            f"| {r['run_id']} | {r['N']} | {r.get('outcome_sim','')} | "
            f"{r.get('status_surr','')} | {r.get('ttf_s', float('nan')):.3g} | "
            f"{ff.get('t_rel_s', float('nan')):.3g} | ({ff.get('edge_i')},{ff.get('edge_j')}) |"
        )
    lines.extend(["", "## Plots", ""])
    for p in summary.get("plot_paths", []):
        lines.append(f"- `{p}`")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "Tema 1/results/ml/runs/gns_baseline_s42_5716ee5b_1791828699/epoch_04.pt"
        ),
    )
    ap.add_argument("--data-dir", type=Path, default=None, help="override cfg data.data_dir")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-end", type=float, default=60.0)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Tema 1/results/ml/diagnostics"),
    )
    args = ap.parse_args()

    cfg_path = _resolve(args.config) if not args.config.is_absolute() else args.config
    cfg = load_yaml(cfg_path)
    if args.data_dir is not None:
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["data_dir"] = str(args.data_dir)

    device = _device(args.device)
    _set_seed(args.seed)
    dcfg = cfg["data"]
    data_dir = Path(dcfg["data_dir"]) if Path(dcfg["data_dir"]).is_absolute() else _resolve(dcfg["data_dir"])
    splits_path = Path(dcfg["splits"]) if Path(dcfg["splits"]).is_absolute() else _resolve(dcfg["splits"])
    splits = load_splits(splits_path)
    store = BallooningRunStore(
        data_dir,
        cache_shards=int(dcfg.get("cache_shards", 2)),
        mmap=bool(dcfg.get("mmap", True)),
    )
    run_ids = stratified_val_runs(store, splits["val"], n_runs=args.n_runs, seed=args.seed)
    if not run_ids:
        raise SystemExit("no val runs found in store for stratified sample")

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else _resolve(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    out_dir = args.out_dir if args.out_dir.is_absolute() else _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(cfg, store, ckpt, device, run_ids)
    dataset = BallooningDataset(
        store,
        run_ids,
        C=int(dcfg.get("C", 5)),
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=dcfg.get("edge_sets"),
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
        noise_std=0.0,
    )

    t0 = time.perf_counter()
    per_run: list[dict] = []
    for rid in run_ids:
        print(f"diagnose {rid}", flush=True)
        per_run.append(
            _analyze_run(model, store, rid, cfg, device, t_end=args.t_end, dataset=dataset)
        )

    plot_paths: list[str] = []
    plot_paths.extend(_plot_edge_dev_by_n(per_run, out_dir))
    plot_paths.extend(_plot_spider_z(per_run, out_dir))
    p_os = _plot_onestep_prefail(per_run, out_dir)
    if p_os:
        plot_paths.append(p_os)

    n_match = sum(1 for r in per_run if r.get("outcome_sim") == r.get("status_surr"))
    n_fail = sum(1 for r in per_run if np.isfinite(float(r.get("ttf_s", float("nan")))))
    summary = {
        "checkpoint": str(ckpt),
        "config": str(cfg_path),
        "data_dir": str(data_dir),
        "n_runs": len(per_run),
        "run_ids": run_ids,
        "physical_edge_tol": PHYS_EDGE_TOL,
        "t_end_s": args.t_end,
        "wall_s": time.perf_counter() - t0,
        "outcome_match_rate": n_match / max(len(per_run), 1),
        "n_edge_failures": n_fail,
        "plot_paths": plot_paths,
        "runs": per_run,
    }
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    # Trim long series in JSON for readability (keep in per-run only if needed — keep full for now)
    summary_json.write_text(json.dumps(summary, indent=2))
    summary_md.write_text(_markdown_summary(summary))
    print(f"wrote {summary_json}")
    print(f"wrote {summary_md}")


if __name__ == "__main__":
    main()
