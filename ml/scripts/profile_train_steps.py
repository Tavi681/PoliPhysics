"""Profile per-step train timings for gns_local (warm-up + N steps).

  PYTHONPATH=. python -m ml.scripts.profile_train_steps \\
    --config ml/configs/gns_local_gpu.yaml --device cuda \\
    --warmup 20 --steps 100
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ml.data.dataset import BallooningDataset, BallooningRunStore, make_loader
from ml.data.noise_stats import load_noise_stats, resolve_noise
from ml.data.splits import load_splits
from ml.models import build_model
from ml.train import _device, _resolve, _set_seed, pin_memory_ok
from ml.utils.config import load_yaml


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(_resolve(args.config) if not Path(args.config).is_absolute() else args.config)
    device = _device(args.device)
    seed = int(cfg.get("train", {}).get("seed", 42))
    _set_seed(seed)

    dcfg = cfg["data"]
    tcfg = cfg["train"]
    data_dir = _resolve(dcfg["data_dir"]) if not Path(dcfg["data_dir"]).is_absolute() else Path(dcfg["data_dir"])
    splits = load_splits(
        _resolve(dcfg["splits"]) if not Path(dcfg["splits"]).is_absolute() else Path(dcfg["splits"])
    )
    store = BallooningRunStore(
        data_dir,
        cache_shards=int(dcfg.get("cache_shards", 8)),
        mmap=bool(dcfg.get("mmap", True)),
    )
    edge_sets = dcfg.get("edge_sets", ["material"])
    noise_stats = load_noise_stats(data_dir / "noise_stats.json")
    scalar_noise, per_type, comp_noise = resolve_noise(
        noise_k=dcfg.get("noise_k"),
        noise_std=dcfg.get("noise_std"),
        noise_stats=noise_stats,
        a_std=store.norm.get("a", {}).get("std"),
        dt=float(store.norm.get("dt", 0.01)),
    )
    ds = BallooningDataset(
        store, splits["train"],
        C=dcfg.get("C", 5),
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets,
        radius=dcfg.get("radius", 0.1),
        noise_std=scalar_noise if comp_noise is None and per_type is None else 0.0,
        noise_std_by_type=per_type,
        noise_std_components=comp_noise,
        normalize_features=dcfg.get("normalize_features", True),
        sample_weight=dcfg.get("sample_weight", "per_run"),
        rotate_augment=False,
        seed=seed,
    )
    loader = make_loader(
        ds,
        batch_size=tcfg.get("batch_size", 8),
        shuffle=True,
        num_workers=tcfg.get("num_workers", 0),
        device=str(device),
        prefetch_factor=tcfg.get("prefetch_factor"),
        shard_aware=bool(dcfg.get("shard_aware", True)),
        seed=seed,
    )
    sample = ds[0]
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    mcfg["node_dim"] = int(sample.x.shape[-1])
    mcfg.setdefault("edge_sets", edge_sets)
    edge_dims = {}
    for s in edge_sets:
        attr = getattr(sample, f"edge_attr_{s}", None)
        edge_dims[s] = int(attr.shape[-1]) if attr is not None and attr.ndim == 2 else (5 if s == "material" else 7)
    mcfg["edge_dims"] = edge_dims
    model = build_model(name, **mcfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg.get("lr", 1e-4)))

    print(
        f"profile: cuda_available={torch.cuda.is_available()} device={device} "
        f"model={next(model.parameters()).device} n_train={len(ds)} "
        f"batch={tcfg.get('batch_size')} workers={tcfg.get('num_workers')}"
    )

    it = iter(loader)
    times = defaultdict(list)
    n_graphs = 0
    # warm-up
    for _ in range(args.warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(device, non_blocking=pin_memory_ok(device))
        opt.zero_grad(set_to_none=True)
        loss = (model(batch) - batch.y).pow(2).mean()
        loss.backward()
        opt.step()
    _sync(device)

    t_wall0 = time.perf_counter()
    for i in range(args.steps):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        t1 = time.perf_counter()
        # assembly already done in workers; attribute fetch cost is in data_wait
        t_h2d0 = time.perf_counter()
        batch = batch.to(device, non_blocking=pin_memory_ok(device))
        _sync(device)
        t_h2d1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        t_f0 = time.perf_counter()
        pred = model(batch)
        loss = (pred - batch.y).pow(2).mean()
        _sync(device)
        t_f1 = time.perf_counter()
        loss.backward()
        _sync(device)
        t_b1 = time.perf_counter()
        opt.step()
        _sync(device)
        t_o1 = time.perf_counter()

        times["data_wait_s"].append(t1 - t0)
        times["h2d_s"].append(t_h2d1 - t_h2d0)
        times["forward_s"].append(t_f1 - t_f0)
        times["backward_s"].append(t_b1 - t_f1)
        times["optim_s"].append(t_o1 - t_b1)
        times["step_s"].append(t_o1 - t0)
        n_graphs += int(getattr(batch, "num_graphs", tcfg.get("batch_size", 8)))
        if i == 0:
            print(
                f"first_step_devices model={next(model.parameters()).device} "
                f"x={batch.x.device} y={batch.y.device}"
            )
            if device.type == "cuda":
                import subprocess
                print(subprocess.check_output(["nvidia-smi"], text=True))

    wall = time.perf_counter() - t_wall0
    summary = {
        "warmup": args.warmup,
        "steps": args.steps,
        "wall_s": wall,
        "steps_per_s": args.steps / max(wall, 1e-9),
        "graphs_per_s": n_graphs / max(wall, 1e-9),
        "mean": {k: float(np.mean(v)) for k, v in times.items()},
        "p50": {k: float(np.median(v)) for k, v in times.items()},
        "device": str(device),
        "batch_size": tcfg.get("batch_size"),
        "num_workers": tcfg.get("num_workers"),
        "cache_shards": dcfg.get("cache_shards"),
        "shard_aware": dcfg.get("shard_aware", True),
        "note": (
            "data_wait_s includes worker fetch+assemble+graph+collate "
            "(not separable once num_workers>0); h2d/forward/backward/optim are on-device."
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
