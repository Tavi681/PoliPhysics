"""Measure graphs/s and peak GPU memory vs batch size (warm-up + timed window).

  PYTHONPATH=. python -m ml.scripts.batch_sweep \
    --config ml/configs/gns_local_gpu.yaml --device cuda \
    --batches 16,64,128,256 --seconds 120 --warmup 30
"""
from __future__ import annotations

import argparse
import json
import time
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


def _run_one(cfg: dict, device: torch.device, batch_size: int, warmup: int, seconds: float) -> dict:
    seed = int(cfg.get("train", {}).get("seed", 42))
    _set_seed(seed)
    dcfg = cfg["data"]
    tcfg = dict(cfg["train"])
    tcfg["batch_size"] = int(batch_size)
    data_dir = Path(dcfg["data_dir"]) if Path(dcfg["data_dir"]).is_absolute() else _resolve(dcfg["data_dir"])
    splits_path = Path(dcfg["splits"]) if Path(dcfg["splits"]).is_absolute() else _resolve(dcfg["splits"])
    splits = load_splits(splits_path)
    store = BallooningRunStore(
        data_dir,
        cache_shards=int(dcfg.get("cache_shards", 4)),
        mmap=bool(dcfg.get("mmap", True)),
    )
    edge_sets = dcfg.get("edge_sets", ["material"])
    noise_stats = load_noise_stats(data_dir / "noise_stats.json")
    scalar_noise, per_type = resolve_noise(
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
        noise_std=scalar_noise,
        noise_std_by_type=per_type,
        normalize_features=dcfg.get("normalize_features", True),
        sample_weight=dcfg.get("sample_weight", "per_run"),
        rotate_augment=False,
        seed=seed,
    )
    loader = make_loader(
        ds,
        batch_size=batch_size,
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
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    it = iter(loader)
    # warm-up
    for _ in range(warmup):
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_steps = 0
    n_graphs = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
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
        n_steps += 1
        n_graphs += int(batch.num_graphs)
    _sync(device)
    elapsed = time.perf_counter() - t0
    peak_mib = 0.0
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    total_mib = 24 * 1024
    if device.type == "cuda":
        total_mib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    return {
        "model": name,
        "batch_size": batch_size,
        "steps": n_steps,
        "graphs": n_graphs,
        "elapsed_s": elapsed,
        "steps_per_s": n_steps / max(elapsed, 1e-9),
        "graphs_per_s": n_graphs / max(elapsed, 1e-9),
        "peak_gpu_mib": peak_mib,
        "peak_gpu_frac": peak_mib / max(total_mib, 1.0),
        "train_frames": len(ds),
        "epoch_hours": (len(ds) / max(n_graphs / max(elapsed, 1e-9), 1e-9)) / 3600.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batches", type=str, default="16,64,128,256")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(_resolve(args.config) if not Path(args.config).is_absolute() else args.config)
    device = _device(args.device)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows = []
    for bs in batches:
        print(f"=== batch={bs} warmup={args.warmup} timed={args.seconds}s ===", flush=True)
        try:
            row = _run_one(cfg, device, bs, args.warmup, args.seconds)
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
        except RuntimeError as e:
            rows.append({"batch_size": bs, "error": str(e)})
            print(f"FAILED batch={bs}: {e}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Pick saturating batch under 80% VRAM
    ok = [r for r in rows if "graphs_per_s" in r and r.get("peak_gpu_frac", 1) < 0.80]
    if ok:
        best = max(ok, key=lambda r: r["graphs_per_s"])
        # Prefer smallest batch within 5% of peak graphs/s (saturation knee)
        peak = best["graphs_per_s"]
        knee = min(
            (r for r in ok if r["graphs_per_s"] >= 0.95 * peak),
            key=lambda r: r["batch_size"],
        )
        print(
            f"PICK batch_size={knee['batch_size']} graphs/s={knee['graphs_per_s']:.1f} "
            f"peak_mib={knee['peak_gpu_mib']:.0f} ({100 * knee['peak_gpu_frac']:.1f}%) "
            f"epoch_h={knee['epoch_hours']:.2f}",
            flush=True,
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
