"""Train a registered surrogate: python -m ml.train --config configs/stub_gnn.yaml"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    SequentialLR,
)

from ml.data.dataset import BallooningDataset, BallooningRunStore, make_loader
from ml.data.integrator import step_torch, DT
from ml.data.normalize import denormalize
from ml.data.splits import load_splits, splits_hash
from ml.models import build_model
from ml.utils.config import load_yaml
from ml.utils.git_info import git_provenance, repo_root, format_git_tag


def _device(name: str | None) -> torch.device:
    if name:
        requested = str(name).lower()
        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"device={name!r} requested but torch.cuda.is_available() is False"
                )
            return torch.device(name)
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pin_memory_ok(device: torch.device) -> bool:
    return device.type == "cuda"


def _log_first_step_devices(model, batch, device: torch.device) -> None:
    pdev = next(model.parameters()).device
    print(
        f"device_check: cuda_available={torch.cuda.is_available()} "
        f"requested={device} model={pdev}"
    )
    for name in ("x", "y", "pos", "edge_index"):
        t = getattr(batch, name, None)
        if t is not None and torch.is_tensor(t):
            print(f"  batch.{name}.device={t.device} shape={tuple(t.shape)}")
    if device.type == "cuda" and pdev.type != "cuda":
        raise RuntimeError(f"model on {pdev} but cuda was requested")
    if device.type == "cuda" and hasattr(batch, "x") and batch.x.device.type != "cuda":
        raise RuntimeError(f"batch.x on {batch.x.device} after .to(cuda)")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def _init_wandb(cfg: dict, run_meta: dict):
    wcfg = cfg.get("wandb", {})
    mode = wcfg.get("mode", "online")
    try:
        import wandb
        if mode == "disabled":
            return None
        # offline fallback if no network
        try:
            run = wandb.init(
                project=wcfg.get("project", "poliphysics-ballooning-ml"),
                entity=wcfg.get("entity"),
                config={**cfg, **run_meta},
                mode=mode,
            )
            return run
        except Exception:
            run = wandb.init(
                project=wcfg.get("project", "poliphysics-ballooning-ml"),
                config={**cfg, **run_meta},
                mode="offline",
            )
            return run
    except Exception as e:
        print(f"W&B unavailable ({e}); continuing without logging")
        return None


@torch.no_grad()
def one_step_mse(model, loader, device, norm) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        err = (pred - batch.y).pow(2).mean()
        total += float(err) * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


@torch.no_grad()
def rollout_mse(
    model, store, run_ids, device, norm, K: int = 100, C: int = 5, max_runs: int = 8,
    *,
    node_features=None, globals_fields=None, edge_sets=None, radius: float = 0.1,
    normalize_features: bool = True,
) -> float:
    """Mean spider-relative position MSE (m²) over K one-step-integrated frames.

    For each of up to ``max_runs`` runs: start at frame ``C``, predict acceleration
    and integrate with the ecosystem integrator for ``K`` steps (horizon ``K·dt``
    with ``dt=0.01 s``). At each step, append
    ``mean_nodes((x_pred - x_gt_rel)²)``; return the mean of those errors across
    steps and runs. Units: m² (relative positions). Not a single end-of-horizon
    error — it averages the per-step position MSE along the short open-loop
    trajectory (teacher-forced graph features from the dataset at each t, with
    predicted ``pos`` / velocity state carried forward).
    """
    model.eval()
    a_mean = torch.as_tensor(norm["a"]["mean"], device=device)
    a_std = torch.as_tensor(norm["a"]["std"], device=device)
    edge_sets = list(edge_sets or model.required_edges() or ["material"])
    errs = []
    for rid in list(run_ids)[:max_runs]:
        rec = store.get(rid)
        T = rec["v"].shape[0]
        t0 = C
        if t0 + K >= T:
            continue
        x = torch.from_numpy(rec["x_rel"][t0].astype(np.float32)).to(device)
        v = torch.from_numpy(rec["v"][t0].astype(np.float32)).to(device)
        from ml.data.dataset import BallooningDataset
        ds = BallooningDataset(
            store, [rid], C=C,
            node_features=node_features,
            globals_fields=globals_fields,
            edge_sets=edge_sets,
            radius=radius,
            normalize_features=normalize_features,
            normalize_target=True, noise_std=0.0,
        )
        idx_map = {t: i for i, (_, t) in enumerate(ds.index)}
        for k in range(K):
            t = t0 + k
            if t not in idx_map:
                break
            data = ds[idx_map[t]].to(device)
            data.pos = x
            data.v_t = v
            pred_n = model(data)
            a = pred_n * a_std + a_mean
            x, v = step_torch(x, v, a, dt=DT)
            x_gt = torch.from_numpy(rec["x_rel"][t0 + k + 1].astype(np.float32)).to(device)
            errs.append(float((x - x_gt).pow(2).mean()))
    return float(np.mean(errs)) if errs else float("nan")


def train_one(cfg: dict, device: torch.device, seed: int) -> Path:
    _set_seed(seed)
    data_dir = _resolve(cfg["data"]["data_dir"])
    splits = load_splits(_resolve(cfg["data"]["splits"]))
    dcfg = cfg["data"]
    store = BallooningRunStore(
        data_dir,
        cache_shards=int(dcfg.get("cache_shards", 4)),
        mmap=bool(dcfg.get("mmap", True)),
    )
    edge_sets = dcfg.get("edge_sets", ["material"])

    ds_kwargs = dict(
        C=dcfg.get("C", 5),
        history_u=dcfg.get("history_u", False),
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets,
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
        sample_weight=dcfg.get("sample_weight", "per_run"),
        rotate_augment=bool(dcfg.get("rotate_augment", False)),
        seed=seed,
    )
    from ml.data.noise_stats import load_noise_stats, resolve_noise

    noise_stats = load_noise_stats(data_dir / "noise_stats.json")
    noise_k = dcfg.get("noise_k", None)
    noise_std_cfg = dcfg.get("noise_std", None)
    # Explicit noise_std without noise_k keeps legacy behaviour; noise_k wins when set.
    scalar_noise, per_type_noise = resolve_noise(
        noise_k=float(noise_k) if noise_k is not None else None,
        noise_std=float(noise_std_cfg) if noise_std_cfg is not None else None,
        noise_stats=noise_stats,
        a_std=store.norm.get("a", {}).get("std"),
        dt=float(store.norm.get("dt", DT)),
    )
    train_ds = BallooningDataset(
        store, splits["train"],
        noise_std=scalar_noise,
        noise_std_by_type=per_type_noise,
        **ds_kwargs,
    )
    val_ds = BallooningDataset(
        store, splits["val"],
        noise_std=0.0, sample_weight="per_frame",
        C=ds_kwargs["C"],
        history_u=ds_kwargs["history_u"],
        node_features=ds_kwargs["node_features"],
        globals_fields=ds_kwargs["globals_fields"],
        edge_sets=edge_sets,
        radius=ds_kwargs["radius"],
        normalize_features=ds_kwargs["normalize_features"],
        rotate_augment=False,
        seed=seed,
    )
    tcfg = cfg["train"]
    train_loader = make_loader(
        train_ds, batch_size=tcfg.get("batch_size", 4), shuffle=True,
        num_workers=tcfg.get("num_workers", 0), device=str(device),
        prefetch_factor=tcfg.get("prefetch_factor"),
        shard_aware=bool(dcfg.get("shard_aware", True)),
        seed=seed,
    )
    val_loader = make_loader(
        val_ds, batch_size=tcfg.get("batch_size", 4), shuffle=False,
        num_workers=0, device=str(device),
        shard_aware=False,
        seed=seed,
    )

    # infer node_dim / edge_dims from a sample
    sample = train_ds[0]
    node_dim = int(sample.x.shape[-1])
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    if name in ("constant_velocity", "follow_air"):
        mcfg["a_mean"] = store.norm["a"]["mean"]
        mcfg["a_std"] = store.norm["a"]["std"]
    else:
        mcfg["node_dim"] = node_dim
        mcfg.setdefault("edge_sets", edge_sets)
        edge_dims = dict(mcfg.get("edge_dims") or {})
        for s in edge_sets:
            attr = getattr(sample, f"edge_attr_{s}", None)
            if attr is not None and attr.ndim == 2 and attr.shape[-1] > 0:
                edge_dims[s] = int(attr.shape[-1])
            else:
                edge_dims.setdefault(s, 5 if s == "material" else 7)
        mcfg["edge_dims"] = edge_dims
    model = build_model(name, **mcfg).to(device)
    # ensure required edges present in dataset
    need = model.required_edges()
    if need and set(need) - set(edge_sets):
        raise RuntimeError(f"model needs edges {need}, dataset has {edge_sets}")

    prov = git_provenance()
    run_id = f"{name}_s{seed}_{format_git_tag(prov)[:8]}_{int(time.time())}"
    ckpt_dir = _resolve(tcfg.get("checkpoint_dir", "Tema 1/results/ml/runs")) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run_id": run_id,
        "model": name,
        "param_count": model.count_parameters(),
        "data_version": store.manifest.get("data_version"),
        "splits_hash": splits_hash(splits),
        "git_commit": prov["commit"],
        "git_dirty": prov["dirty"],
        "device": str(device),
        "seed": seed,
        "noise_k": noise_k,
        "noise_std": scalar_noise,
        "noise_std_by_type": None if per_type_noise is None else per_type_noise.tolist(),
        "noise_stats_overall_rms": None if noise_stats is None else noise_stats.get("overall_rms"),
    }
    with open(ckpt_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    wb = _init_wandb(cfg, run_meta)
    max_steps = int(tcfg.get("max_steps", 1000))
    if max_steps <= 0 or name in ("constant_velocity", "follow_air"):
        # save baseline checkpoint and exit
        torch.save({"model": model.state_dict(), "meta": run_meta, "cfg": cfg}, ckpt_dir / "best.pt")
        torch.save({"model": model.state_dict(), "meta": run_meta, "cfg": cfg}, ckpt_dir / "last.pt")
        print(f"baseline checkpoint → {ckpt_dir}")
        return ckpt_dir

    opt_name = str(tcfg.get("optimizer", "adamw")).lower()
    lr = float(tcfg.get("lr", 1e-3))
    wd = float(tcfg.get("weight_decay", 1e-4))
    if opt_name == "adam":
        opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched_cfg = tcfg.get("lr_schedule") or {}
    sched_type = sched_cfg.get("type", "cosine")
    warmup = int(sched_cfg.get("warmup_steps", 0))
    if sched_type == "cosine":
        sched = SequentialLR(
            opt,
            [
                LinearLR(opt, start_factor=0.1, total_iters=max(warmup, 1)),
                CosineAnnealingLR(opt, T_max=max(max_steps - warmup, 1)),
            ],
            milestones=[max(warmup, 1)],
        )
    elif sched_type == "exponential":
        # lr_end = lr * gamma^max_steps  (Sanchez-Gonzalez-style decay)
        lr_end = float(sched_cfg.get("lr_end", 1e-6))
        gamma = (lr_end / max(lr, 1e-12)) ** (1.0 / max(max_steps, 1))
        sched = ExponentialLR(opt, gamma=gamma)
    else:
        sched = None

    best = float("inf")
    patience = int(tcfg.get("early_stopping", {}).get("patience", 20))
    bad = 0
    step = 0
    epoch = 0
    t0 = time.time()
    model.train()
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
    it = iter(train_loader)
    log_every = int(tcfg.get("log_every", 10))
    logged_devices = False
    while step < max_steps:
        t_wait0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            it = iter(train_loader)
            batch = next(it)
        data_wait_s = time.perf_counter() - t_wait0
        batch = batch.to(device, non_blocking=pin_memory_ok(device))
        if not logged_devices:
            _log_first_step_devices(model, batch, device)
            logged_devices = True
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = (pred - batch.y).pow(2).mean()
        loss.backward()
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        if sched is not None:
            sched.step()
        step += 1

        if wb is not None and (step % log_every == 0 or step == 1):
            import wandb
            payload = {
                "train/loss_step": float(loss),
                "train/data_wait_s": data_wait_s,
                "step": step,
            }
            if device.type == "cuda":
                try:
                    payload["train/gpu_util"] = float(
                        torch.cuda.utilization(device.index or 0)
                    )
                except Exception:
                    payload["train/gpu_util"] = float("nan")
                try:
                    payload["train/gpu_mem_frac"] = float(
                        torch.cuda.memory_allocated(device) / max(torch.cuda.get_device_properties(device).total_memory, 1)
                    )
                except Exception:
                    pass
            wandb.log(payload)

        if step % int(tcfg.get("val_every", 50)) == 0 or step == max_steps:
            osm = one_step_mse(model, val_loader, device, store.norm)
            rom = rollout_mse(
                model, store, splits["val"], device, store.norm,
                K=int(tcfg.get("rollout_K", 100)), C=dcfg.get("C", 5),
                node_features=dcfg.get("node_features"),
                globals_fields=dcfg.get("globals_fields"),
                edge_sets=edge_sets,
                radius=dcfg.get("radius", 0.1),
                normalize_features=dcfg.get("normalize_features", True),
            )
            print(f"step {step}: train_loss={float(loss):.4e} val_1step={osm:.4e} val_rollout={rom:.4e}")
            if wb is not None:
                import wandb
                wandb.log({"train/loss": float(loss), "val/one_step_mse": osm, "val/rollout_mse": rom, "step": step})
            metric = rom if np.isfinite(rom) else osm
            torch.save({"model": model.state_dict(), "meta": run_meta, "cfg": cfg, "step": step}, ckpt_dir / "last.pt")
            if metric < best:
                best = metric
                bad = 0
                torch.save({"model": model.state_dict(), "meta": run_meta, "cfg": cfg, "step": step}, ckpt_dir / "best.pt")
            else:
                bad += 1
                if bad >= patience:
                    print("early stopping")
                    break
            model.train()

    run_meta["train_time_s"] = time.time() - t0
    with open(ckpt_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    # Refresh ckpt meta so evaluate can read C_train without run_meta fallback
    for name in ("last.pt", "best.pt"):
        p = ckpt_dir / name
        if not p.exists():
            continue
        try:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            meta = dict(blob.get("meta") or {})
            meta["train_time_s"] = run_meta["train_time_s"]
            blob["meta"] = meta
            torch.save(blob, p)
        except Exception:
            pass
    if wb is not None:
        import wandb
        wandb.finish()
    return ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--subset", type=str, default=None, help="unused alias; data_dir in config")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_yaml(_resolve(args.config) if not Path(args.config).is_absolute() else args.config)
    # allow configs relative to ml/
    if not _resolve(cfg["data"]["data_dir"]).exists():
        alt = repo_root() / "ml" / args.config
        pass
    device = _device(args.device)
    seeds = cfg.get("train", {}).get("seeds") or [cfg.get("train", {}).get("seed", 42)]
    for seed in seeds:
        path = train_one(cfg, device, int(seed))
        print(f"done → {path}")


if __name__ == "__main__":
    main()
