"""Train a registered surrogate: python -m ml.train --config configs/stub_gnn.yaml"""
from __future__ import annotations

import argparse
import json
import os
import threading
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


def _phase(name: str, t0: float | None = None) -> float:
    """Print a wall-clock phase timestamp; return now for chaining."""
    now = time.time()
    if t0 is None:
        print(f"PHASE {name} t={now:.3f}", flush=True)
    else:
        print(f"PHASE {name} t={now:.3f} dt={now - t0:.3f}s", flush=True)
    return now


def _mem_snapshot(tag: str = "") -> None:
    """Log host RAM / swap and RSS of this process (+ children if psutil)."""
    try:
        import psutil
    except ImportError:
        print(f"MEM {tag} psutil_unavailable", flush=True)
        return
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    proc = psutil.Process()
    kids = proc.children(recursive=True)
    rss = proc.memory_info().rss
    kid_rss = [(c.pid, c.memory_info().rss) for c in kids]
    print(
        f"MEM {tag} host_used_gb={vm.used / 1e9:.2f}/{vm.total / 1e9:.2f} "
        f"avail_gb={vm.available / 1e9:.2f} swap_used_gb={sw.used / 1e9:.2f} "
        f"main_rss_gb={rss / 1e9:.3f} workers={len(kids)} "
        f"worker_rss_gb={[round(r / 1e9, 3) for _, r in kid_rss]}",
        flush=True,
    )


class _NvmlUtilSampler:
    """Sample GPU util via NVML every 1s during *training* (paused in validation)."""

    def __init__(self, device_index: int = 0, interval_s: float = 1.0):
        self.device_index = int(device_index)
        self.interval_s = float(interval_s)
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception as e:
            print(f"NVML util sampler unavailable: {e}", flush=True)
            return
        self._thread = threading.Thread(target=self._loop, name="nvml-util", daemon=True)
        self._thread.start()
        print("PHASE nvml_util_sampling_started", flush=True)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def _loop(self) -> None:
        assert self._nvml is not None and self._handle is not None
        while not self._stop.wait(self.interval_s):
            if self._paused.is_set():
                continue
            try:
                u = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                self.samples.append(u)
            except Exception:
                break

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        n = len(self.samples)
        avg = float(np.mean(self.samples)) if n else float("nan")
        out = {"n": n, "avg_pct": avg, "samples": self.samples[-60:]}
        print(
            f"NVML_UTIL_SUMMARY n={n} avg_pct={avg:.2f} (train_steps_only)",
            flush=True,
        )
        try:
            if self._nvml is not None:
                self._nvml.nvmlShutdown()
        except Exception:
            pass
        return out


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
    t_start = _phase("config_load")
    _set_seed(seed)
    data_dir = _resolve(cfg["data"]["data_dir"])
    splits = load_splits(_resolve(cfg["data"]["splits"]))
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    cache_shards = int(dcfg.get("cache_shards", 4))
    num_workers = int(tcfg.get("num_workers", 0))
    print(
        f"PHASE cache_budget cache_shards={cache_shards} num_workers={num_workers} "
        f"(each worker has its own LRU; mmap shares OS pages)",
        flush=True,
    )
    _mem_snapshot("after_config")

    _phase("data_index_manifest", t_start)
    store = BallooningRunStore(
        data_dir,
        cache_shards=cache_shards,
        mmap=bool(dcfg.get("mmap", True)),
    )
    edge_sets = dcfg.get("edge_sets", ["material"])
    _mem_snapshot("after_store")

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
    t_ds = _phase("dataset_index_build_start", t_start)
    train_ds = BallooningDataset(
        store, splits["train"],
        noise_std=scalar_noise,
        noise_std_by_type=per_type_noise,
        **ds_kwargs,
    )
    _phase(f"train_dataset_ready n={len(train_ds)}", t_ds)
    _mem_snapshot("after_train_ds")
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
    _phase(f"val_dataset_ready n={len(val_ds)}", t_ds)
    _mem_snapshot("after_val_ds")

    t_loader = _phase("dataloader_construct", t_start)
    train_loader = make_loader(
        train_ds, batch_size=tcfg.get("batch_size", 4), shuffle=True,
        num_workers=num_workers, device=str(device),
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
    _phase("dataloader_ready", t_loader)

    # infer node_dim / edge_dims from a sample
    t_sample = _phase("first_sample_getitem", t_start)
    sample = train_ds[0]
    _phase("first_sample_done", t_sample)
    node_dim = int(sample.x.shape[-1])
    mcfg = dict(cfg["model"])
    name = mcfg.pop("name")
    t_model = _phase("model_build", t_start)
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
    _phase(f"model_ready params={model.count_parameters()}", t_model)
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
        "batch_size": int(tcfg.get("batch_size", 4)),
        "lr": float(tcfg.get("lr", 1e-3)),
        "noise_k": noise_k,
        "noise_std": scalar_noise,
        "noise_std_by_type": None if per_type_noise is None else per_type_noise.tolist(),
        "noise_stats_overall_rms": None if noise_stats is None else noise_stats.get("overall_rms"),
        "cache_shards": cache_shards,
        "num_workers": num_workers,
    }
    with open(ckpt_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Schedule: prefer max_epochs × steps_per_epoch over a raw max_steps cap.
    steps_per_epoch = max(1, int(np.ceil(len(train_ds) / max(int(tcfg.get("batch_size", 4)), 1))))
    max_epochs = tcfg.get("max_epochs", None)
    if max_epochs is not None:
        max_epochs = int(max_epochs)
        max_steps = int(max_epochs * steps_per_epoch)
    else:
        max_steps = int(tcfg.get("max_steps", 1000))
        max_epochs = int(np.ceil(max_steps / steps_per_epoch)) if max_steps > 0 else 0

    run_meta["max_epochs"] = max_epochs
    run_meta["steps_per_epoch"] = steps_per_epoch
    run_meta["max_steps"] = max_steps
    run_meta["n_train_frames"] = len(train_ds)

    # Cost cap for this VM session (shared across sequential configs via env).
    cost_usd_per_hour = float(
        os.environ.get(
            "VM_USD_PER_HOUR",
            str(tcfg.get("vm_usd_per_hour", 0.853624312)),  # g2-standard-8 + L4 us-central1
        )
    )
    cost_cap_usd = float(os.environ.get("COST_CAP_USD", str(tcfg.get("cost_cap_usd", 40.0))))
    session_t0 = float(os.environ.get("TRAIN_SESSION_T0", time.time()))
    run_meta["vm_usd_per_hour"] = cost_usd_per_hour
    run_meta["cost_cap_usd"] = cost_cap_usd

    t_wb = _phase("wandb_init", t_start)
    wb = _init_wandb(cfg, run_meta)
    _phase("wandb_ready", t_wb)
    # Rollout reference lines (constant_velocity / follow_air) if diagnose wrote them.
    rollout_refs = None
    for cand in (
        Path(os.environ.get("ROLLOUT_REFS_JSON", "")),
        _resolve("Tema 1/results/ml/logs/rollout_refs.json"),
        Path("/home/octav/poliphysics/Tema 1/results/ml/logs/rollout_refs.json"),
    ):
        if cand and str(cand) and cand.is_file():
            try:
                rollout_refs = json.loads(cand.read_text())
                print(f"PHASE loaded_rollout_refs {cand}", flush=True)
                break
            except Exception as e:
                print(f"rollout_refs load failed {cand}: {e}", flush=True)
    if wb is not None:
        try:
            import wandb
            wandb.config.update(
                {
                    "max_epochs": max_epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "max_steps": max_steps,
                    "batch_size": int(tcfg.get("batch_size", 4)),
                    "lr": float(tcfg.get("lr", 1e-4)),
                    "lr_end": float((tcfg.get("lr_schedule") or {}).get("lr_end", 1e-6)),
                    "early_stopping_patience_epochs": int(
                        (tcfg.get("early_stopping") or {}).get("patience_epochs", 2)
                    ),
                    "cost_cap_usd": cost_cap_usd,
                    "vm_usd_per_hour": cost_usd_per_hour,
                },
                allow_val_change=True,
            )
            if rollout_refs:
                wandb.summary["ref/constant_velocity_rollout_mse"] = float(
                    rollout_refs["constant_velocity"]
                )
                wandb.summary["ref/follow_air_rollout_mse"] = float(rollout_refs["follow_air"])
                wandb.summary["ref/rollout_horizon_steps"] = int(rollout_refs["horizon_steps"])
                wandb.summary["ref/rollout_horizon_s"] = float(rollout_refs["horizon_seconds"])
                if "gns_local_best_reported" in rollout_refs:
                    wandb.summary["ref/gns_local_best_rollout_mse"] = float(
                        rollout_refs["gns_local_best_reported"]
                    )
                # Constant lines at step 0 for charts.
                wandb.log(
                    {
                        "ref/constant_velocity_rollout_mse": float(
                            rollout_refs["constant_velocity"]
                        ),
                        "ref/follow_air_rollout_mse": float(rollout_refs["follow_air"]),
                        "step": 0,
                    }
                )
        except Exception:
            pass
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
        # lr_end = lr * gamma^(max_epochs * steps_per_epoch)
        lr_end = float(sched_cfg.get("lr_end", 1e-6))
        gamma = (lr_end / max(lr, 1e-12)) ** (1.0 / max(max_steps, 1))
        sched = ExponentialLR(opt, gamma=gamma)
    else:
        sched = None

    best = float("inf")
    es_cfg = tcfg.get("early_stopping") or {}
    patience_epochs = int(es_cfg.get("patience_epochs", es_cfg.get("patience", 2)))
    best_epoch = 0
    step = 0
    epoch = 0
    t0 = time.time()
    print(
        f"PHASE schedule max_epochs={max_epochs} steps_per_epoch={steps_per_epoch} "
        f"max_steps={max_steps} lr={lr}→{sched_cfg.get('lr_end', None)} "
        f"early_stop_patience_epochs={patience_epochs} "
        f"cost_cap_usd={cost_cap_usd} @ ${cost_usd_per_hour}/h",
        flush=True,
    )
    model.train()
    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)
    t_iter = _phase("dataloader_worker_start_iter", t_start)
    it = iter(train_loader)
    _phase("dataloader_iter_ready", t_iter)
    _mem_snapshot("after_loader_iter")
    log_every = int(tcfg.get("log_every", 10))
    logged_devices = False
    nvml = _NvmlUtilSampler(device_index=device.index or 0) if device.type == "cuda" else None
    util_gate_min = float(os.environ.get("UTIL_GATE_MIN", "15"))
    util_gate_pct = float(os.environ.get("UTIL_GATE_PCT", "70"))
    util_gate_val_frac_max = float(os.environ.get("UTIL_GATE_VAL_FRAC", "0.10"))
    util_gate_enabled = os.environ.get("UTIL_GATE", "0") == "1"
    util_gate_deadline: float | None = None
    mem_every_s = float(os.environ.get("MEM_SAMPLE_EVERY_S", "10"))
    next_mem_t = time.time() + mem_every_s

    from ml.validation import (
        fixed_val_frame_indices,
        one_step_mse_sample,
        rollout_mse_batched,
        stratified_val_runs,
    )

    val_every = int(tcfg.get("val_every", 1000))  # one-step sample cadence
    val_one_step_frames = int(tcfg.get("val_one_step_frames", 256))
    val_rollout_runs_n = int(tcfg.get("val_rollout_runs", 32))
    val_rollout_K = int(tcfg.get("rollout_K", 20))
    val_frame_idx = fixed_val_frame_indices(val_ds, n_frames=val_one_step_frames, seed=seed)
    val_rollout_ids = stratified_val_runs(
        store, splits["val"], n_runs=val_rollout_runs_n, seed=seed
    )
    # Default: defer rollout until we can size it to ≤10% wall; override via config.
    val_rollout_every = tcfg.get("val_rollout_every", None)
    val_time_s = 0.0
    train_time_s = 0.0
    last_train_t = time.perf_counter()
    rollout_every_reported = False
    print(
        f"PHASE train_loop_enter max_steps={max_steps} batch_size={tcfg.get('batch_size')} "
        f"lr={lr} util_gate={util_gate_enabled} val_every={val_every} "
        f"val_rollout_runs={len(val_rollout_ids)} val_one_step_frames={len(val_frame_idx)}",
        flush=True,
    )
    stop_reason = None
    while step < max_steps:
        # Cost cap (session wall clock × $/h).
        session_cost = (time.time() - session_t0) / 3600.0 * cost_usd_per_hour
        if session_cost > cost_cap_usd:
            stop_reason = f"cost_cap (${session_cost:.2f} > ${cost_cap_usd:.2f})"
            print(f"COST_CAP STOP: {stop_reason}", flush=True)
            torch.save(
                {"model": model.state_dict(), "meta": run_meta, "cfg": cfg, "step": step, "epoch": epoch},
                ckpt_dir / "last.pt",
            )
            if not (ckpt_dir / "best.pt").exists():
                torch.save(
                    {"model": model.state_dict(), "meta": run_meta, "cfg": cfg, "step": step, "epoch": epoch},
                    ckpt_dir / "best.pt",
                )
            run_meta["stop_reason"] = stop_reason
            run_meta["session_cost_usd"] = session_cost
            with open(ckpt_dir / "run_meta.json", "w") as f:
                json.dump(run_meta, f, indent=2)
            if wb is not None:
                import wandb
                wandb.finish()
            raise SystemExit(43)

        t_wait0 = time.perf_counter()
        epoch_ended = False
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            epoch_ended = True
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if epoch >= max_epochs:
                stop_reason = f"max_epochs={max_epochs}"
                print(f"EPOCH STOP: completed {epoch} epochs", flush=True)
                break
            # Early stop after `patience_epochs` full epochs with no rollout improve.
            if epoch - best_epoch > patience_epochs and best < float("inf"):
                stop_reason = (
                    f"early_stopping (no rollout improve for {patience_epochs} epochs; "
                    f"best_epoch={best_epoch} best={best:.4e})"
                )
                print(f"EARLY_STOP: {stop_reason}", flush=True)
                break
            it = iter(train_loader)
            batch = next(it)
        data_wait_s = time.perf_counter() - t_wait0
        if step == 0:
            _phase("first_batch_fetched", t_start)
            _mem_snapshot("first_batch")
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
        if step == 1:
            _phase("first_step_done", t_start)
            if nvml is not None:
                nvml.start()
                if util_gate_enabled:
                    util_gate_deadline = time.time() + util_gate_min * 60.0
                    print(
                        f"UTIL_GATE armed: avg>={util_gate_pct}% over {util_gate_min} min "
                        f"from first step",
                        flush=True,
                    )
        now = time.time()
        session_cost = (now - session_t0) / 3600.0 * cost_usd_per_hour
        # Log every step for the first 50 (stall diagnosis); then every log_every.
        if step <= 50 or step % log_every == 0:
            print(
                f"STEP {step} epoch={epoch} loss={float(loss):.4e} "
                f"data_wait_s={data_wait_s:.4f} elapsed_s={now - t0:.1f} "
                f"session_cost_usd={session_cost:.2f}",
                flush=True,
            )
        if now >= next_mem_t:
            _mem_snapshot(f"step_{step}")
            next_mem_t = now + mem_every_s

        # Accumulate pure train wall (excluding validation pauses).
        train_time_s += time.perf_counter() - last_train_t
        last_train_t = time.perf_counter()

        if util_gate_deadline is not None and now >= util_gate_deadline and nvml is not None:
            summary = nvml.stop()
            avg = float(summary["avg_pct"])
            wall = max(train_time_s + val_time_s, 1e-9)
            val_frac = val_time_s / wall
            print(
                f"UTIL_GATE_AVG_{util_gate_min:g}m_train={avg:.2f} (n={summary['n']}) "
                f"val_frac={val_frac:.4f} (max={util_gate_val_frac_max})",
                flush=True,
            )
            run_meta["util_gate"] = {
                **summary,
                "val_frac": val_frac,
                "train_time_s": train_time_s,
                "val_time_s": val_time_s,
            }
            fail_util = summary["n"] > 0 and avg < util_gate_pct
            fail_val = val_frac > util_gate_val_frac_max
            if fail_util or fail_val:
                print(
                    f"UTIL_GATE FAIL util={avg:.1f}% (need>={util_gate_pct}) "
                    f"val_frac={val_frac:.3f} (need<={util_gate_val_frac_max}) — STOP.",
                    flush=True,
                )
                torch.save(
                    {"model": model.state_dict(), "meta": run_meta, "cfg": cfg, "step": step},
                    ckpt_dir / "last.pt",
                )
                if wb is not None:
                    import wandb
                    wandb.finish()
                raise SystemExit(42)
            print("UTIL_GATE PASS", flush=True)
            util_gate_deadline = None
            nvml = _NvmlUtilSampler(device_index=device.index or 0)
            nvml.start()

        if wb is not None and (step % log_every == 0 or step == 1):
            import wandb
            wall = max(train_time_s + val_time_s, 1e-9)
            cur_lr = float(opt.param_groups[0]["lr"])
            payload = {
                "train/loss_step": float(loss),
                "train/data_wait_s": data_wait_s,
                "train/batch_size": int(tcfg.get("batch_size", 4)),
                "train/lr": cur_lr,
                "train/epoch": epoch,
                "train/session_cost_usd": session_cost,
                "val/wall_frac": val_time_s / wall,
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

        do_one_step = step % val_every == 0 or step == max_steps
        # Auto-size rollout cadence once we know one rollout cost.
        if val_rollout_every is None:
            do_rollout = step == val_every  # first probe at first one-step val
        else:
            do_rollout = step % int(val_rollout_every) == 0 or step == max_steps

        if do_one_step or do_rollout:
            if nvml is not None:
                nvml.pause()
            t_val0 = time.perf_counter()
            _phase(f"val_start_step_{step}", t_start)
            osm = float("nan")
            rom = float("nan")
            rollout_dt = 0.0
            if do_one_step:
                osm = one_step_mse_sample(
                    model, val_ds, val_frame_idx, device,
                    batch_size=int(tcfg.get("batch_size", 16)),
                )
            if do_rollout:
                t_roll0 = time.perf_counter()
                rom, rtiming = rollout_mse_batched(
                    model, store, val_rollout_ids, device, store.norm,
                    K=val_rollout_K, C=dcfg.get("C", 5),
                    node_features=dcfg.get("node_features"),
                    globals_fields=dcfg.get("globals_fields"),
                    edge_sets=edge_sets,
                    radius=dcfg.get("radius", 0.1),
                    normalize_features=dcfg.get("normalize_features", True),
                )
                rollout_dt = time.perf_counter() - t_roll0
            dt_val = time.perf_counter() - t_val0
            val_time_s += dt_val
            if do_rollout and not rollout_every_reported and train_time_s > 1.0 and step > 0:
                sps = step / max(train_time_s, 1e-9)
                # Size cadence so *rollout* ≤ 10% wall (one-step sample is small).
                auto_every = int(max(val_every, np.ceil(10.0 * max(rollout_dt, 1e-3) * sps)))
                if val_rollout_every is None:
                    val_rollout_every = auto_every
                else:
                    val_rollout_every = max(int(val_rollout_every), auto_every)
                print(
                    f"VAL_ROLLOUT_EVERY={val_rollout_every} "
                    f"(target rollout_frac<=10%; rollout_dt={rollout_dt:.2f}s sps={sps:.2f})",
                    flush=True,
                )
                run_meta["val_rollout_every"] = int(val_rollout_every)
                rollout_every_reported = True
            wall = max(train_time_s + val_time_s, 1e-9)
            val_frac = val_time_s / wall
            print(
                f"step {step}: train_loss={float(loss):.4e} "
                f"val_1step={osm:.4e} val_rollout={rom:.4e} "
                f"val_frac={val_frac:.3f} val_dt={dt_val:.2f}s",
                flush=True,
            )
            if wb is not None:
                import wandb
                payload = {
                    "train/loss": float(loss),
                    "train/epoch": epoch,
                    "train/session_cost_usd": (time.time() - session_t0) / 3600.0 * cost_usd_per_hour,
                    "val/wall_frac": val_frac,
                    "val/dt_s": dt_val,
                    "step": step,
                }
                if np.isfinite(osm):
                    payload["val/one_step_mse"] = osm
                if np.isfinite(rom):
                    payload["val/rollout_mse"] = rom
                    payload["val/rollout_every"] = int(val_rollout_every or 0)
                wandb.log(payload)
            # Checkpoint selection driven by rollout metric.
            torch.save(
                {"model": model.state_dict(), "meta": run_meta, "cfg": cfg, "step": step, "epoch": epoch},
                ckpt_dir / "last.pt",
            )
            if np.isfinite(rom) and rom < best:
                best = rom
                best_epoch = epoch
                torch.save(
                    {"model": model.state_dict(), "meta": run_meta, "cfg": cfg, "step": step, "epoch": epoch},
                    ckpt_dir / "best.pt",
                )
                print(f"NEW_BEST rollout={rom:.4e} epoch={epoch} step={step}", flush=True)
            model.train()
            if nvml is not None:
                nvml.resume()
            last_train_t = time.perf_counter()

        if epoch_ended and wb is not None:
            import wandb
            wandb.log({
                "train/epoch": epoch,
                "train/session_cost_usd": (time.time() - session_t0) / 3600.0 * cost_usd_per_hour,
                "step": step,
            })

    if nvml is not None:
        run_meta["nvml_util"] = nvml.stop()
    run_meta["train_time_s"] = time.time() - t0
    run_meta["val_time_s"] = val_time_s
    run_meta["val_wall_frac"] = val_time_s / max(train_time_s + val_time_s, 1e-9)
    run_meta["epochs_completed"] = epoch
    run_meta["best_rollout_mse"] = best if best < float("inf") else None
    run_meta["best_epoch"] = best_epoch
    run_meta["stop_reason"] = stop_reason or ("max_steps" if step >= max_steps else "completed")
    run_meta["session_cost_usd"] = (time.time() - session_t0) / 3600.0 * cost_usd_per_hour
    if val_rollout_every is not None:
        run_meta["val_rollout_every"] = int(val_rollout_every)
    if not (ckpt_dir / "best.pt").exists() and (ckpt_dir / "last.pt").exists():
        import shutil
        shutil.copy2(ckpt_dir / "last.pt", ckpt_dir / "best.pt")
    print(
        f"TRAIN_DONE reason={run_meta['stop_reason']} steps={step} epoch={epoch} "
        f"best_rollout={best} session_cost_usd={run_meta['session_cost_usd']:.2f}",
        flush=True,
    )
    with open(ckpt_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    # Refresh ckpt meta so evaluate can read C_train without run_meta fallback
    for ckpt_name in ("last.pt", "best.pt"):
        p = ckpt_dir / ckpt_name
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
