"""Loss-scale + rollout reference diagnostics (CPU/MPS; safe alongside GPU train).

  PYTHONPATH=. python -m ml.scripts.diagnose_loss_refs \\
    --config ml/configs/gns_local_gpu.yaml \\
    --checkpoint "Tema 1/results/ml/runs/<run>/best.pt" \\
    --device cpu --out Tema\\ 1/results/ml/logs/diagnose_loss_refs.json

Writes rollout_refs.json for W&B constant lines (picked up by train at start).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ml.data.dataset import BallooningDataset, BallooningRunStore, make_loader
from ml.data.noise_stats import a_std_by_type_matrix, resolve_noise
from ml.data.splits import load_splits
from ml.models import build_model
from ml.models.baselines import ConstantVelocity, FollowAir
from ml.train import _device, _resolve, _set_seed, one_step_mse
from ml.utils.config import load_yaml
from ml.utils.git_info import repo_root
from ml.validation import (
    fixed_val_frame_indices,
    one_step_mse_by_type,
    one_step_mse_sample,
    rollout_mse_batched,
    stratified_val_runs,
)


def _loss_batches(model, loader, device, n_batches: int) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = batch.to(device)
            pred = model(batch)
            # Exact training loss: mean over all elements (nodes × 3).
            err = (pred - batch.y).pow(2).mean()
            total += float(err) * batch.num_graphs
            n += batch.num_graphs
    return total / max(n, 1)


def _zero_acc_loss(loader, store, device, n_batches: int) -> float:
    """Loss of predicting physical a=0 → normalized (-mean/std)."""
    a_mean = torch.as_tensor(store.norm["a"]["mean"], dtype=torch.float32, device=device).view(1, 3)
    a_std = torch.as_tensor(store.norm["a"]["std"], dtype=torch.float32, device=device).view(1, 3)
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = batch.to(device)
            zeros = torch.zeros(batch.y.shape[0], 3, device=device, dtype=batch.y.dtype)
            pred = (zeros - a_mean) / a_std
            err = (pred - batch.y).pow(2).mean()
            total += float(err) * batch.num_graphs
            n += batch.num_graphs
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs-out", type=Path, default=None,
                    help="rollout_refs.json for train.py / W&B (default next to --out)")
    ap.add_argument("--wandb-run-path", type=str, default=None,
                    help="optional entity/project/run_id to summary-log refs onto existing run")
    ap.add_argument(
        "--require-below-zero-acc",
        action="store_true",
        help="exit 44 if clean ckpt loss is not below zero-acc overall on train and val",
    )
    args = ap.parse_args()

    cfg = load_yaml(_resolve(args.config) if not Path(args.config).is_absolute() else args.config)
    device = _device(args.device)
    seed = int(args.seed)
    _set_seed(seed)
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    data_dir = Path(dcfg["data_dir"]) if Path(dcfg["data_dir"]).is_absolute() else _resolve(dcfg["data_dir"])
    splits = load_splits(
        Path(dcfg["splits"]) if Path(dcfg["splits"]).is_absolute() else _resolve(dcfg["splits"])
    )
    store = BallooningRunStore(
        data_dir,
        cache_shards=int(dcfg.get("cache_shards", 2)),
        mmap=bool(dcfg.get("mmap", True)),
    )
    edge_sets = dcfg.get("edge_sets", ["material"])
    noise_mode = str(dcfg.get("noise_mode", "per_type"))
    scalar_noise, per_type_comp, comp_noise = resolve_noise(
        noise_k=dcfg.get("noise_k"),
        noise_std=dcfg.get("noise_std"),
        noise_mode=noise_mode,
        a_std=store.norm.get("a", {}).get("std"),
        a_std_by_type=a_std_by_type_matrix(store.norm),
        dt=float(store.norm.get("dt", 0.01)),
    )
    ds_kw = dict(
        C=dcfg.get("C", 5),
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets,
        radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
        sample_weight=dcfg.get("sample_weight", "per_run"),
        rotate_augment=False,
        seed=seed,
    )
    train_noisy = BallooningDataset(
        store, splits["train"],
        noise_std=scalar_noise if per_type_comp is None and comp_noise is None else 0.0,
        noise_std_by_type_comp=per_type_comp,
        noise_std_components=comp_noise,
        **ds_kw,
    )
    train_clean = BallooningDataset(
        store, splits["train"],
        noise_std=0.0, noise_std_by_type_comp=None, noise_std_components=None, **ds_kw,
    )
    val_ds = BallooningDataset(
        store, splits["val"],
        noise_std=0.0, sample_weight="per_frame",
        C=ds_kw["C"], node_features=ds_kw["node_features"],
        globals_fields=ds_kw["globals_fields"], edge_sets=edge_sets,
        radius=ds_kw["radius"], normalize_features=ds_kw["normalize_features"],
        rotate_augment=False, seed=seed,
    )
    # Fixed 50 batches: same sampler seed → reproducible.
    loader_noisy = make_loader(
        train_noisy, batch_size=int(tcfg.get("batch_size", 16)), shuffle=True,
        num_workers=0, device=str(device), shard_aware=True, seed=seed,
    )
    loader_clean = make_loader(
        train_clean, batch_size=int(tcfg.get("batch_size", 16)), shuffle=True,
        num_workers=0, device=str(device), shard_aware=True, seed=seed,
    )

    loss_def = {
        "formula": "mean_i,c (pred_{i,c} - y_{i,c})^2",
        "aggregation": "torch mean over all nodes and 3 acceleration components in the batch",
        "per_type_weighting": False,
        "target": "a_t = (v_{t+1}-v_t)/dt, then normalize: (a - a_mean)/a_std from TRAIN norm_stats",
        "a_mean": np.asarray(store.norm["a"]["mean"]).tolist(),
        "a_std": np.asarray(store.norm["a"]["std"]).tolist(),
        "noise": (
            "RW over C velocities; noise_mode=per_type → sigma[type,c]=k*a_std_type[c]*dt; "
            "global → k*a_std[c]*dt; increments /sqrt(C); "
            "target a=(v_{t+1}^gt - v_t^noisy)/dt; target norm stays global"
        ),
        "noise_k": dcfg.get("noise_k"),
        "noise_mode": noise_mode,
        "noise_std_by_type_comp": (
            None if per_type_comp is None else np.asarray(per_type_comp).tolist()
        ),
        "noise_std_components": (
            None if comp_noise is None else np.asarray(comp_noise).tolist()
        ),
        "a_by_type": (
            None
            if "a_by_type" not in store.norm
            else {
                k: {"std": np.asarray(v["std"]).tolist(), "count": int(v["count"])}
                for k, v in store.norm["a_by_type"].items()
            }
        ),
        "zero_acc_note": "physical a=0 → normalized pred = -a_mean/a_std",
    }

    n_fixed = int(tcfg.get("val_one_step_frames", 256))
    train_idx = fixed_val_frame_indices(train_clean, n_frames=n_fixed, seed=seed)
    val_idx = fixed_val_frame_indices(val_ds, n_frames=n_fixed, seed=seed)

    print("Computing zero-acc losses...", flush=True)
    zero_noisy = _zero_acc_loss(loader_noisy, store, device, args.n_batches)
    zero_clean = _zero_acc_loss(loader_clean, store, device, args.n_batches)
    # Zero-acc on the same fixed index sets used for one-step ckpt comparison.
    class _ZeroAcc:
        def eval(self):
            return self

        def __call__(self, batch):
            a_mean = torch.as_tensor(
                store.norm["a"]["mean"], dtype=torch.float32, device=batch.y.device
            ).view(1, 3)
            a_std = torch.as_tensor(
                store.norm["a"]["std"], dtype=torch.float32, device=batch.y.device
            ).view(1, 3)
            zeros = torch.zeros(batch.y.shape[0], 3, device=batch.y.device, dtype=batch.y.dtype)
            return (zeros - a_mean) / a_std

    zero_model = _ZeroAcc()
    bs = int(tcfg.get("batch_size", 16))
    zero_train_by_type = one_step_mse_by_type(zero_model, train_clean, train_idx, device, batch_size=bs)
    zero_val_by_type = one_step_mse_by_type(zero_model, val_ds, val_idx, device, batch_size=bs)
    zero_train_fixed = zero_train_by_type["overall"]
    zero_val_fixed = zero_val_by_type["overall"]
    print(
        f"zero_noisy={zero_noisy:.6e} zero_clean={zero_clean:.6e} "
        f"zero_train_fixed256={zero_train_fixed:.6e} zero_val_fixed256={zero_val_fixed:.6e}",
        flush=True,
    )
    print(f"zero_train_by_type={zero_train_by_type}", flush=True)
    print(f"zero_val_by_type={zero_val_by_type}", flush=True)

    ckpt_noisy = ckpt_clean = val_one_step = train_one_step_fixed = float("nan")
    ckpt_path = args.checkpoint
    model = None
    blob = {}
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path) if Path(ckpt_path).is_absolute() else _resolve(ckpt_path)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sample = train_clean[0]
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
        model.load_state_dict(blob["model"])
        model.to(device).eval()
        print("Computing checkpoint losses...", flush=True)
        ckpt_noisy = _loss_batches(model, loader_noisy, device, args.n_batches)
        ckpt_clean = _loss_batches(model, loader_clean, device, args.n_batches)
        train_by_type = one_step_mse_by_type(model, train_clean, train_idx, device, batch_size=bs)
        val_by_type = one_step_mse_by_type(model, val_ds, val_idx, device, batch_size=bs)
        train_one_step_fixed = train_by_type["overall"]
        val_one_step = val_by_type["overall"]
        print(
            f"ckpt_noisy={ckpt_noisy:.6e} ckpt_clean={ckpt_clean:.6e} "
            f"train_1step_fixed={train_one_step_fixed:.6e} val_1step={val_one_step:.6e}",
            flush=True,
        )
        print(f"ckpt_train_by_type={train_by_type}", flush=True)
        print(f"ckpt_val_by_type={val_by_type}", flush=True)
    else:
        train_by_type = {k: float("nan") for k in ("overall", "spider", "interior", "tip")}
        val_by_type = {k: float("nan") for k in ("overall", "spider", "interior", "tip")}

    # Irreducible noise: excess of noisy zero-acc over clean zero-acc,
    # and excess of noisy ckpt over clean ckpt.
    irreducible = {
        "zero_noisy_minus_clean": float(zero_noisy - zero_clean),
        "ckpt_noisy_minus_clean": float(ckpt_noisy - ckpt_clean) if np.isfinite(ckpt_noisy) else None,
        "interpretation": (
            "zero_noisy - zero_clean ≈ training-loss floor from RW noise at k "
            "(targets shift when latest v is noised). ckpt_noisy - ckpt_clean is "
            "the same gap for the fitted model."
        ),
    }

    # Rollout refs on same stratified 32 runs
    K = int(tcfg.get("rollout_K", 20))
    C = int(dcfg.get("C", 5))
    n_runs = int(tcfg.get("val_rollout_runs", 32))
    val_ids = stratified_val_runs(store, splits["val"], n_runs=n_runs, seed=seed)
    print(f"Rollout refs on {len(val_ids)} runs K={K} ({K * 0.01:.2f}s)...", flush=True)

    cv = ConstantVelocity(a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"]).to(device)
    fa = FollowAir(a_mean=store.norm["a"]["mean"], a_std=store.norm["a"]["std"]).to(device)
    rom_cv, t_cv = rollout_mse_batched(
        cv, store, val_ids, device, store.norm, K=K, C=C,
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets, radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
    )
    rom_fa, t_fa = rollout_mse_batched(
        fa, store, val_ids, device, store.norm, K=K, C=C,
        node_features=dcfg.get("node_features"),
        globals_fields=dcfg.get("globals_fields"),
        edge_sets=edge_sets, radius=dcfg.get("radius", 0.1),
        normalize_features=dcfg.get("normalize_features", True),
    )
    # gns_local best from user / meta if present
    gns_local_best = None
    if ckpt_path is not None and ckpt_path.exists():
        meta = blob.get("meta") or {}
        gns_local_best = meta.get("best_rollout_mse")
    if gns_local_best is None:
        gns_local_best = 2.79e-5  # reported at step 72k

    refs = {
        "metric": "val_rollout_mse",
        "units": "m^2_rel_pos",
        "horizon_steps": K,
        "horizon_seconds": K * 0.01,
        "n_runs": len(val_ids),
        "run_ids": val_ids,
        "seed": seed,
        "constant_velocity": float(rom_cv),
        "follow_air": float(rom_fa),
        "gns_local_best_reported": float(gns_local_best),
        "timing_cv": t_cv,
        "timing_fa": t_fa,
    }
    print(json.dumps(refs, indent=2), flush=True)

    report = {
        "loss_definition": loss_def,
        "n_batches": args.n_batches,
        "batch_size": int(tcfg.get("batch_size", 16)),
        "losses": {
            "zero_acc_noisy": zero_noisy,
            "zero_acc_clean": zero_clean,
            "zero_acc_train_fixed256": zero_train_fixed,
            "zero_acc_val_fixed256": zero_val_fixed,
            "zero_acc_train_by_type": zero_train_by_type,
            "zero_acc_val_by_type": zero_val_by_type,
            "best_ckpt_noisy": ckpt_noisy,
            "best_ckpt_clean": ckpt_clean,
            "train_one_step_clean_fixed256": train_one_step_fixed,
            "val_one_step_clean": val_one_step,
            "ckpt_train_by_type": train_by_type,
            "ckpt_val_by_type": val_by_type,
        },
        "sanity_below_zero_acc": {
            "train_overall": (
                bool(train_one_step_fixed < zero_train_fixed)
                if np.isfinite(train_one_step_fixed)
                else None
            ),
            "val_overall": (
                bool(val_one_step < zero_val_fixed) if np.isfinite(val_one_step) else None
            ),
            "train_by_type": {
                k: bool(train_by_type[k] < zero_train_by_type[k])
                for k in ("spider", "interior", "tip")
                if np.isfinite(train_by_type.get(k, float("nan")))
            },
            "val_by_type": {
                k: bool(val_by_type[k] < zero_val_by_type[k])
                for k in ("spider", "interior", "tip")
                if np.isfinite(val_by_type.get(k, float("nan")))
            },
        },
        "frame_sets": {
            "n_fixed": n_fixed,
            "train_indices_head": train_idx[:8],
            "val_indices_head": val_idx[:8],
            "same_norm": True,
            "same_window_ok_filter": True,
            "same_u_source": "stored rec['u'] / u_next_frame (not ks_torch)",
            "graph": "built per sample from absolute positions at t",
        },
        "irreducible_noise": irreducible,
        "rollout_refs": refs,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "created_unix": time.time(),
    }
    args.out = Path(args.out) if Path(args.out).is_absolute() else _resolve(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    refs_path = args.refs_out
    if refs_path is None:
        refs_path = args.out.parent / "rollout_refs.json"
    else:
        refs_path = Path(refs_path) if Path(refs_path).is_absolute() else _resolve(refs_path)
    refs_path.parent.mkdir(parents=True, exist_ok=True)
    refs_path.write_text(json.dumps(refs, indent=2))
    print(f"wrote {args.out}", flush=True)
    print(f"wrote {refs_path}", flush=True)

    # Optional: patch existing W&B run summary
    if args.wandb_run_path:
        try:
            import wandb
            api = wandb.Api()
            run = api.run(args.wandb_run_path)
            run.summary["ref/constant_velocity_rollout_mse"] = float(rom_cv)
            run.summary["ref/follow_air_rollout_mse"] = float(rom_fa)
            run.summary["ref/rollout_horizon_steps"] = K
            run.summary["ref/rollout_horizon_s"] = K * 0.01
            run.summary.update()
            print(f"updated W&B summary {args.wandb_run_path}", flush=True)
        except Exception as e:
            print(f"W&B summary update failed: {e}", flush=True)

    if args.require_below_zero_acc and ckpt_path is not None:
        ok = report["sanity_below_zero_acc"]
        train_ok = bool(ok.get("train_overall"))
        val_ok = bool(ok.get("val_overall"))
        train_types = ok.get("train_by_type") or {}
        val_types = ok.get("val_by_type") or {}
        types_ok = all(train_types.get(k) for k in ("spider", "interior", "tip")) and all(
            val_types.get(k) for k in ("spider", "interior", "tip")
        )
        if not (train_ok and val_ok):
            print(
                f"SANITY_FAIL train_ok={train_ok} val_ok={val_ok} types_ok={types_ok} detail={ok}",
                flush=True,
            )
            raise SystemExit(44)
        print(f"SANITY_OK below zero-acc (overall; per-type in JSON): {ok}", flush=True)


if __name__ == "__main__":
    main()
