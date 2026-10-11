"""Profile one validation pass: one_step_mse + instrumented rollout_mse.

  cd repo && source ml/.venv/bin/activate
  PYTHONPATH=. python -m ml.scripts.profile_val_pass

Uses ml.train.one_step_mse and mirrors train.rollout_mse with per-step buckets.
"""
from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ml.data.dataset import BallooningDataset, BallooningRunStore, make_loader
from ml.data.integrator import step_torch, DT
from ml.data.splits import load_splits
from ml.models import build_model
from ml.train import _device, _resolve, _set_seed, one_step_mse
from ml.utils.config import load_yaml


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass
class BucketTotals:
    model_forward: float = 0.0
    air_velocity: float = 0.0
    graph_construction: float = 0.0
    python_overhead: float = 0.0
    n_steps: int = 0

    def add_step(self, model: float, air: float, graph: float, overhead: float) -> None:
        self.model_forward += model
        self.air_velocity += air
        self.graph_construction += graph
        self.python_overhead += overhead
        self.n_steps += 1

    @property
    def wall(self) -> float:
        return self.model_forward + self.air_velocity + self.graph_construction + self.python_overhead


@contextlib.contextmanager
def _air_time_patch():
    """Accumulate wall time inside u_next_frame / make_ks (also dataset import)."""
    import ml.data.air as air_mod
    import ml.data.dataset as ds_mod

    acc = {"dt": 0.0}

    def wrap(fn):
        def inner(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            acc["dt"] += time.perf_counter() - t0
            return out

        return inner

    orig_unext = air_mod.u_next_frame
    orig_ks = air_mod.make_ks
    patched_unext = wrap(orig_unext)
    patched_ks = wrap(orig_ks)
    air_mod.u_next_frame = patched_unext
    air_mod.make_ks = patched_ks
    ds_mod.u_next_frame = patched_unext
    try:
        yield acc
    finally:
        air_mod.u_next_frame = orig_unext
        air_mod.make_ks = orig_ks
        ds_mod.u_next_frame = orig_unext


@torch.no_grad()
def profile_rollout_mse(
    model,
    store,
    run_ids,
    device,
    norm,
    *,
    K: int = 20,
    C: int = 5,
    max_runs: int = 8,
    node_features=None,
    globals_fields=None,
    edge_sets=None,
    radius: float = 0.1,
    normalize_features: bool = True,
) -> tuple[float, BucketTotals, dict[str, BucketTotals]]:
    """Same logic as train.rollout_mse with per-step timing buckets."""
    model.eval()
    a_mean = torch.as_tensor(norm["a"]["mean"], device=device)
    a_std = torch.as_tensor(norm["a"]["std"], device=device)
    edge_sets = list(edge_sets or model.required_edges() or ["material"])
    errs: list[float] = []
    totals = BucketTotals()
    per_run: dict[str, BucketTotals] = {}

    with _air_time_patch() as air_acc:
        for rid in list(run_ids)[:max_runs]:
            run_tot = BucketTotals()
            per_run[str(rid)] = run_tot
            rec = store.get(rid)
            T = rec["v"].shape[0]
            t0 = C
            if t0 + K >= T:
                continue
            x = torch.from_numpy(rec["x_rel"][t0].astype(np.float32)).to(device)
            v = torch.from_numpy(rec["v"][t0].astype(np.float32)).to(device)
            ds = BallooningDataset(
                store,
                [rid],
                C=C,
                node_features=node_features,
                globals_fields=globals_fields,
                edge_sets=edge_sets,
                radius=radius,
                normalize_features=normalize_features,
                normalize_target=True,
                noise_std=0.0,
            )
            idx_map = {t: i for i, (_, t) in enumerate(ds.index)}
            for k in range(K):
                t = t0 + k
                if t not in idx_map:
                    break
                step_t0 = time.perf_counter()
                air_before = air_acc["dt"]
                g0 = time.perf_counter()
                data = ds[idx_map[t]]
                g1 = time.perf_counter()
                air_step = air_acc["dt"] - air_before
                graph_step = (g1 - g0) - air_step

                data = data.to(device)
                data.pos = x
                data.v_t = v

                m0 = time.perf_counter()
                pred_n = model(data)
                _sync(device)
                m1 = time.perf_counter()
                model_step = m1 - m0

                i0 = time.perf_counter()
                a = pred_n * a_std + a_mean
                x, v = step_torch(x, v, a, dt=DT)
                x_gt = torch.from_numpy(rec["x_rel"][t0 + k + 1].astype(np.float32)).to(device)
                errs.append(float((x - x_gt).pow(2).mean()))
                step_t1 = time.perf_counter()
                overhead_step = (step_t1 - step_t0) - model_step - air_step - graph_step
                if overhead_step < 0:
                    overhead_step = 0.0

                totals.add_step(model_step, air_step, graph_step, overhead_step)
                run_tot.add_step(model_step, air_step, graph_step, overhead_step)

    rom = float(np.mean(errs)) if errs else float("nan")
    return rom, totals, per_run


def _build_model_and_val(cfg: dict, device: torch.device, seed: int):
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    data_dir = _resolve(dcfg["data_dir"])
    splits = load_splits(_resolve(dcfg["splits"]))
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
        rotate_augment=False,
        seed=seed,
    )
    val_ds = BallooningDataset(
        store,
        splits["val"],
        noise_std=0.0,
        sample_weight="per_frame",
        **ds_kwargs,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=tcfg.get("batch_size", 2),
        shuffle=False,
        num_workers=0,
        device=str(device),
        shard_aware=False,
        seed=seed,
    )
    sample = val_ds[0]
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
    return model, store, splits, val_loader, dcfg, tcfg


@torch.no_grad()
def profile_one_step_pass(model, val_loader, device, norm) -> dict:
    """Time one full val pass via train.one_step_mse."""
    n_batches = len(val_loader)
    n_graphs = len(val_loader.dataset)
    t0 = time.perf_counter()
    mse = one_step_mse(model, val_loader, device, norm)
    _sync(device)
    wall = time.perf_counter() - t0
    return {
        "n_batches": n_batches,
        "n_graphs": n_graphs,
        "wall_s": wall,
        "one_step_mse": mse,
    }


def _ms(x: float, n: int) -> float:
    return 1000.0 * x / max(n, 1)


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / max(whole, 1e-12)


def print_report(label: str, one_step: dict, totals: BucketTotals, per_run: dict[str, BucketTotals], rom: float):
    n = totals.n_steps
    wall = totals.wall
    print(f"\n{'=' * 72}")
    print(f"CONFIG: {label}")
    print(f"rollout_mse={rom:.6e}  rollout_steps={n}")
    print(f"{'=' * 72}")
    print("\n1) one_step full val")
    print(f"   n_batches={one_step['n_batches']}  n_graphs={one_step['n_graphs']}")
    print(f"   wall_s={one_step['wall_s']:.4f}")
    print(f"   one_step_mse={one_step['one_step_mse']:.6e}")

    print("\n2) rollout mean ms/step")
    print(f"   model_forward     {_ms(totals.model_forward, n):8.3f} ms/step")
    print(f"   air_velocity      {_ms(totals.air_velocity, n):8.3f} ms/step")
    print(f"   graph_construction{_ms(totals.graph_construction, n):8.3f} ms/step")
    print(f"   python_overhead   {_ms(totals.python_overhead, n):8.3f} ms/step")
    print(f"   TOTAL             {_ms(wall, n):8.3f} ms/step  (wall {wall:.3f} s)")

    print("\n   per-run totals (s):")
    for rid, rt in per_run.items():
        if rt.n_steps == 0:
            continue
        print(
            f"     {rid}: steps={rt.n_steps} wall={rt.wall:.3f}s "
            f"model={rt.model_forward:.3f} air={rt.air_velocity:.3f} "
            f"graph={rt.graph_construction:.3f} overhead={rt.python_overhead:.3f}"
        )

    print("\n3) fraction of rollout wall time")
    for name, val in [
        ("model_forward", totals.model_forward),
        ("air_velocity", totals.air_velocity),
        ("graph_construction", totals.graph_construction),
        ("python_overhead", totals.python_overhead),
    ]:
        print(f"   {name:20s} {_pct(val, wall):6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        action="append",
        type=Path,
        default=None,
        help="YAML config (repeatable); default: gns_baseline + gns_local",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--rollout-k", type=int, default=20)
    ap.add_argument("--max-runs", type=int, default=8)
    args = ap.parse_args()

    configs = args.config or [
        Path("ml/configs/gns_baseline.yaml"),
        Path("ml/configs/gns_local.yaml"),
    ]
    device = _device(args.device)
    print(f"device={device} rollout_K={args.rollout_k} max_runs={args.max_runs}")

    for cfg_path in configs:
        cfg = load_yaml(_resolve(cfg_path))
        seed = int(cfg.get("train", {}).get("seed", 42))
        _set_seed(seed)
        model, store, splits, val_loader, dcfg, tcfg = _build_model_and_val(cfg, device, seed)
        label = f"{cfg_path} ({cfg['model']['name']})"
        print(f"\n>>> {label}  val_runs={len(splits['val'])}")

        one_step = profile_one_step_pass(model, val_loader, device, store.norm)
        rom, totals, per_run = profile_rollout_mse(
            model,
            store,
            splits["val"],
            device,
            store.norm,
            K=args.rollout_k,
            C=dcfg.get("C", 5),
            max_runs=args.max_runs,
            node_features=dcfg.get("node_features"),
            globals_fields=dcfg.get("globals_fields"),
            edge_sets=dcfg.get("edge_sets"),
            radius=dcfg.get("radius", 0.1),
            normalize_features=dcfg.get("normalize_features", True),
        )
        print_report(label, one_step, totals, per_run, rom)


if __name__ == "__main__":
    main()
