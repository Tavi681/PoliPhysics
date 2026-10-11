"""PyG dataset: sample = (run, t) with velocity history and target a_t."""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data

from ml.data.air import u_next_frame
from ml.data.edge_features import (
    build_edge_sets_gns,
    interaction_edge_attr,
    material_edge_attr,
)
from ml.data.frames import DT
from ml.data.graph import build_edge_sets
from ml.data.normalize import load_norm_stats, normalize


class BallooningRunStore:
    """Shard-backed run store with an LRU of *raw* consolidated blobs.

    Consolidated shards stay memory-mapped; individual runs are sliced on
    ``get()`` (no full-shard expand into RAM). Legacy list shards are still
    expanded once into a rid→record map.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        cache_shards: int = 4,
        preload: bool = False,
        mmap: bool = True,
    ):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.norm = load_norm_stats(self.data_dir / "norm_stats.json")
        self._rid_to_shard: dict[str, str] = {
            r["run_id"]: r["shard"] for r in self.manifest["records"]
        }
        self._cache_shards = max(1, int(cache_shards))
        self._mmap = bool(mmap)
        # sh -> ("consolidated", blob, rid->i) | ("legacy", rid->rec)
        self._shard_cache: OrderedDict[str, tuple] = OrderedDict()
        self.runs = _RunIdView(self)
        if preload:
            for sh in self.manifest["shards"]:
                self._ensure_shard(sh)

    def __contains__(self, run_id: str) -> bool:
        return run_id in self._rid_to_shard

    def get(self, run_id: str) -> dict:
        sh = self._rid_to_shard.get(run_id)
        if sh is None:
            raise KeyError(run_id)
        kind, payload, index = self._ensure_shard(sh)
        if kind == "legacy":
            return payload[run_id]
        from ml.data.shard_format import expand_one_run
        return expand_one_run(payload, index[run_id], copy=False)

    def build_frame_index(
        self, run_ids: Sequence[str], *, C: int
    ) -> tuple[list[tuple[str, int]], dict[str, int]]:
        """Build (run_id, t) index from window_ok only — no heavy array expand."""
        from ml.data.shard_format import iter_window_ok

        index: list[tuple[str, int]] = []
        run_sample_count: dict[str, int] = {}
        by_shard: dict[str, list[str]] = {}
        for rid in run_ids:
            if rid not in self._rid_to_shard:
                continue
            by_shard.setdefault(self._rid_to_shard[rid], []).append(rid)
        for sh in sorted(by_shard.keys()):
            kind, payload, _index = self._ensure_shard(sh)
            want_sh = set(by_shard[sh])
            if kind == "consolidated":
                for rid, wok, T in iter_window_ok(payload):
                    if rid not in want_sh:
                        continue
                    n_ok = 0
                    for t in range(C, T - 1):
                        if not bool(wok[t]):
                            continue
                        index.append((rid, t))
                        n_ok += 1
                    run_sample_count[rid] = n_ok
            else:
                for rid in by_shard[sh]:
                    rec = payload[rid]
                    wok = rec["window_ok"]
                    T = int(rec["v"].shape[0])
                    n_ok = 0
                    for t in range(C, T - 1):
                        if not bool(wok[t]):
                            continue
                        index.append((rid, t))
                        n_ok += 1
                    run_sample_count[rid] = n_ok
        return index, run_sample_count

    def _ensure_shard(self, sh: str) -> tuple:
        if sh in self._shard_cache:
            self._shard_cache.move_to_end(sh)
            return self._shard_cache[sh]
        from ml.data.shard_format import FORMAT, load_shard_raw

        obj = load_shard_raw(self.data_dir / sh, mmap=self._mmap)
        if isinstance(obj, dict) and obj.get("format") == FORMAT:  # consolidated_v1
            rid_i = {rid: i for i, rid in enumerate(obj["run_ids"])}
            entry: tuple = ("consolidated", obj, rid_i)
        elif isinstance(obj, list):
            pack = {rec["run_id"]: rec for rec in obj}
            entry = ("legacy", pack, None)
        else:
            raise TypeError(f"unknown shard type {type(obj)} at {sh}")
        self._shard_cache[sh] = entry
        self._shard_cache.move_to_end(sh)
        while len(self._shard_cache) > self._cache_shards:
            self._shard_cache.popitem(last=False)
        return entry

    # backward-compat alias used by older tests
    def _load_shard(self, sh: str) -> dict[str, dict]:
        kind, payload, index = self._ensure_shard(sh)
        if kind == "legacy":
            return payload
        from ml.data.shard_format import expand_one_run
        return {rid: expand_one_run(payload, i, copy=True) for rid, i in index.items()}


class _RunIdView:
    def __init__(self, store: BallooningRunStore):
        self._store = store

    def __contains__(self, rid: str) -> bool:
        return rid in self._store._rid_to_shard

    def __iter__(self):
        return iter(self._store._rid_to_shard)

    def __len__(self) -> int:
        return len(self._store._rid_to_shard)

    def __getitem__(self, rid: str) -> dict:
        return self._store.get(rid)

    def keys(self):
        return self._store._rid_to_shard.keys()

    def get(self, rid: str, default=None):
        if rid not in self._store._rid_to_shard:
            return default
        return self._store.get(rid)


def _norm_vec(x: np.ndarray, stats: dict | None) -> np.ndarray:
    if stats is None:
        return x.astype(np.float32)
    return normalize(x.astype(np.float32), stats["mean"], stats["std"])


class BallooningDataset(Dataset):
    """One item = (run_id, t) with history of C velocities.

    Target a_t = (v_{t+1} - v_t) / dt  (velocity-based; check B).
    Positions are spider-relative (no absolute xy in node features by default).
    """

    def __init__(
        self,
        store: BallooningRunStore,
        run_ids: Sequence[str],
        *,
        C: int = 5,
        history_u: bool = False,
        node_features: Sequence[str] | None = None,
        globals_fields: Sequence[str] | None = None,
        edge_sets: Sequence[str] | None = None,
        radius: float = 0.1,
        noise_std: float = 0.0,
        noise_std_by_type: np.ndarray | Sequence[float] | None = None,
        noise_std_components: np.ndarray | Sequence[float] | None = None,
        sample_weight: str = "per_run",
        normalize_features: bool = True,
        normalize_target: bool = True,
        rotate_augment: bool = False,
        seed: int = 0,
    ):
        self.store = store
        self.C = int(C)
        self.history_u = history_u
        self.node_features = list(
            node_features or ["v_hist", "u", "charge", "node_type"]
        )
        self.globals_fields = list(globals_fields or [])
        self.edge_sets = list(edge_sets or ["material"])
        self.radius = float(radius)
        self.noise_std = float(noise_std)
        self.noise_std_by_type = (
            None
            if noise_std_by_type is None
            else np.asarray(noise_std_by_type, dtype=np.float32).reshape(-1)
        )
        self.noise_std_components = (
            None
            if noise_std_components is None
            else np.asarray(noise_std_components, dtype=np.float32).reshape(-1)
        )
        self.normalize_features = bool(normalize_features)
        self.normalize_target = normalize_target
        self.rotate_augment = bool(rotate_augment)
        self.rng = np.random.default_rng(seed)
        self.dt = float(store.norm.get("dt", DT))
        self.norm = store.norm

        # Index via window_ok metadata only (no per-run expand of v/x/u).
        self.index, self._run_sample_count = store.build_frame_index(
            run_ids, C=self.C
        )
        self.sample_weight = sample_weight

    def __len__(self) -> int:
        return len(self.index)

    def weights(self) -> torch.Tensor:
        w = torch.ones(len(self.index), dtype=torch.double)
        if self.sample_weight == "per_frame":
            return w
        for i, (rid, _) in enumerate(self.index):
            n = max(self._run_sample_count.get(rid, 1), 1)
            w[i] = 1.0 / n
        return w

    def _rotate_yaw(self, arr: np.ndarray, c: float, s: float) -> np.ndarray:
        """Rotate ...x3 about z by angle with cos c, sin s."""
        out = arr.copy()
        x, y = arr[..., 0], arr[..., 1]
        out[..., 0] = c * x - s * y
        out[..., 1] = s * x + c * y
        return out

    def _build_node_features(
        self,
        rec: dict,
        t: int,
        v_hist: np.ndarray,
        u_t: np.ndarray,
        u_next: np.ndarray | None,
        spider: np.ndarray,
    ) -> torch.Tensor:
        feats = []
        n = rec["v"].shape[1]
        g = rec["globals"]
        for name in self.node_features:
            if name == "v_hist":
                vh = v_hist.reshape(n, -1)
                if self.normalize_features and "v" in self.norm:
                    # normalize each 3-vector block with v stats
                    blocks = []
                    for k in range(self.C):
                        blocks.append(_norm_vec(v_hist[k], self.norm["v"]))
                    vh = np.concatenate(blocks, axis=-1)
                feats.append(vh)
            elif name == "u":
                feats.append(_norm_vec(u_t, self.norm.get("u") if self.normalize_features else None))
            elif name == "u_next":
                if u_next is None:
                    raise KeyError("u_next requested but unavailable")
                feats.append(
                    _norm_vec(u_next, self.norm.get("u") if self.normalize_features else None)
                )
            elif name == "u_hist":
                # history u_{t-C+1..t-1} (exclude u_t; u_t is separate)
                u_h = rec["u"][t - self.C + 1 : t]  # (C-1,n,3)
                if u_h.shape[0] == 0:
                    feats.append(np.zeros((n, 0), dtype=np.float32))
                else:
                    blocks = []
                    for k in range(u_h.shape[0]):
                        blocks.append(
                            _norm_vec(u_h[k], self.norm.get("u") if self.normalize_features else None)
                        )
                    feats.append(np.concatenate(blocks, axis=-1))
            elif name == "charge":
                feats.append(rec["charge"][:, None].astype(np.float32))
            elif name == "node_type":
                # legacy one-hot for stub_gnn; MeshGraphNet uses embedding instead
                nt = rec["node_type"].astype(np.int64)
                feats.append(np.eye(3, dtype=np.float32)[nt])
            elif name == "x_rel":
                feats.append(
                    _norm_vec(rec["x_rel"][t], self.norm.get("x_rel") if self.normalize_features else None)
                )
            elif name == "spider_z":
                feats.append(np.full((n, 1), float(spider[2]), dtype=np.float32))
            else:
                raise KeyError(name)

        # globals broadcast
        for gf in self.globals_fields:
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
                raise KeyError(f"unknown globals field {gf}")
            feats.append(np.full((n, 1), val, dtype=np.float32))

        if not feats:
            return torch.zeros(n, 0, dtype=torch.float32)
        x = np.concatenate(feats, axis=-1).astype(np.float32)
        return torch.from_numpy(x)

    def __getitem__(self, i: int) -> Data:
        rid, t = self.index[i]
        rec = self.store.get(rid)
        dt = self.dt
        v = rec["v"]
        v_hist = v[t - self.C + 1 : t + 1].copy()
        v_t = v[t].copy()
        v_tp1 = v[t + 1].copy()
        a = (v_tp1 - v_t) / dt

        if (
            self.noise_std > 0
            or (self.noise_std_by_type is not None and np.any(self.noise_std_by_type > 0))
            or (self.noise_std_components is not None and np.any(self.noise_std_components > 0))
        ):
            # Random-walk noise over C velocity steps (Sanchez-Gonzalez et al.).
            # Std at the most recent velocity equals the configured velocity scale
            # (default: k * a_std * dt per component). Increments are /sqrt(C).
            #
            # Target convention (implemented): clean next velocity, noisy current:
            #   a = (v_{t+1}^{gt} - v_t^{noisy}) / dt
            # so the model must undo the injected noise. Next-state is NOT shifted.
            eps = self.rng.normal(0.0, 1.0, size=v_hist.shape).astype(np.float32)
            if self.noise_std_components is not None:
                eps = eps * self.noise_std_components.reshape(1, 1, -1)
            elif self.noise_std_by_type is not None:
                nt = rec["node_type"].astype(np.int64)
                scale = self.noise_std_by_type[np.clip(nt, 0, len(self.noise_std_by_type) - 1)]
                eps = eps * scale[None, :, None]
            else:
                eps = eps * np.float32(self.noise_std)
            eps = eps * np.float32(1.0 / np.sqrt(max(self.C, 1)))
            noise_rw = np.cumsum(eps, axis=0)
            v_hist = v_hist + noise_rw
            a = (v_tp1 - v_hist[-1]) / dt

        x_rel = rec["x_rel"][t].astype(np.float32).copy()
        u_t = rec["u"][t].astype(np.float32).copy()
        spider = rec["spider_abs"][t].astype(np.float64).copy()

        # optional yaw augmentation (about vertical)
        if self.rotate_augment:
            ang = float(self.rng.uniform(0.0, 2.0 * np.pi))
            c, s = float(np.cos(ang)), float(np.sin(ang))
            x_rel = self._rotate_yaw(x_rel, c, s)
            v_hist = self._rotate_yaw(v_hist, c, s)
            u_t = self._rotate_yaw(u_t, c, s)
            a = self._rotate_yaw(a.astype(np.float32), c, s)
            spider_xy = spider.copy()
            spider_xy[0], spider_xy[1] = c * spider[0] - s * spider[1], s * spider[0] + c * spider[1]
            spider = spider_xy

        need_unext = "u_next" in self.node_features
        u_next = u_next_frame(rec, t) if need_unext else None
        if need_unext and self.rotate_augment:
            u_next = self._rotate_yaw(u_next, c, s)

        pos = torch.from_numpy(x_rel)
        pos_abs = pos + torch.from_numpy(spider.astype(np.float32))

        node_type = torch.from_numpy(rec["node_type"].astype(np.int64))
        thread_id = torch.from_numpy(rec["thread_id"].astype(np.int64))
        charge_t = torch.from_numpy(rec["charge"].astype(np.float32))

        use_gns_edges = any(s in ("interaction",) for s in self.edge_sets)
        if use_gns_edges or "interaction" in self.edge_sets:
            edge_dict = build_edge_sets_gns(
                self.edge_sets,
                edges_material=rec["edges"],
                pos=pos_abs,
                thread_id=thread_id,
                node_type=node_type,
                radius=self.radius,
            )
            inter_flags = edge_dict.pop("_interaction_flags")
        else:
            edge_dict = build_edge_sets(
                self.edge_sets,
                edges_material=rec["edges"],
                pos=pos_abs,
                thread_id=thread_id,
                node_type=node_type,
                radius=self.radius,
            )
            inter_flags = torch.zeros(0)

        g = rec["globals"]
        l0 = float(g["L"]) / max(int(g["N_t"]), 1)

        feat = self._build_node_features(rec, t, v_hist, u_t, u_next, spider)
        target = torch.from_numpy(a.astype(np.float32))
        if self.normalize_target:
            target = torch.from_numpy(
                normalize(a.astype(np.float32), self.norm["a"]["mean"], self.norm["a"]["std"])
            )

        data = Data(
            x=feat,
            pos=pos,
            y=target,
            node_type=node_type,
            thread_id=thread_id,
            charge=charge_t,
            edge_index=edge_dict.get("material", torch.zeros(2, 0, dtype=torch.long)),
            num_nodes=int(pos.shape[0]),
        )
        for name, eidx in edge_dict.items():
            setattr(data, f"edge_index_{name}", eidx)
            if name == "material":
                data.edge_attr_material = material_edge_attr(pos, eidx, l0=l0)
            elif name == "interaction":
                data.edge_attr_interaction = interaction_edge_attr(
                    pos, eidx, charge_t, inter_flags
                )

        data.spider_abs = torch.from_numpy(spider.astype(np.float32))
        data.spider_z = torch.tensor([float(spider[2])], dtype=torch.float32)
        data.run_N = torch.tensor([g["N"]], dtype=torch.long)
        data.sigma_w = torch.tensor([g["sigma_w"]], dtype=torch.float32)
        data.U_h = torch.tensor([g["U_h"]], dtype=torch.float32)
        data.seed = torch.tensor([g["seed"]], dtype=torch.long)
        data.h = torch.tensor([g["h"]], dtype=torch.float32)
        data.z0 = torch.tensor([g["z0"]], dtype=torch.float32)
        data.t_index = torch.tensor([t], dtype=torch.long)
        data.run_id = rid
        data.v_t = torch.from_numpy(v_hist[-1].astype(np.float32))
        data.u_t = torch.from_numpy(u_t.astype(np.float32))
        if u_next is not None:
            data.u_next = torch.from_numpy(u_next.astype(np.float32))
        return data


def make_loader(
    dataset: BallooningDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
    prefetch_factor: int | None = None,
    shard_aware: bool = True,
    seed: int = 0,
):
    from torch_geometric.loader import DataLoader

    from ml.data.shard_sampler import ShardAwareSampler

    # mmap shard tensors use FD-backed storage; pin_memory + workers can break ancdata IPC.
    pin = str(device).startswith("cuda") and not getattr(
        getattr(dataset, "store", None), "_mmap", False
    )
    sampler = None
    if shuffle:
        if shard_aware and hasattr(dataset.store, "_rid_to_shard"):
            w = dataset.weights() if dataset.sample_weight == "per_run" else None
            sampler = ShardAwareSampler(
                dataset.index,
                dataset.store._rid_to_shard,
                weights=w,
                seed=seed,
            )
            shuffle = False
        elif dataset.sample_weight == "per_run":
            sampler = WeightedRandomSampler(
                weights=dataset.weights(),
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor if prefetch_factor is not None else 4)
    return DataLoader(dataset, **kwargs)
