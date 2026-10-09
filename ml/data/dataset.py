"""PyG dataset: sample = (run, t) with velocity history and target a_t."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data

from ml.data.frames import DT
from ml.data.graph import build_edge_sets
from ml.data.normalize import load_norm_stats, normalize


class BallooningRunStore:
    """Loads sharded .pt packs into memory (fine for local subset)."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "manifest.json") as f:
            self.manifest = json.load(f)
        self.norm = load_norm_stats(self.data_dir / "norm_stats.json")
        self.runs: dict[str, dict] = {}
        for sh in self.manifest["shards"]:
            pack = torch.load(self.data_dir / sh, weights_only=False)
            for rec in pack:
                self.runs[rec["run_id"]] = rec

    def get(self, run_id: str) -> dict:
        return self.runs[run_id]


class BallooningDataset(Dataset):
    """One item = (run_id, t) with history of C velocities.

    Target a_t = (v_{t+1} - v_t) / dt  (velocity-based; check B).
    Positions are spider-relative (+ spider absolute z / xyz in globals).
    """

    def __init__(
        self,
        store: BallooningRunStore,
        run_ids: Sequence[str],
        *,
        C: int = 5,
        history_u: bool = False,
        node_features: Sequence[str] | None = None,
        edge_sets: Sequence[str] | None = None,
        radius: float = 0.1,
        noise_std: float = 0.0,
        sample_weight: str = "per_run",
        normalize_target: bool = True,
        seed: int = 0,
    ):
        self.store = store
        self.C = int(C)
        self.history_u = history_u
        self.node_features = list(node_features or ["v_hist", "u", "charge", "node_type"])
        self.edge_sets = list(edge_sets or ["material"])
        self.radius = float(radius)
        self.noise_std = float(noise_std)
        self.normalize_target = normalize_target
        self.rng = np.random.default_rng(seed)
        self.dt = float(store.norm.get("dt", DT))
        self.norm = store.norm

        self.index: list[tuple[str, int]] = []
        self._run_sample_count: dict[str, int] = {}
        for rid in run_ids:
            if rid not in store.runs:
                continue
            rec = store.runs[rid]
            ok = rec["window_ok"]
            # need t>=C and t+1 < T; also extend window_ok for this C
            T = rec["v"].shape[0]
            n_ok = 0
            for t in range(self.C, T - 1):
                if not bool(ok[t]):
                    continue
                # ensure history frames exist (always true if t>=C)
                self.index.append((rid, t))
                n_ok += 1
            self._run_sample_count[rid] = n_ok

        self.sample_weight = sample_weight

    def __len__(self) -> int:
        return len(self.index)

    def weights(self) -> torch.Tensor:
        """Sampler weights: equal per run (default) or uniform per frame."""
        w = torch.ones(len(self.index), dtype=torch.double)
        if self.sample_weight == "per_frame":
            return w
        # per_run: weight ∝ 1 / n_samples_in_run
        for i, (rid, _) in enumerate(self.index):
            n = max(self._run_sample_count.get(rid, 1), 1)
            w[i] = 1.0 / n
        return w

    def _build_node_features(
        self,
        rec: dict,
        t: int,
        v_hist: np.ndarray,
        u_t: np.ndarray,
    ) -> torch.Tensor:
        feats = []
        n = rec["v"].shape[1]
        for name in self.node_features:
            if name == "v_hist":
                feats.append(v_hist.reshape(n, -1))  # (n, 3C)
            elif name == "u":
                feats.append(u_t)
            elif name == "u_hist":
                u_hist = rec["u"][t - self.C + 1 : t + 1]  # (C,n,3)
                feats.append(np.transpose(u_hist, (1, 0, 2)).reshape(n, -1))
            elif name == "charge":
                feats.append(rec["charge"][:, None])
            elif name == "node_type":
                # one-hot 3
                nt = rec["node_type"].astype(np.int64)
                oh = np.eye(3, dtype=np.float32)[nt]
                feats.append(oh)
            elif name == "x_rel":
                feats.append(rec["x_rel"][t])
            else:
                raise KeyError(name)
        x = np.concatenate(feats, axis=-1).astype(np.float32)
        return torch.from_numpy(x)

    def __getitem__(self, i: int) -> Data:
        rid, t = self.index[i]
        rec = self.store.get(rid)
        dt = self.dt
        v = rec["v"]  # (T,n,3)
        # history velocities v_{t-C+1} .. v_t  → shape (C,n,3)
        v_hist = v[t - self.C + 1 : t + 1].copy()
        v_t = v[t].copy()
        v_tp1 = v[t + 1].copy()
        a = (v_tp1 - v_t) / dt

        # Sanchez-Gonzalez random-walk noise on velocity history
        if self.noise_std > 0:
            # noise on the C history steps; correct target consistently:
            # observed v'_t = v_t + sum eps, a' = (v_{t+1} - v'_t)/dt
            eps = self.rng.normal(0.0, self.noise_std, size=v_hist.shape).astype(np.float32)
            # random walk along history axis
            noise_rw = np.cumsum(eps, axis=0)
            v_hist = v_hist + noise_rw
            v_t_noisy = v_hist[-1]
            a = (v_tp1 - v_t_noisy) / dt

        x_rel = rec["x_rel"][t]
        u_t = rec["u"][t]
        spider = rec["spider_abs"][t]

        pos = torch.from_numpy(x_rel.astype(np.float32))
        # absolute positions for radius graph: relative + spider
        pos_abs = pos + torch.from_numpy(spider.astype(np.float32))

        node_type = torch.from_numpy(rec["node_type"].astype(np.int64))
        thread_id = torch.from_numpy(rec["thread_id"].astype(np.int64))
        edge_dict = build_edge_sets(
            self.edge_sets,
            edges_material=rec["edges"],
            pos=pos_abs,
            thread_id=thread_id,
            node_type=node_type,
            radius=self.radius,
        )

        feat = self._build_node_features(rec, t, v_hist, u_t)
        target = torch.from_numpy(a.astype(np.float32))
        if self.normalize_target:
            target = torch.from_numpy(
                normalize(a.astype(np.float32), self.norm["a"]["mean"], self.norm["a"]["std"])
            )

        g = rec["globals"]
        data = Data(
            x=feat,
            pos=pos,
            y=target,
            node_type=node_type,
            thread_id=thread_id,
            edge_index=edge_dict.get("material", torch.zeros(2, 0, dtype=torch.long)),
            num_nodes=int(pos.shape[0]),
        )
        # attach extra edge sets
        for name, eidx in edge_dict.items():
            setattr(data, f"edge_index_{name}", eidx)
        # globals (broadcast per-graph; PyG batching uses data attributes)
        # float32 for MPS compatibility (absolute z still carried; shards keep float64)
        data.spider_abs = torch.from_numpy(spider.astype(np.float32))
        data.spider_z = torch.tensor([float(spider[2])], dtype=torch.float32)
        data.run_N = torch.tensor([g["N"]], dtype=torch.long)
        data.sigma_w = torch.tensor([g["sigma_w"]], dtype=torch.float32)
        data.U_h = torch.tensor([g["U_h"]], dtype=torch.float32)
        data.seed = torch.tensor([g["seed"]], dtype=torch.long)
        data.h = torch.tensor([g["h"]], dtype=torch.float32)
        data.z0 = torch.tensor([g["z0"]], dtype=torch.float32)
        data.t_index = torch.tensor([t], dtype=torch.long)
        data.run_id = rid  # string; careful in batching
        data.v_t = torch.from_numpy(v_hist[-1].astype(np.float32))
        data.u_t = torch.from_numpy(u_t.astype(np.float32))
        return data


def make_loader(
    dataset: BallooningDataset,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
):
    from torch_geometric.loader import DataLoader

    pin = device == "cuda"
    sampler = None
    if shuffle and dataset.sample_weight == "per_run":
        sampler = WeightedRandomSampler(
            weights=dataset.weights(),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
