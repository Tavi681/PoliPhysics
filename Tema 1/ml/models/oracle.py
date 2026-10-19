"""Oracle accelerations = training targets (velocity finite difference + global a-norm)."""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from ml.data.frames import DT
from ml.data.normalize import normalize
from ml.models.base import SurrogateModel, register_model


@register_model("oracle_gt_accel")
class OracleGroundTruthAccel(SurrogateModel):
    """Return the exact training-pipeline target for successive GT frames.

    At call ``k`` (0-based after ``reset``), returns
    ``normalize((v[t+1]-v[t])/dt)`` with ``t = C + k``, using the same global
    ``a_mean`` / ``a_std`` as training (not per-type). Ignores input features.
    """

    def __init__(self, rec: dict, *, C: int = 5, a_mean=None, a_std=None, dt: float = DT):
        super().__init__()
        self.rec = rec
        self.C = int(C)
        self.dt = float(dt)
        self.a_mean = np.asarray(
            a_mean if a_mean is not None else rec.get("_a_mean", [0, 0, 0]),
            dtype=np.float32,
        )
        self.a_std = np.asarray(
            a_std if a_std is not None else rec.get("_a_std", [1, 1, 1]),
            dtype=np.float32,
        )
        self._k = 0
        self.noise_std_by_type: np.ndarray | None = None  # normalized units, shape (3,) spider/interior/tip
        self.noise_std_spider_adj: float | None = None  # override for nodes adjacent to spider

    def reset(self) -> None:
        self._k = 0

    def required_edges(self) -> list[str]:
        return ["material"]

    def _spider_adj_mask(self) -> np.ndarray:
        edges = np.asarray(self.rec["edges"], dtype=np.int64)
        n = int(self.rec["node_type"].shape[0])
        adj = np.zeros(n, dtype=bool)
        for i, j in edges:
            if i == 0:
                adj[int(j)] = True
            if j == 0:
                adj[int(i)] = True
        return adj

    def _target_norm(self, t: int) -> np.ndarray:
        v = self.rec["v"]
        a = (v[t + 1].astype(np.float32) - v[t].astype(np.float32)) / np.float32(self.dt)
        return normalize(a, self.a_mean, self.a_std)

    def forward(self, batch: Data | Batch) -> torch.Tensor:
        t = self.C + self._k
        self._k += 1
        if t + 1 >= int(self.rec["v"].shape[0]):
            # Past recorded trajectory: zero accel in normalized space.
            n = int(batch.num_nodes) if not isinstance(batch.num_nodes, torch.Tensor) else int(batch.x.size(0))
            out = torch.zeros(n, 3, device=batch.x.device, dtype=batch.x.dtype)
            return out
        y = self._target_norm(t)
        if self.noise_std_by_type is not None:
            nt = self.rec["node_type"].astype(np.int64)
            scale = np.asarray(self.noise_std_by_type, dtype=np.float32)
            if scale.ndim == 1:
                eps = np.random.randn(*y.shape).astype(np.float32) * scale[np.clip(nt, 0, len(scale) - 1), None]
            else:
                eps = np.random.randn(*y.shape).astype(np.float32) * scale[np.clip(nt, 0, 2)]
            if self.noise_std_spider_adj is not None:
                adj = self._spider_adj_mask()
                eps[adj] = np.random.randn(int(adj.sum()), 3).astype(np.float32) * float(
                    self.noise_std_spider_adj
                )
            y = y + eps
        return torch.from_numpy(y).to(device=batch.x.device, dtype=batch.x.dtype)
