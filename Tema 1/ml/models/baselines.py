"""Non-learned baselines for paper tables."""
from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from ml.models.base import SurrogateModel, register_model


@register_model("constant_velocity")
class ConstantVelocity(SurrogateModel):
    """a = 0 (normalized space: we emit zeros; denorm gives a=mean if mean≠0 —
    training pipeline applies denorm of model output, so return normalized zero:
    (0 - mean)/std = -mean/std)."""

    def __init__(self, a_mean=None, a_std=None, **kwargs):
        super().__init__()
        # register buffers so denorm inverse is consistent: predict physical a=0
        if a_mean is None:
            a_mean = torch.zeros(3)
        if a_std is None:
            a_std = torch.ones(3)
        self.register_buffer("a_mean", torch.as_tensor(a_mean, dtype=torch.float32).view(1, 3))
        self.register_buffer("a_std", torch.as_tensor(a_std, dtype=torch.float32).view(1, 3))

    def required_edges(self) -> list[str]:
        return []

    def forward(self, batch: Data | Batch) -> torch.Tensor:
        n = batch.num_nodes if isinstance(batch.num_nodes, int) else int(batch.x.size(0))
        # physical a = 0 → normalized
        zeros = torch.zeros(n, 3, device=batch.x.device, dtype=batch.x.dtype)
        return (zeros - self.a_mean) / self.a_std


@register_model("follow_air")
class FollowAir(SurrogateModel):
    """v_{t+1} = u at the node; spider vertical component settles at -w_s.

    w_s is passed via batch.w_s (per-graph) or computed from globals if present.
    Emits normalized a_t = (v_{t+1} - v_t) / dt.
    """

    def __init__(self, a_mean=None, a_std=None, dt: float = 0.01, **kwargs):
        super().__init__()
        self.dt = float(dt)
        if a_mean is None:
            a_mean = torch.zeros(3)
        if a_std is None:
            a_std = torch.ones(3)
        self.register_buffer("a_mean", torch.as_tensor(a_mean, dtype=torch.float32).view(1, 3))
        self.register_buffer("a_std", torch.as_tensor(a_std, dtype=torch.float32).view(1, 3))

    def required_edges(self) -> list[str]:
        return []

    def forward(self, batch: Data | Batch) -> torch.Tensor:
        v_t = batch.v_t
        u = batch.u_t
        v_next = u.clone()
        # spider nodes: node_type == 0
        spider = batch.node_type == 0
        if hasattr(batch, "w_s"):
            # w_s may be per-graph; map to nodes via batch.batch
            if hasattr(batch, "batch") and batch.w_s.numel() > 1:
                ws = batch.w_s[batch.batch]
            else:
                ws = batch.w_s.reshape(-1)[0].expand(v_next.size(0))
            v_next[spider, 2] = -ws[spider] if ws.shape[0] == v_next.shape[0] else -float(batch.w_s.reshape(-1)[0])
        else:
            v_next[spider, 2] = 0.0
        a = (v_next - v_t) / self.dt
        return (a - self.a_mean) / self.a_std
