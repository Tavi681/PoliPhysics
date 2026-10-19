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
    """Massless tracers: every node (spider included) takes the local air velocity.

    Paper definition: for each node k, ``v_k = u(x_k, t)`` in all three components,
    where ``u`` is evaluated at the current absolute positions and current time
    (``u(x^t, t)``; see ``full_rollout_surrogate``). No settling speed, no
    electrostatic drift, no special case for the spider.

    Emits normalized ``a_t = (v_{t+1} - v_t) / dt`` with ``v_{t+1} = u_t``.
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
        v_next = u
        a = (v_next - v_t) / self.dt
        return (a - self.a_mean) / self.a_std
