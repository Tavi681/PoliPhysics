"""Small encoder → 2 message-passing layers on material edges → decoder.

Edit this file (or copy it) when adding a learned architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing

from ml.models.base import SurrogateModel, register_model


class MaterialConv(MessagePassing):
    """Plain mean aggregation message passing (no torch_scatter extras)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.lin_msg = nn.Linear(2 * in_dim, out_dim)
        self.lin_upd = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x)
        return self.act(self.lin_upd(torch.cat([x, out], dim=-1)))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.act(self.lin_msg(torch.cat([x_i, x_j], dim=-1)))


@register_model("stub_gnn")
class StubGNN(SurrogateModel):
    def __init__(
        self,
        node_dim: int = 3 * 5 + 3 + 1 + 3,  # v_hist + u + charge + node_type oh
        hidden: int = 64,
        out_dim: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mp1 = MaterialConv(hidden, hidden)
        self.mp2 = MaterialConv(hidden, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def required_edges(self) -> list[str]:
        return ["material"]

    def forward(self, batch: Data | Batch) -> torch.Tensor:
        x = self.encoder(batch.x)
        e = batch.edge_index_material if hasattr(batch, "edge_index_material") else batch.edge_index
        x = self.mp1(x, e)
        x = self.mp2(x, e)
        return self.decoder(x)
