"""Classic GNS / MeshGraphNet baseline (Sanchez-Gonzalez 2020, Pfaff 2021).

Architecture is problem-agnostic; node/edge feature content is assembled by the
dataset / feature helpers and selected via config.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from ml.models.base import SurrogateModel, register_model


def _mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    *,
    n_hidden: int = 2,
    layernorm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    if layernorm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.shape[-1], device=src.device, dtype=src.dtype)
    if index.numel() == 0:
        return out
    return out.index_add(0, index, src)


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = _scatter_sum(src, index, dim_size)
    if index.numel() == 0:
        return out
    counts = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    counts.index_add_(0, index, torch.ones(index.shape[0], device=src.device, dtype=src.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(-1)


class _ProcessorStep(nn.Module):
    def __init__(
        self,
        latent: int,
        edge_sets: Sequence[str],
        *,
        mlp_hidden: int,
        mlp_depth: int,
        aggregation: str,
    ):
        super().__init__()
        self.edge_sets = list(edge_sets)
        self.aggregation = aggregation
        self.edge_mlps = nn.ModuleDict({
            s: _mlp(3 * latent, mlp_hidden, latent, n_hidden=mlp_depth, layernorm=True)
            for s in self.edge_sets
        })
        # node update sees [v, sum_e for each set]
        self.node_mlp = _mlp(
            latent * (1 + len(self.edge_sets)),
            mlp_hidden,
            latent,
            n_hidden=mlp_depth,
            layernorm=True,
        )

    def forward(
        self,
        v: torch.Tensor,
        edges: dict[str, torch.Tensor],
        edge_index: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n = v.shape[0]
        agg_fn = _scatter_sum if self.aggregation == "sum" else _scatter_mean
        new_edges: dict[str, torch.Tensor] = {}
        pooled: list[torch.Tensor] = []
        for s in self.edge_sets:
            e = edges[s]
            ei = edge_index[s]
            if ei.numel() == 0:
                e_p = e
                pooled.append(torch.zeros(n, v.shape[-1], device=v.device, dtype=v.dtype))
            else:
                src, dst = ei[0], ei[1]
                inp = torch.cat([e, v[src], v[dst]], dim=-1)
                e_p = e + self.edge_mlps[s](inp)
                pooled.append(agg_fn(e_p, dst, n))
            new_edges[s] = e_p
        v_p = v + self.node_mlp(torch.cat([v] + pooled, dim=-1))
        return v_p, new_edges


@register_model("gns_baseline")
@register_model("gns_local")
class MeshGraphNet(SurrogateModel):
    """Shared MeshGraphNet; edge sets / features come from the batch + config."""

    def __init__(
        self,
        *,
        node_dim: int = 0,
        edge_dims: dict[str, int] | None = None,
        edge_sets: Sequence[str] | None = None,
        latent: int = 128,
        n_mp_steps: int = 10,
        mlp_hidden: int | None = None,
        mlp_depth: int = 2,
        aggregation: str = "sum",
        node_type_embed_dim: int = 16,
        n_node_types: int = 3,
        out_dim: int = 3,
        # accepted / ignored aliases from older configs
        hidden: int | None = None,
        n_layers: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if hidden is not None:
            latent = int(hidden)
        if n_layers is not None:
            n_mp_steps = int(n_layers)
        self.latent = int(latent)
        self.n_mp_steps = int(n_mp_steps)
        self.aggregation = str(aggregation)
        self.mlp_hidden = int(mlp_hidden if mlp_hidden is not None else latent)
        self.mlp_depth = int(mlp_depth)
        self.edge_sets = list(edge_sets or ["material", "interaction"])
        self.edge_dims = dict(edge_dims or {s: 5 for s in self.edge_sets})
        # material default 5: dx(3)+dist+stretch; interaction 7: dx(3)+dist+inv_r2+qq+flag
        for s in self.edge_sets:
            self.edge_dims.setdefault(s, 5 if s == "material" else 7)

        self.node_type_emb = nn.Embedding(n_node_types, int(node_type_embed_dim))
        # continuous node features + embedding
        enc_in = int(node_dim) + int(node_type_embed_dim)
        self.node_encoder = _mlp(
            enc_in, self.mlp_hidden, self.latent, n_hidden=self.mlp_depth, layernorm=True
        )
        self.edge_encoders = nn.ModuleDict({
            s: _mlp(
                int(self.edge_dims[s]), self.mlp_hidden, self.latent,
                n_hidden=self.mlp_depth, layernorm=True,
            )
            for s in self.edge_sets
        })
        self.processor = nn.ModuleList([
            _ProcessorStep(
                self.latent, self.edge_sets,
                mlp_hidden=self.mlp_hidden, mlp_depth=self.mlp_depth,
                aggregation=self.aggregation,
            )
            for _ in range(self.n_mp_steps)
        ])
        self.decoder = _mlp(
            self.latent, self.mlp_hidden, out_dim, n_hidden=self.mlp_depth, layernorm=False
        )

    def required_edges(self) -> list[str]:
        return list(self.edge_sets)

    def forward(self, batch: Data | Batch) -> torch.Tensor:
        # continuous features in batch.x; node_type indices separate
        nt = batch.node_type.view(-1).long()
        x = torch.cat([batch.x, self.node_type_emb(nt)], dim=-1)
        v = self.node_encoder(x)

        edges: dict[str, torch.Tensor] = {}
        eidx: dict[str, torch.Tensor] = {}
        for s in self.edge_sets:
            ei = getattr(batch, f"edge_index_{s}", None)
            if ei is None and s == "material":
                ei = batch.edge_index
            if ei is None:
                ei = torch.zeros(2, 0, dtype=torch.long, device=v.device)
            eidx[s] = ei
            ef = getattr(batch, f"edge_attr_{s}", None)
            if ef is None:
                # zeros if attrs missing (should be provided by dataset)
                ef = torch.zeros(ei.shape[1], self.edge_dims[s], device=v.device, dtype=v.dtype)
            edges[s] = self.edge_encoders[s](ef)

        for step in self.processor:
            v, edges = step(v, edges, eidx)
        return self.decoder(v)
