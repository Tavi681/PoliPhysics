"""Graph edge constructors for ballooning surrogates."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _as_coo(edges: np.ndarray | torch.Tensor) -> torch.Tensor:
    e = torch.as_tensor(edges, dtype=torch.long)
    if e.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long)
    if e.dim() == 2 and e.shape[1] == 2:
        return e.T.contiguous()
    return e.contiguous()


def material_edges(
    edges: np.ndarray | torch.Tensor,
    thread_id: np.ndarray | torch.Tensor | None = None,
    bidirectional: bool = True,
) -> torch.Tensor:
    """Material edges along threads (simulator Topology.edges already includes
    spider→first-node of each thread). Both directions by default.
    """
    fwd = _as_coo(edges)
    if not bidirectional or fwd.numel() == 0:
        return fwd
    return torch.cat([fwd, fwd.flip(0)], dim=1)


def radius_cross_thread(
    pos: torch.Tensor,
    thread_id: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Radius graph between nodes on different threads (incl. spider)."""
    d = torch.cdist(pos, pos)
    tid = thread_id.view(-1)
    diff_thread = tid[:, None] != tid[None, :]
    mask = (d <= radius) & diff_thread & (d > 0)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


def all_tips_edges(node_type: torch.Tensor) -> torch.Tensor:
    tips = (node_type == 2).nonzero(as_tuple=True)[0]
    if len(tips) < 2:
        return torch.zeros(2, 0, dtype=torch.long)
    src, dst = torch.meshgrid(tips, tips, indexing="ij")
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def fully_connected(n: int) -> torch.Tensor:
    idx = torch.arange(n)
    src, dst = torch.meshgrid(idx, idx, indexing="ij")
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def build_edge_sets(
    names: Iterable[str],
    *,
    edges_material: np.ndarray,
    pos: torch.Tensor,
    thread_id: torch.Tensor,
    node_type: torch.Tensor,
    radius: float = 0.1,
) -> dict[str, torch.Tensor]:
    out = {}
    for name in names:
        if name == "material":
            out[name] = material_edges(edges_material, thread_id)
        elif name == "radius":
            out[name] = radius_cross_thread(pos, thread_id, radius)
        elif name == "all_tips":
            out[name] = all_tips_edges(node_type)
        elif name == "fully_connected":
            out[name] = fully_connected(int(pos.shape[0]))
        else:
            raise KeyError(f"unknown edge set {name}")
    return out
