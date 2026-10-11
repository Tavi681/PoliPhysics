"""Edge-set constructors and relative geometric features for GNS / MeshGraphNet."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ml.data.graph import all_tips_edges, material_edges, radius_cross_thread


def interaction_edges(
    pos: torch.Tensor,
    thread_id: torch.Tensor,
    node_type: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radius cross-thread + all-tips; returns (edge_index [2,E], type_flag [E]).

    type_flag: 0 = radius, 1 = tips. Tips edges that already appear as radius keep flag 1.
    """
    er = radius_cross_thread(pos, thread_id, radius)
    et = all_tips_edges(node_type)
    if er.numel() == 0 and et.numel() == 0:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.float32)
    parts_i, parts_f = [], []
    if er.numel():
        parts_i.append(er)
        parts_f.append(torch.zeros(er.shape[1], dtype=torch.float32))
    if et.numel():
        parts_i.append(et)
        parts_f.append(torch.ones(et.shape[1], dtype=torch.float32))
    ei = torch.cat(parts_i, dim=1)
    flags = torch.cat(parts_f, dim=0)
    return ei, flags


def build_edge_sets_gns(
    names: Iterable[str],
    *,
    edges_material: np.ndarray,
    pos: torch.Tensor,
    thread_id: torch.Tensor,
    node_type: torch.Tensor,
    radius: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Like build_edge_sets, plus combined ``interaction`` set."""
    out: dict[str, torch.Tensor] = {}
    flags: dict[str, torch.Tensor] = {}
    for name in names:
        if name == "material":
            out[name] = material_edges(edges_material, thread_id)
        elif name == "interaction":
            ei, fl = interaction_edges(pos, thread_id, node_type, radius)
            out[name] = ei
            flags[name] = fl
        elif name == "radius":
            out[name] = radius_cross_thread(pos, thread_id, radius)
        elif name == "all_tips":
            out[name] = all_tips_edges(node_type)
        else:
            raise KeyError(f"unknown edge set {name}")
    out["_interaction_flags"] = flags.get("interaction", torch.zeros(0))
    return out


def material_edge_attr(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    l0: float,
) -> torch.Tensor:
    """[dx(3), |dx|, |dx|/l0] for each directed edge."""
    if edge_index.numel() == 0:
        return torch.zeros(0, 5, dtype=pos.dtype, device=pos.device)
    src, dst = edge_index[0], edge_index[1]
    dx = pos[src] - pos[dst]
    dist = torch.linalg.norm(dx, dim=-1, keepdim=True).clamp_min(1e-12)
    stretch = dist / max(float(l0), 1e-12)
    return torch.cat([dx, dist, stretch], dim=-1)


def interaction_edge_attr(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    charge: torch.Tensor,
    type_flag: torch.Tensor,
    *,
    inv_r2_clip: float = 1e6,
) -> torch.Tensor:
    """[dx(3), |dx|, clip(1/r^2), q_i*q_j, type_flag]."""
    if edge_index.numel() == 0:
        return torch.zeros(0, 7, dtype=pos.dtype, device=pos.device)
    src, dst = edge_index[0], edge_index[1]
    dx = pos[src] - pos[dst]
    dist = torch.linalg.norm(dx, dim=-1, keepdim=True).clamp_min(1e-12)
    inv_r2 = (1.0 / (dist * dist)).clamp(max=inv_r2_clip)
    qq = (charge[src] * charge[dst]).unsqueeze(-1)
    fl = type_flag.to(dtype=pos.dtype, device=pos.device).view(-1, 1)
    return torch.cat([dx, dist, inv_r2, qq, fl], dim=-1)
