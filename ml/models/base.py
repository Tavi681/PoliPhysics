"""Surrogate model interface + registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

MODEL_REGISTRY: dict[str, Callable[..., "SurrogateModel"]] = {}


def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls
    return deco


def build_model(name: str, **kwargs) -> "SurrogateModel":
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


class SurrogateModel(nn.Module, ABC):
    """Predicts normalized acceleration [num_nodes, 3]."""

    @abstractmethod
    def required_edges(self) -> list[str]:
        ...

    @abstractmethod
    def forward(self, batch: Data | Batch) -> torch.Tensor:
        ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
