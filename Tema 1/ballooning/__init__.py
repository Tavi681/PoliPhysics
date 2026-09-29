"""Discrete Elastic Rods (DER) simulation of multi-thread charged ballooning spiders.

Implements Algorithm 1 of Habchi & Jawed, "Ballooning in spiders using multiple
silk threads", Phys. Rev. E 105, 034401 (2022), exactly as written in
``tema1_objective.md``. See ``docs/algorithm_mapping.md`` for the line-by-line map.
"""

from .params import Params

__all__ = ["Params"]
