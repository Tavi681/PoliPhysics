"""Electric field and air-velocity models (fields.py).

Electric field E(z) [V/m] acting along +z, and air velocity u(x, t) [m/s].
All fields expose an analytic derivative dE/dz for the Jacobian of the lift force.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class ElectricField(Protocol):
    """Vertical atmospheric electric field E(z)."""

    def E(self, z: float) -> float:  # noqa: N802 - matches paper symbol
        ...

    def dE(self, z: float) -> float:  # noqa: N802
        ...


class GorhamField:
    """Gorham flat-earth model: E = E0 exp(-alpha z) (Gorham flat-earth field)."""

    def __init__(self, E0: float = 120.0, alpha: float = 3e-4):
        self.E0 = E0
        self.alpha = alpha

    def E(self, z: float) -> float:  # noqa: N802
        return self.E0 * math.exp(-self.alpha * z)

    def dE(self, z: float) -> float:  # noqa: N802
        return -self.alpha * self.E0 * math.exp(-self.alpha * z)


class ChamberField:
    """Morley & Gorham chamber fit: E1 e^{-z/z1} + E2 e^{-z/z2} + Einf (chamber field fit)."""

    def __init__(self, E1: float = 2.52e5, z1: float = 1.51e-3,
                 E2: float = 5.07e4, z2: float = 7.93e-3, Einf: float = 7.41e3):
        self.E1 = E1
        self.z1 = z1
        self.E2 = E2
        self.z2 = z2
        self.Einf = Einf

    def E(self, z: float) -> float:  # noqa: N802
        return (self.E1 * math.exp(-z / self.z1)
                + self.E2 * math.exp(-z / self.z2) + self.Einf)

    def dE(self, z: float) -> float:  # noqa: N802
        return (-self.E1 / self.z1 * math.exp(-z / self.z1)
                - self.E2 / self.z2 * math.exp(-z / self.z2))


class ConstantField:
    """Uniform field E(z) = E0."""

    def __init__(self, E0: float):
        self.E0 = E0

    def E(self, z: float) -> float:  # noqa: N802
        return self.E0

    def dE(self, z: float) -> float:  # noqa: N802
        return 0.0


class AirFlow(Protocol):
    """Ambient air velocity field u(x, t) -> (3,)."""

    def u(self, x: np.ndarray, t: float) -> np.ndarray:
        ...


class ZeroFlow:
    """Still air, u = 0 everywhere."""

    def u(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.zeros(3)


class UniformFlow:
    """Spatially uniform, steady flow u = U."""

    def __init__(self, U):
        self.U = np.asarray(U, dtype=float)

    def u(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.U.copy()


class KinematicSimulation:
    """Reserved interface for a synthetic turbulent flow (to be implemented later).

    A future implementation should provide ``u(x, t) -> (3,)`` consistent with the
    ``AirFlow`` protocol, driven by the ``/turb`` metadata (sigma_w, ell, U_h,
    lambda, seed, and the k_n, a_n, b_n, omega_n mode coefficients). It is a
    documented stub only; instantiating and calling it raises ``NotImplementedError``.
    """

    def __init__(self, *args, **kwargs):
        self._config = (args, kwargs)

    def u(self, x: np.ndarray, t: float) -> np.ndarray:  # pragma: no cover - stub
        raise NotImplementedError(
            "KinematicSimulation is a reserved interface and is not implemented yet."
        )


def make_field(P) -> ElectricField:
    """Construct the electric field selected in ``P.field_model``."""
    if P.field_model == "gorham":
        return GorhamField()
    if P.field_model == "chamber":
        return ChamberField()
    if P.field_model == "constant":
        return ConstantField(P.E_constant)
    raise ValueError(f"unknown field_model {P.field_model!r}")


def make_flow(P) -> AirFlow:
    """Construct the air-flow model selected in ``P.flow_model``."""
    if P.flow_model == "zero":
        return ZeroFlow()
    if P.flow_model == "uniform":
        return UniformFlow(P.flow_velocity)
    raise ValueError(f"unknown flow_model {P.flow_model!r}")
