"""Constitutive law and derived material quantities.

The nominal tension of a thread of cross-section ``A`` is

    T(eps) = A * (E0 * eps + b * eps**3)   for 0 < eps < eps_b,
    T(eps) = 0                             for eps <= 0,

i.e. no compression and no bending stiffness. The strain-energy density is

    Phi(eps) = E0 * eps**2 / 2 + b * eps**4 / 4,

and the material's specific failure energy is e_mat = Phi(eps_b) / rho.

Wave speeds (all in SI units):

    c_L0  = sqrt(E0 / rho)                     small-strain longitudinal speed
    c_L   = sqrt(sigma / (rho * eps))          chord (secant) speed
    c_T   = sqrt(sigma / (rho * (1 + eps)))    transverse (kink) speed
    c_tan = max_eps sqrt((E0 + 3 b eps**2)/rho) maximum tangent wave speed

where sigma(eps) = T(eps) / A = E0 * eps + b * eps**3 is the nominal stress.

See the objective document for the validation targets of these quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "Material",
    "MATERIALS",
    "get_material",
    "register_material",
]


@dataclass(frozen=True)
class Material:
    """A thread material.

    Parameters
    ----------
    name:
        Registry key / human-readable label.
    E0:
        Small-strain Young modulus [Pa].
    b:
        Cubic stiffening coefficient [Pa].
    eps_b:
        Breaking strain [-].
    rho:
        Mass density [kg/m^3].
    """

    name: str
    E0: float
    b: float
    eps_b: float
    rho: float

    # -- nominal stress / tension -------------------------------------------------
    def sigma(self, eps):
        """Nominal stress sigma = T / A [Pa]. Zero for eps <= 0 (no compression)."""
        eps = np.asarray(eps, dtype=float)
        s = self.E0 * eps + self.b * eps**3
        return np.where(eps > 0.0, s, 0.0)

    def tension(self, eps, A):
        """Nominal tension T = A * sigma [N]."""
        return A * self.sigma(eps)

    def tangent_modulus(self, eps):
        """Tangent modulus dsigma/deps = E0 + 3 b eps**2 [Pa] (for eps > 0)."""
        eps = np.asarray(eps, dtype=float)
        dt = self.E0 + 3.0 * self.b * eps**2
        return np.where(eps > 0.0, dt, self.E0)

    # -- energy -------------------------------------------------------------------
    def Phi(self, eps):
        """Strain-energy density [J/m^3]. Zero for eps <= 0."""
        eps = np.asarray(eps, dtype=float)
        phi = self.E0 * eps**2 / 2.0 + self.b * eps**4 / 4.0
        return np.where(eps > 0.0, phi, 0.0)

    @property
    def e_mat(self):
        """Specific failure energy e_mat = Phi(eps_b) / rho [J/kg]."""
        return float(self.Phi(self.eps_b)) / self.rho

    @property
    def sigma_b(self):
        """Breaking stress sigma(eps_b) [Pa]."""
        return float(self.sigma(self.eps_b))

    # -- wave speeds --------------------------------------------------------------
    @property
    def c_L0(self):
        """Small-strain longitudinal speed sqrt(E0/rho) [m/s]."""
        return math.sqrt(self.E0 / self.rho)

    def c_L(self, eps):
        """Chord (secant) speed sqrt(sigma/(rho*eps)) [m/s].

        For eps -> 0 this tends to c_L0.
        """
        eps = np.asarray(eps, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            cl = np.sqrt(self.sigma(eps) / (self.rho * eps))
        # Limit at eps -> 0 is c_L0.
        cl = np.where(eps > 0.0, cl, self.c_L0)
        return cl

    def c_T(self, eps):
        """Transverse (kink) speed sqrt(sigma/(rho*(1+eps))) [m/s]."""
        eps = np.asarray(eps, dtype=float)
        return np.sqrt(self.sigma(eps) / (self.rho * (1.0 + eps)))

    def c_tan(self, eps=None):
        """Tangent wave speed sqrt((E0+3 b eps**2)/rho) [m/s].

        Called with no argument, returns the *maximum* tangent speed over the
        admissible range [0, eps_b], which is the value used for the CFL time
        step. The maximum is attained at eps_b because b >= 0.
        """
        if eps is None:
            eps = self.eps_b
        eps = np.asarray(eps, dtype=float)
        return np.sqrt((self.E0 + 3.0 * self.b * eps**2) / self.rho)

    @property
    def c_tan_max(self):
        """Maximum tangent wave speed over [0, eps_b] [m/s]."""
        return float(self.c_tan(self.eps_b))

    # -- inverse law --------------------------------------------------------------
    def strain_from_tension(self, T_e, A, tol=1e-14, max_iter=100):
        """Invert T(eps) = T_e by Newton iteration on the cubic.

        Returns the strain eps such that A*(E0*eps + b*eps**3) = T_e, checking
        that 0 <= eps < eps_b. Scalar in, scalar out.
        """
        sigma_target = T_e / A
        if sigma_target <= 0.0:
            return 0.0
        # Initial guess from the linear part.
        eps = sigma_target / self.E0
        for _ in range(max_iter):
            f = self.E0 * eps + self.b * eps**3 - sigma_target
            fp = self.E0 + 3.0 * self.b * eps**2
            step = f / fp
            eps -= step
            if abs(step) < tol:
                break
        if not (0.0 <= eps < self.eps_b):
            raise ValueError(
                f"strain_from_tension: eps={eps!r} outside [0, eps_b={self.eps_b}) "
                f"for T_e={T_e}, A={A}"
            )
        return float(eps)


# Two model materials from the objective table.
MATERIALS: dict[str, Material] = {
    "S": Material(name="S", E0=2e9, b=22.2e9, eps_b=0.30, rho=1300.0),
    "D": Material(name="D", E0=110e9, b=0.0, eps_b=0.032, rho=975.0),
}


def get_material(name: str) -> Material:
    """Look up a material in the registry by name."""
    try:
        return MATERIALS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown material {name!r}; known materials: {sorted(MATERIALS)}"
        ) from exc


def register_material(mat: Material, *, overwrite: bool = False) -> None:
    """Add a material to the registry (e.g. loaded from YAML)."""
    if mat.name in MATERIALS and not overwrite:
        raise ValueError(f"material {mat.name!r} already registered")
    MATERIALS[mat.name] = mat
