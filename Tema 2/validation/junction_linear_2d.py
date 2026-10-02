"""Transverse-corrected linear junction theory (round-2 item 1).

For a massless hub with small displacement, an incident longitudinal pulse on
thread 0 is resisted by the off-axis threads both longitudinally (impedance
``Z_L = mu0 c_L``) and transversely (impedance ``Z_T = mu0 c_T``), both evaluated
at the prestress ``eps_p``. The linear result keeps the 1D form with ``S_N``
replaced by

    S_N' = S_N + (Z_T/Z_L) * sum_{j=1}^{N-1} sin^2(phi_j)
         = S_N + (Z_T/Z_L) * (N/2),

    T0/Tinc     = 2 S_N' / (1 + S_N'),
    dT_j/Tinc   = -2 cos(phi_j) / (1 + S_N').

Lagrangian wave speeds at the prestress:

    c_L   = sqrt(T'(eps_p)/rho)          = sqrt((E0 + 3 b eps_p^2)/rho),
    c_T   = sqrt(T(eps_p)/(mu0 (1+eps_p)))  (``with`` the (1+eps_p) factor), or
    c_T   = sqrt(T(eps_p)/mu0)              (``wo`` variant, without the factor).

Since T = A sigma and mu0 = rho A, the cross-section cancels: c_T^2 = sigma/rho
(wo) or sigma/(rho (1+eps_p)) (with).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["S_N", "junction_linear_2d", "linear_ratios_1d"]


def S_N(N: int) -> float:
    """1D junction coefficient: 1 for N=2, N/2-1 for N>=3."""
    return 1.0 if N == 2 else N / 2 - 1


def _c_L(material, eps_p: float) -> float:
    return math.sqrt((material.E0 + 3.0 * material.b * eps_p ** 2) / material.rho)


def _c_T(material, eps_p: float, variant: str) -> float:
    sigma_p = float(material.sigma(eps_p))
    if variant == "with":
        return math.sqrt(sigma_p / (material.rho * (1.0 + eps_p)))
    if variant == "wo":
        return math.sqrt(sigma_p / material.rho)
    raise ValueError(f"unknown c_T variant {variant!r}")


def linear_ratios_1d(N: int):
    """1D predictions: (T0/Tinc, dT_j/Tinc array over threads)."""
    s = S_N(N)
    phi = 2 * np.pi * np.arange(N) / N
    T0 = 2 * s / (1 + s)
    dT = -2 * np.cos(phi) / (1 + s)
    return T0, dT


def junction_linear_2d(material, N: int, eps_p: float, variant: str = "with"):
    """Transverse-corrected linear junction prediction.

    Returns a dict with S_N, S_N', Z_T/Z_L, c_L, c_T, T0/Tinc and the per-thread
    dT_j/Tinc array (index 0 is thread 0 itself), plus dTmax/dTmin over the
    off-axis threads (max is the opposite thread, min the nearest same-side one).
    """
    s = S_N(N)
    phi = 2 * np.pi * np.arange(N) / N
    cL = _c_L(material, eps_p)
    cT = _c_T(material, eps_p, variant)
    ZT_over_ZL = cT / cL
    # Transverse stiffening: sum_{j>=1} sin^2(phi_j). This equals N/2 for N>=3,
    # but is 0 for N=2 (collinear threads, sin(pi)=0), so compute it explicitly.
    sin2sum = float(np.sum(np.sin(phi[1:]) ** 2))
    s2d = s + ZT_over_ZL * sin2sum
    T0 = 2 * s2d / (1 + s2d)
    dT = -2 * np.cos(phi) / (1 + s2d)
    off = dT[1:]
    return {
        "N": N, "eps_p": eps_p, "variant": variant,
        "SN": s, "SN2d": s2d, "ZT_over_ZL": ZT_over_ZL, "sin2sum": sin2sum,
        "c_L": cL, "c_T": cT,
        "T0": T0, "dT": dT, "dTmax": float(off.max()), "dTmin": float(off.min()),
    }
