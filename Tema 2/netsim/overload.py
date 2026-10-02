"""Dynamic overload of an N-star hub (Item 5a).

A flat ``N``-radial star (no prestress) is loaded by a constant vertical force
``F`` applied to the central hub. Viscous damping is used only to reach the
static equilibrium (strain ``eps0`` in every radial); then, at ``t_b``, one
radial is removed -- instantaneously (``t_f = 0``) or with its tension ramped
linearly to zero over a finite release time ``t_f`` -- and the response of the
remaining radials is measured:

* ``eps_s`` : static strain after removal, re-equilibrated *with* damping;
* ``eps_m`` : peak dynamic strain after removal, with damping switched *off*.

The hub carries a configurable point mass ``m_hub`` so that the dynamic
amplification is governed by a single light mode (a drone-held node would be too
heavy). The forcing/removal are expressed through :func:`remove_thread` and
:class:`ForcingSpec`, i.e. the ``constant_force`` forcing mode.

No physics or tolerances are tuned here; the force ``F`` is derived analytically
from the target strain and all reported numbers are measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ._kernels import segment_forces, segment_strains
from .discretize import discretize
from .topology import star

__all__ = ["ThreadRemoval", "remove_thread", "ForcingSpec", "OverloadResult",
           "static_force_for_strain", "run_overload"]


@dataclass
class ThreadRemoval:
    """Schedule the removal of a coarse thread (``parent``) at time ``t_b``.

    The tension of the removed segments decays linearly to zero over ``t_f``
    (``t_f = 0`` is instantaneous removal).
    """

    parent: int
    t_b: float
    t_f: float = 0.0


def remove_thread(e: int, t_b: float, t_f: float = 0.0) -> ThreadRemoval:
    """``remove_thread(e, t_b)`` API: remove thread ``e`` at ``t_b`` over ``t_f``."""
    return ThreadRemoval(parent=int(e), t_b=float(t_b), t_f=float(t_f))


@dataclass
class ForcingSpec:
    """Constant-force forcing of a single node (the ``constant_force`` mode)."""

    node: int
    force: np.ndarray  # (3,) constant force vector [N]
    removals: List[ThreadRemoval] = field(default_factory=list)


@dataclass
class OverloadResult:
    N: int
    material: str
    eps0_target: float
    eps0_sim: float
    eps_s: float
    eps_m: float
    F: float
    w0: float
    k_eff: float
    omega_n: float
    T_n: float
    n_eff: float
    t_f: float
    t_f_over_Tn: float
    daf: float  # eps_m / eps_s


def static_force_for_strain(material, N, R, eps0):
    """Vertical hub force giving static radial strain ``eps0`` in a flat star.

    Each radial (rest length ``R``) stretches to ``R(1+eps0)`` when the hub
    deflects by ``w0 = R sqrt((1+eps0)^2 - 1)``; the vertical balance of the N
    radials gives ``F = N T(eps0) w0 / (R(1+eps0))`` (A = A_hat = 1e-6).
    """
    L = R * (1.0 + eps0)
    w0 = math.sqrt(max(L * L - R * R, 0.0))
    A = 1e-6
    T = A * float(material.sigma(eps0))
    F = N * T * (w0 / L)
    return F, w0


def _radial_strain(material, N, R, A_hat, eps0, w):
    """Analytical radial strain for hub deflection ``w`` (flat star)."""
    return math.sqrt(R * R + w * w) / R - 1.0


def _restoring_force(material, N, R, A_hat, w):
    """Vertical restoring force of the N radials for hub deflection ``w``."""
    L = math.sqrt(R * R + w * w)
    eps = L / R - 1.0
    T = A_hat * float(material.sigma(eps))
    return N * T * (w / L) if L > 0 else 0.0


def _hub_stiffness(material, N, R, A_hat, w0):
    """k_eff = dF/dw at the operating point (central difference)."""
    dw = max(1e-6 * R, 1e-9)
    fp = _restoring_force(material, N, R, A_hat, w0 + dw)
    fm = _restoring_force(material, N, R, A_hat, w0 - dw)
    return (fp - fm) / (2.0 * dw)


def run_overload(N, material, *, eps0, m_hub, R=1.0, n_s=4, A_hat=1e-6,
                 remove_parent=0, t_f_over_Tn=0.0, C=0.4,
                 settle_periods=40.0, dyn_periods=12.0, damp_ratio=1.0):
    """Run the overload scenario for one (N, material, eps0, t_f) point."""
    net = star(N, R, 0.0, material=material, A_hat=A_hat)
    disc = discretize(net, material, n_s, warn_ratio=1e9)

    x = disc.x0.copy()
    v = np.zeros_like(x)
    mass = disc.mass.copy()
    hub = 0  # coarse node 0 is the hub
    mass[hub] += m_hub  # the hub carries an extra point mass
    anchored = disc.anchored.copy()
    free = ~anchored

    # Target force and operating point (needed to seed the equilibrium geometry).
    F, w0 = static_force_for_strain(material, N, R, eps0)

    # Seed the straight, stretched equilibrium geometry (hub pulled down by w0,
    # interior nodes on the straight lines hub->anchor). Starting flat would make
    # the hub free-fall through a zero-stiffness configuration and overshoot.
    hub_pos = np.array([0.0, 0.0, -w0])
    x[hub] = hub_pos
    for i in np.nonzero(free)[0]:
        if i == hub:
            continue
        parent = int(disc.node_parent_thread[i])
        if parent < 0:
            continue
        anchor = disc.x0[parent + 1]  # star edge e = (0, e+1)
        t = float(disc.node_thread_pos[i])
        x[i] = (1.0 - t) * hub_pos + t * anchor
    seg_edges = disc.seg_edges
    seg_rest = disc.seg_rest_length
    seg_A = disc.seg_A
    seg_parent = disc.seg_parent
    E0, b = material.E0, material.b

    removed_seg = seg_parent == remove_parent
    remaining_seg = ~removed_seg

    # Operating-point stiffness and natural period.
    k_eff = _hub_stiffness(material, N, R, A_hat, w0)
    omega_n = math.sqrt(max(k_eff, 1e-30) / m_hub)
    T_n = 2.0 * math.pi / omega_n
    n_eff = w0 * k_eff / F if F > 0 else float("nan")

    force_vec = np.array([0.0, 0.0, -F])

    # Time step: min of material CFL and hub-mode stability.
    c_tan = material.c_tan_max
    dt_cfl = C * float(np.min(seg_rest)) / c_tan
    dt_hub = C * 2.0 / omega_n
    dt = min(dt_cfl, dt_hub)

    # Per-unit-mass damping on the hub mode (critical ~ 2 omega_n).
    damp = damp_ratio * 2.0 * omega_n

    def forces(xc, intact, ramp_removed):
        f = segment_forces(xc, seg_edges, seg_rest, seg_A, intact & remaining_seg,
                           E0, b)
        if ramp_removed > 0.0:
            fr = segment_forces(xc, seg_edges, seg_rest, seg_A,
                                intact & removed_seg, E0, b)
            f = f + ramp_removed * fr
        f[hub] += force_vec
        return f

    def step(xc, vc, ac, intact, damping, ramp_removed):
        xn = xc.copy()
        xn[free] = xc[free] + vc[free] * dt + 0.5 * ac[free] * dt * dt
        fn = forces(xn, intact, ramp_removed)
        an = fn / mass[:, None] - damping * vc
        an[anchored] = 0.0
        vn = vc.copy()
        vn[free] = vc[free] + 0.5 * (ac[free] + an[free]) * dt
        return xn, vn, an

    # --- Phase 1: settle to static equilibrium (all radials intact). ----------
    # ramp_removed = 1.0 keeps the soon-to-be-removed radial fully present.
    intact = np.ones(disc.n_seg, dtype=bool)
    a = forces(x, intact, 1.0) / mass[:, None] - damp * v
    a[anchored] = 0.0
    n_settle = int(settle_periods * T_n / dt) + 1
    for _ in range(n_settle):
        x, v, a = step(x, v, a, intact, damp, 1.0)

    eps_all, _ = segment_strains(x, seg_edges, seg_rest)
    eps0_sim = float(np.max(eps_all))  # all radials equal by symmetry
    x_eq, v_eq = x.copy(), v.copy()

    t_f = t_f_over_Tn * T_n

    def phase2(x0, damping, duration, measure_peak):
        xc, vc = x0.copy(), np.zeros_like(x0)
        intact_local = np.ones(disc.n_seg, dtype=bool)
        ac = forces(xc, intact_local, 1.0) / mass[:, None] - damping * vc
        ac[anchored] = 0.0
        n_steps = int(duration / dt) + 1
        peak = 0.0
        last = 0.0
        for k in range(n_steps):
            t = k * dt
            if t_f <= 0.0:
                ramp = 0.0
                intact_local[removed_seg] = False
            else:
                if t < t_f:
                    ramp = 1.0 - t / t_f
                else:
                    ramp = 0.0
                    intact_local[removed_seg] = False
            xc, vc, ac = step(xc, vc, ac, intact_local, damping, ramp)
            eps_rem, _ = segment_strains(xc, seg_edges, seg_rest)
            cur = float(np.max(eps_rem[remaining_seg]))
            peak = max(peak, cur)
            last = cur
        return peak if measure_peak else last

    # --- Phase 2a: static after removal (damping ON). -------------------------
    eps_s = phase2(x_eq, damp, settle_periods * T_n, measure_peak=False)
    # --- Phase 2b: dynamic after removal (damping OFF). -----------------------
    eps_m = phase2(x_eq, 0.0, dyn_periods * T_n, measure_peak=True)

    return OverloadResult(
        N=N, material=material.name, eps0_target=eps0, eps0_sim=eps0_sim,
        eps_s=eps_s, eps_m=eps_m, F=F, w0=w0, k_eff=k_eff, omega_n=omega_n,
        T_n=T_n, n_eff=n_eff, t_f=t_f, t_f_over_Tn=t_f_over_Tn,
        daf=(eps_m / eps_s if eps_s > 0 else float("nan")),
    )
