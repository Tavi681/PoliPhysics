"""Explicit velocity-Verlet integrator (Algorithm 1).

One step follows the order prescribed by Algorithm 1:

1. x^{n+1} = x^n + v^n dt + (1/2) a^n dt^2 for free nodes and the drone; anchors
   stay fixed; in ``gripped`` mode the gripped node follows the drone.
2. Compute segment strains at x^{n+1}.
3. Permanently remove all segments with eps >= eps_b (even several at once);
   book their elastic energy as failure dissipation and record the failures.
4. Compute tensions (zero if slack), then contact forces, then a^{n+1}.
5. v^{n+1} = v^n + (1/2)(a^n + a^{n+1}) dt.
6. Save an output frame every dt_out.
7. Check the stopping criteria.

Time step (see README): dt = min(C min(ell_e/n_s)/c_tan, C 2/omega_c), using the
maximum tangent wave speed c_tan (a correction to the paper, which writes the
chord speed) and a contact limit omega_c = sqrt(k_eff/m_min),
k_eff = (3/2) k_c delta_ref^{1/2}, delta_ref = frac * r_d.

Both a drone driver and a kinematic-forcing driver (a prescribed velocity or
displacement at one node, for validation) are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from ._kernels import segment_forces, segment_strains, have_numba
from .contact import (
    GrippedState,
    hertz_penalty_nodes,
    sphere_segment_penalty,
    nearest_node,
)

logger = logging.getLogger("netsim.integrator")

__all__ = ["Trajectory", "compute_time_step", "integrate", "resolve_k_c",
           "PenetrationError"]


class PenetrationError(RuntimeError):
    """Raised when the contact penetration exceeds delta_ref (abort guard)."""

    def __init__(self, delta, delta_ref, t, step):
        self.delta = delta
        self.delta_ref = delta_ref
        self.t = t
        self.step = step
        super().__init__(
            f"contact penetration delta={delta:.4g} m exceeded delta_ref="
            f"{delta_ref:.4g} m at t={t:.4g} s (step {step}); the contact time "
            f"step was under-resolved. Increase k_c / k_c_factor, lower C, or "
            f"set contact.penetration_guard='reduce_dt'.")


def resolve_k_c(contact, material, seg_A, seg_rest, r_d):
    """Return the effective penalty stiffness k_c.

    In ``relative`` mode, k_c is chosen so that the contact frequency at
    delta_ref equals ``k_c_factor`` times the axial frequency of a segment:
    omega_c(delta_ref) = k_c_factor * sqrt((E0 A / l_s) / m_node). With
    k_eff = (3/2) k_c sqrt(delta_ref) and omega = sqrt(k/m_node), the node mass
    cancels and

        k_c = k_c_factor^2 * (E0 A / l_s) / ((3/2) sqrt(delta_ref)),

    evaluated at the stiffest (finest, largest-A) segment for a conservative
    (high) stiffness.
    """
    if contact.k_c_mode == "absolute":
        return float(contact.k_c)
    k_axial = float(np.max(material.E0 * np.asarray(seg_A) / np.asarray(seg_rest)))
    delta_ref = contact.delta_ref_frac * r_d
    k_c = contact.k_c_factor ** 2 * k_axial / (1.5 * np.sqrt(delta_ref))
    return float(k_c)


@dataclass
class Trajectory:
    """Recorded frames and outcome of a run."""

    t: np.ndarray
    x: np.ndarray            # (n_t, n_node, 3)
    v: np.ndarray            # (n_t, n_node, 3)
    intact: np.ndarray       # (n_t, n_seg) bool
    drone: np.ndarray        # (n_t, 6) pos+vel (NaN in kinematic mode)
    energy: np.ndarray       # (n_t, 6): KE_dr, KE_net, U_el, U_fail, U_c, dE_cap
    failures: np.ndarray     # (n_f, 3): seg index, parent thread, time
    failure_midpoints: np.ndarray  # (n_f, 3)
    outcome: str = "timeout"
    arrested: bool = False
    w_max: float = 0.0
    R_d: float = 0.0
    energy_error: float = 0.0
    dt: float = 0.0
    dt_cfl: float = 0.0
    dt_contact: float = float("inf")
    U_prestress: float = 0.0
    E0_total: float = 0.0
    steps: int = 0
    delta_max: float = 0.0
    delta_ref: float = 0.0
    k_c: float = 0.0
    # Lateral drone position diagnostics (round 4).
    drone_x_arrest: float = float("nan")
    drone_y_arrest: float = float("nan")
    drone_x_first_fail: float = float("nan")
    drone_y_first_fail: float = float("nan")
    t_perforate: float = float("nan")


def compute_time_step(material, seg_rest, mass, drone, contact, C, *,
                      drone_mode=True, k_c=None):
    """Return (dt, dt_cfl, dt_contact) with both limits logged."""
    if k_c is None:
        k_c = contact.k_c
    min_spacing = float(np.min(seg_rest))
    c_tan = material.c_tan_max
    dt_cfl = C * min_spacing / c_tan
    dt_contact = float("inf")
    if drone_mode:
        m_free = mass[mass > 0]
        m_min = float(min(m_free.min(), drone.M)) if m_free.size else drone.M
        delta_ref = contact.delta_ref_frac * drone.r_d
        k_eff = 1.5 * k_c * np.sqrt(delta_ref)
        omega_c = np.sqrt(k_eff / m_min)
        dt_contact = C * 2.0 / omega_c
    dt = min(dt_cfl, dt_contact)
    logger.info("time step: dt_cfl=%.4g s (c_tan=%.1f m/s), dt_contact=%.4g s, "
                "chosen dt=%.4g s", dt_cfl, c_tan, dt_contact, dt)
    return dt, dt_cfl, dt_contact


def _contact_energy(x, xd, r_d, k_c, active, seg_edges=None, intact=None,
                    segment_contact=False):
    """Contact potential energy at the current configuration."""
    _, _, e = hertz_penalty_nodes(x, xd, r_d, k_c, active)
    if segment_contact and seg_edges is not None:
        _, _, es = sphere_segment_penalty(x, seg_edges, intact, xd, r_d, k_c,
                                          active)
        e += es
    return e


def integrate(disc, material, drone, numerics, contact, output,
              kinematic=None, *, net_R=1.0, impact_point=None,
              stop_on_failure=False, stop_after_n_failures=None):
    """Run Algorithm 1 and return a :class:`Trajectory`.

    Parameters
    ----------
    disc:
        :class:`~netsim.discretize.Discretization`.
    material:
        :class:`~netsim.materials.Material`.
    drone, numerics, contact, output:
        Config dataclasses.
    kinematic:
        Optional :class:`~netsim.config.KinematicConfig`; if enabled, the drone
        is not used and a node is driven kinematically.
    net_R:
        Frame radius (for the ``perforated`` criterion and R_d).
    impact_point:
        (px, py) in-plane impact point; defaults to ``drone.p``.
    stop_after_n_failures:
        If set, stop once this many permanent failures have been recorded
        (useful for quasi-static criterion-B pulls).
    """
    drone_mode = not (kinematic is not None and kinematic.enabled)

    x = disc.x0.copy()
    v = np.zeros_like(x)
    mass = disc.mass.copy()
    anchored = disc.anchored.copy()
    seg_edges = disc.seg_edges
    seg_rest = disc.seg_rest_length
    seg_A = disc.seg_A
    intact = np.ones(disc.n_seg, dtype=bool)
    E0, b, eps_b = material.E0, material.b, material.eps_b
    use_numba = numerics.use_numba and have_numba()
    damping = numerics.damping
    g_vec = np.array([0.0, 0.0, -9.81]) if drone.gravity else np.zeros(3)

    # Base free mask (integrated with Verlet).
    base_free = ~anchored
    forced_node = -1
    free_lateral = False
    kdir = np.zeros(3)
    x_forced0 = None
    if not drone_mode:
        forced_node = kinematic.node
        kdir = kinematic.unit_direction()
        x_forced0 = x[forced_node].copy()
        free_lateral = bool(getattr(kinematic, "free_lateral", False))
        if free_lateral:
            # Keep the node in the free set; only the kdir component is
            # overwritten each step (lateral DOFs respond to net forces).
            pass
        else:
            base_free[forced_node] = False

    # Drone state.
    if drone_mode:
        p = np.asarray(impact_point if impact_point is not None else drone.p,
                       dtype=float)
        p3 = np.array([p[0], p[1], 0.0])
        xd = np.array([p[0], p[1], drone.r_d])   # tangent to net
        vd = np.array([0.0, 0.0, -drone.v0])
        M_eff = float(drone.M)
        grip = GrippedState()
    else:
        p3 = np.zeros(3)
        xd = np.full(3, np.nan)
        vd = np.full(3, np.nan)
        M_eff = float("nan")
        grip = GrippedState()

    r_d = drone.r_d
    # Resolve the penalty stiffness (absolute or relative to axial stiffness).
    k_c = resolve_k_c(contact, material, seg_A, seg_rest, r_d)
    if contact.k_c_mode == "relative":
        logger.info("k_c resolved to %.4g N/m^1.5 (relative, factor=%.3g)",
                    k_c, contact.k_c_factor)

    dt, dt_cfl, dt_contact = compute_time_step(
        material, seg_rest, mass, drone, contact, numerics.C,
        drone_mode=drone_mode, k_c=k_c)
    # Base step used when no segment is near breaking. Near-break shrink is
    # applied per-step only; do not permanently ratchet dt downward or
    # continue_to_B / long runs become ~4× slower after the first approach.
    dt_base = dt

    seg_contact = contact.segment_contact
    delta_ref = contact.delta_ref_frac * r_d
    delta_max_run = 0.0
    if drone_mode:
        _mfree = mass[mass > 0]
        m_min = float(min(_mfree.min(), drone.M)) if _mfree.size else drone.M
    else:
        m_min = 0.0

    def accelerations(xc, vc, xdc, vdc, intactc, gripc, M_eff_c):
        """Nodal and drone accelerations at the given state."""
        f = segment_forces(xc, seg_edges, seg_rest, seg_A, intactc, E0, b,
                           use_numba=use_numba)
        ce = 0.0
        fd = np.zeros(3)
        if drone_mode:
            active = base_free.copy()
            if gripc.active:
                active[gripc.node] = False
            fc, fcd, ce1 = hertz_penalty_nodes(xc, xdc, r_d, k_c, active)
            f = f + fc
            fd = fd + fcd
            ce += ce1
            if seg_contact:
                fs, fsd, ce2 = sphere_segment_penalty(
                    xc, seg_edges, intactc, xdc, r_d, k_c, active)
                f = f + fs
                fd = fd + fsd
                ce += ce2
            if gripc.active:
                fd = fd + f[gripc.node]  # transmit force on gripped node
        with np.errstate(invalid="ignore"):
            a = f / mass[:, None] - damping * vc
        a = a + g_vec
        a[anchored] = 0.0
        if not drone_mode:
            a[forced_node] = 0.0 if not free_lateral else a[forced_node]
            if free_lateral:
                # Constraint force cancels acceleration along the prescribed
                # direction; lateral components remain free.
                a[forced_node] = (a[forced_node]
                                  - np.dot(a[forced_node], kdir) * kdir)
        if drone_mode and gripc.active:
            a[gripc.node] = 0.0
        ad = np.zeros(3)
        if drone_mode:
            ad = fd / M_eff_c - damping * vdc + g_vec
        return a, ad, ce

    # Initial accelerations (a^0).
    a, ad, ce = accelerations(x, v, xd, vd, intact, grip, M_eff)

    # Output buffers.
    t_list, x_list, v_list, intact_list, drone_list, energy_list = (
        [], [], [], [], [], [])
    failures = []
    failure_mid = []

    from .energy import (elastic_energy, kinetic_energy_net,
                         kinetic_energy_drone)

    U_prestress = elastic_energy(x, seg_edges, seg_rest, seg_A, intact, material)
    U_failure = 0.0
    U_capture = 0.0

    def record(t):
        exclude = None
        if drone_mode and grip.active:
            exclude = np.zeros(disc.n_node, dtype=bool)
            exclude[grip.node] = True
        ke_net = kinetic_energy_net(v, mass, exclude)
        if drone_mode:
            ke_dr = kinetic_energy_drone(M_eff, vd)
            active = base_free.copy()
            if grip.active:
                active[grip.node] = False
            uc = _contact_energy(x, xd, r_d, k_c, active, seg_edges, intact,
                                 seg_contact)
        else:
            ke_dr = 0.0
            uc = 0.0
        u_el = elastic_energy(x, seg_edges, seg_rest, seg_A, intact, material)
        t_list.append(t)
        x_list.append(x.copy())
        v_list.append(v.copy())
        intact_list.append(intact.copy())
        drone_list.append(np.concatenate([xd, vd]))
        energy_list.append([ke_dr, ke_net, u_el, U_failure, uc, U_capture])

    # Initial frame.
    record(0.0)
    E_kin0 = (0.5 * drone.M * drone.v0 ** 2) if drone_mode else 0.0
    E_norm = E_kin0 + U_prestress
    E0_total = sum(energy_list[0])

    t = 0.0
    n_out = 0
    max_steps = int(np.ceil(numerics.t_end / dt)) + 2
    had_negative_vz = False
    min_drone_z = xd[2] if drone_mode else 0.0
    R_d = 0.0
    outcome = "timeout"
    arrested = False
    steps = 0
    drone_x_arrest = float("nan")
    drone_y_arrest = float("nan")
    drone_x_first_fail = float("nan")
    drone_y_first_fail = float("nan")
    t_perforate = float("nan")
    eps_old, _ = segment_strains(x, seg_edges, seg_rest)

    while t < numerics.t_end:
        # Locally reduce dt when any intact segment is approaching breaking.
        # Restore from dt_base each step so the shrink is temporary.
        dt = dt_base
        near = intact & (eps_old > 0.85 * eps_b)
        if np.any(near):
            dt = min(dt, 0.25 * dt_cfl)

        # --- Alg.1 line 1: position update -----------------------------------
        x_new = x.copy()
        x_new[base_free] = (x[base_free] + v[base_free] * dt
                            + 0.5 * a[base_free] * dt * dt)
        if drone_mode:
            xd_new = xd + vd * dt + 0.5 * ad * dt * dt
        else:
            xd_new = xd
            # Kinematic forcing of the driven node.
            if free_lateral:
                # Verlet already advanced the free node; overwrite only the
                # prescribed component along kdir.
                lat = (x_new[forced_node]
                       - np.dot(x_new[forced_node], kdir) * kdir)
                if kinematic.mode == "velocity":
                    along = (np.dot(x[forced_node], kdir)
                             + kinematic.value(t) * dt)
                else:  # displacement
                    along = (np.dot(x_forced0, kdir)
                             + kinematic.value(t + dt))
                x_new[forced_node] = lat + along * kdir
            elif kinematic.mode == "velocity":
                x_new[forced_node] = x[forced_node] + kinematic.value(t) * kdir * dt
            else:  # displacement
                x_new[forced_node] = x_forced0 + kinematic.value(t + dt) * kdir

        # Grip activation (first contact) and slaving.
        if drone_mode and contact.mode == "gripped" and not grip.active:
            dist = np.linalg.norm(x_new - xd_new, axis=1)
            delta = r_d - dist
            if np.any(base_free & (delta > 0.0)):
                g = nearest_node(x_new, p3, base_free)
                grip.active = True
                grip.node = g
                grip.offset = (x_new[g] - xd_new).copy()
                grip.mass = float(mass[g])
                # Inelastic capture: book dE_cap = 1/2 mu |v_rel|^2 and set the
                # combined velocity to the centre-of-mass velocity. Leaving vd
                # unchanged while adding the node mass to M_eff was a bookkeeping
                # error that left a C-independent energy floor of order dE_cap.
                v_rel = v[g] - vd
                mu = M_eff * grip.mass / (M_eff + grip.mass)
                U_capture = 0.5 * mu * float(np.dot(v_rel, v_rel))
                v_com = (M_eff * vd + grip.mass * v[g]) / (M_eff + grip.mass)
                M_eff = M_eff + grip.mass
                vd = v_com.copy()
                v[g] = vd.copy()
                logger.info("gripped node %d at t=%.4g s (mass %.4g kg, "
                            "dE_cap=%.4g J)", g, t, grip.mass, U_capture)
        if drone_mode and grip.active:
            x_new[grip.node] = xd_new + grip.offset

        # Penetration tracking and runtime guard.
        if drone_mode:
            active_p = base_free.copy()
            if grip.active:
                active_p[grip.node] = False
            dvec = x_new[active_p] - xd_new
            if dvec.size:
                dd = np.sqrt(np.sum(dvec * dvec, axis=1))
                step_delta = float(np.max(r_d - dd))
                if step_delta > delta_max_run:
                    delta_max_run = step_delta
                if step_delta > delta_ref and contact.penetration_guard != "off":
                    if contact.penetration_guard == "abort":
                        raise PenetrationError(step_delta, delta_ref, t + dt,
                                               steps + 1)
                    else:  # reduce_dt: shrink dt for subsequent steps
                        k_eff = 1.5 * k_c * np.sqrt(step_delta)
                        omega = np.sqrt(k_eff / m_min)
                        dt_new = contact.penetration_margin * 2.0 / omega
                        if dt_new < dt_base:
                            logger.info("reduce_dt: delta=%.4g>%.4g, dt %.3g->"
                                        "%.3g", step_delta, delta_ref, dt_base,
                                        dt_new)
                            dt_base = dt_new
                            dt = min(dt, dt_base)

        # --- Alg.1 line 2: strains -------------------------------------------
        eps_new, _ = segment_strains(x_new, seg_edges, seg_rest)

        # --- Alg.1 line 3: permanent failure ---------------------------------
        # Book failure energy at the interpolated crossing eps = eps_b (not at
        # the overshot eps_new). For stiff D the CFL step can jump well past
        # eps_b in one dt; booking Phi(eps_new) then leaves a late-run residual.
        failed = intact & (eps_new >= eps_b)
        any_failed = bool(np.any(failed))
        if any_failed:
            for s in np.nonzero(failed)[0]:
                e0 = float(eps_old[s])
                e1 = float(eps_new[s])
                # Linear interpolation of the crossing time within the step;
                # book Phi at eps_b (the continuum breaking value).
                phi_book = float(material.Phi(eps_b))
                mid = 0.5 * (x_new[seg_edges[s, 0]] + x_new[seg_edges[s, 1]])
                if e1 > e0 and e0 < eps_b <= e1:
                    alpha = (eps_b - e0) / (e1 - e0)
                    # Midpoint at the interpolated crossing (blend old/new).
                    mid_old = 0.5 * (x[seg_edges[s, 0]] + x[seg_edges[s, 1]])
                    mid = (1.0 - alpha) * mid_old + alpha * mid
                    t_fail = t + alpha * dt
                else:
                    t_fail = t + dt
                U_failure += seg_A[s] * seg_rest[s] * phi_book
                failures.append((int(s), int(disc.seg_parent[s]), t_fail))
                failure_mid.append(mid.copy())
                R_d = max(R_d, float(np.linalg.norm(mid[:2] - p3[:2])))
                if drone_mode and not np.isfinite(drone_x_first_fail):
                    drone_x_first_fail = float(xd_new[0])
                    drone_y_first_fail = float(xd_new[1])
            intact[failed] = False
        eps_old = eps_new

        # --- Alg.1 line 4: forces and a^{n+1} --------------------------------
        a_new, ad_new, ce = accelerations(x_new, v, xd_new, vd, intact, grip,
                                          M_eff)

        # --- Alg.1 line 5: velocity update -----------------------------------
        v_new = v.copy()
        v_new[base_free] = v[base_free] + 0.5 * (a[base_free] + a_new[base_free]) * dt
        if drone_mode:
            vd_new = vd + 0.5 * (ad + ad_new) * dt
            if grip.active:
                v_new[grip.node] = vd_new
        else:
            vd_new = vd
            if free_lateral:
                # Lateral velocity from Verlet; prescribed component from the
                # kinematic law.
                lat_v = (v_new[forced_node]
                         - np.dot(v_new[forced_node], kdir) * kdir)
                if kinematic.mode == "velocity":
                    v_presc = kinematic.value(t + dt)
                else:
                    # Finite-difference of the prescribed displacement.
                    v_presc = (kinematic.value(t + dt) - kinematic.value(t)) / dt
                v_new[forced_node] = lat_v + v_presc * kdir
            elif kinematic.mode == "velocity":
                v_new[forced_node] = kinematic.value(t + dt) * kdir
            else:
                v_new[forced_node] = (x_new[forced_node] - x[forced_node]) / dt

        # Commit.
        x, v, xd, vd, a, ad = x_new, v_new, xd_new, vd_new, a_new, ad_new
        t += dt
        steps += 1

        if drone_mode:
            min_drone_z = min(min_drone_z, xd[2])

        # --- Alg.1 line 6: output --------------------------------------------
        cur_out = int(np.floor(t / numerics.dt_out + 1e-9))
        if cur_out > n_out:
            n_out = cur_out
            record(t)

        # Optional early stop once a failure has occurred (quasi-static runs).
        if stop_on_failure and any_failed:
            outcome = "failure"
            if not t_list or t_list[-1] < t:
                record(t)
            break
        if stop_after_n_failures is not None and failures:
            # Count time-separated failure *events*, not individual segments: a
            # cascade that breaks several segments in one step (or within a
            # few CFL steps) is one event. Criterion B needs the next event
            # after load redistribution.
            gap = max(1e-4, 10.0 * dt_cfl)
            n_events = 0
            t_ev = -1e99
            for _s, _p, tf in failures:
                if float(tf) > t_ev + gap:
                    n_events += 1
                    t_ev = float(tf)
            if n_events >= int(stop_after_n_failures):
                outcome = "failure"
                if not t_list or t_list[-1] < t:
                    record(t)
                break

        # --- Alg.1 line 7: stopping ------------------------------------------
        if drone_mode:
            vz = vd[2]
            if vz < 0.0:
                had_negative_vz = True
            if had_negative_vz and vz >= 0.0:
                outcome = "arrested"
                arrested = True
                drone_x_arrest = float(xd[0])
                drone_y_arrest = float(xd[1])
                break
            # Early perforation ("point of no return"): still descending, and
            # either no intact segment is in contact / connected to the gripped
            # node, or the drone centre has passed below the deepest intact
            # material by more than r_d. The classical z < -2R is the fallback.
            perforated_early = False
            if vz < 0.0:
                active_c = base_free.copy()
                if grip.active:
                    active_c[grip.node] = False
                in_contact = False
                if active_c.any():
                    dvec = x[active_c] - xd
                    dd = np.sqrt(np.sum(dvec * dvec, axis=1))
                    in_contact = bool(np.any(r_d - dd > 0.0))
                grip_connected = False
                if grip.active:
                    g = grip.node
                    conn = (seg_edges[:, 0] == g) | (seg_edges[:, 1] == g)
                    grip_connected = bool(np.any(intact & conn))
                if grip.active and not grip_connected and not in_contact:
                    perforated_early = True
                elif not grip.active and not in_contact and xd[2] < 0.0:
                    if np.any(intact):
                        nodes_i = np.unique(seg_edges[intact].ravel())
                        z_deep = float(np.min(x[nodes_i, 2]))
                        if xd[2] < z_deep - r_d:
                            perforated_early = True
                    else:
                        perforated_early = True
            if perforated_early or xd[2] < -2.0 * net_R:
                outcome = "perforated"
                t_perforate = float(t)
                drone_x_arrest = float(xd[0])
                drone_y_arrest = float(xd[1])
                logger.info("perforated at t=%.4g s (early=%s, z=%.4g)",
                            t, perforated_early, float(xd[2]))
                break
        if steps > max_steps * 4:  # safety guard
            logger.warning("step guard triggered; stopping")
            break

    # Ensure a final frame at the stop time.
    if not t_list or t_list[-1] < t:
        record(t)

    # A timeout after the drone has already reversed (vz crossed through zero
    # from below, or is non-negative at stop) is an arrest: the bounce happened
    # but the run hit t_end before the discrete check caught it, or the stiff
    # contact kept vz asymptotically near zero. Counting that as a failure of
    # the criterion would make criterion B fail on heavy nets that criterion A
    # (same physics, later in the scan) accepts.
    if (outcome == "timeout" and drone_mode and had_negative_vz
            and float(vd[2]) >= -1e-12):
        outcome = "arrested"
        arrested = True
        drone_x_arrest = float(xd[0])
        drone_y_arrest = float(xd[1])
        logger.info("timeout after vz reversal treated as arrest (vz=%.4g)",
                    float(vd[2]))
    elif outcome == "timeout":
        logger.info("run reached t_end without arrest/perforation (timeout)")
        if drone_mode:
            drone_x_arrest = float(xd[0])
            drone_y_arrest = float(xd[1])

    # Assemble.
    t_arr = np.asarray(t_list)
    energy_arr = np.asarray(energy_list)
    E_tot = energy_arr.sum(axis=1)
    if E_norm > 0:
        energy_error = float(np.max(np.abs(E_tot - E0_total)) / E_norm)
    else:
        energy_error = 0.0

    w_max = float((drone.r_d - min_drone_z)) if drone_mode else 0.0

    if failures:
        failures_arr = np.asarray(failures, dtype=float)
        failure_mid_arr = np.asarray(failure_mid, dtype=float)
    else:
        failures_arr = np.zeros((0, 3))
        failure_mid_arr = np.zeros((0, 3))

    return Trajectory(
        t=t_arr,
        x=np.asarray(x_list),
        v=np.asarray(v_list),
        intact=np.asarray(intact_list),
        drone=np.asarray(drone_list),
        energy=energy_arr,
        failures=failures_arr,
        failure_midpoints=failure_mid_arr,
        outcome=outcome,
        arrested=arrested,
        w_max=w_max,
        R_d=R_d,
        energy_error=energy_error,
        dt=dt,
        dt_cfl=dt_cfl,
        dt_contact=dt_contact,
        U_prestress=U_prestress,
        E0_total=E0_total,
        steps=steps,
        delta_max=delta_max_run,
        delta_ref=delta_ref,
        k_c=k_c,
        drone_x_arrest=drone_x_arrest,
        drone_y_arrest=drone_y_arrest,
        drone_x_first_fail=drone_x_first_fail,
        drone_y_first_fail=drone_y_first_fail,
        t_perforate=t_perforate,
    )
