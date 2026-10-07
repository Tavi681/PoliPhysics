"""Implicit DER time stepping -- Algorithm 1 (integrator.py).

This module implements Algorithm 1 of the objective *exactly*, line by line. Each
step below is delimited by a comment "# Alg.1 line <k>: <text>". The residual and
Jacobian follow Eq. (eq:residual) and Eq. (eq:jacobian):

    Residual (Eq. (eq:residual)):
      F(xi) = M/dt ((xi - xi^n)/dt - xi_dot^n) + grad E_el(xi) - f_ext(xi)
    Jacobian (Eq. (eq:jacobian)):
      J = M/dt^2 + Hess E_el - d f_ext/d xi
        = M/dt^2 + Hess E_el + C/dt - d(F_l + F_r)/d xi
    where C is the block-diagonal RFT resistance matrix (drag term, tangents
    lagged) so that d F_v / d xi = -C/dt.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .params import Params
from .geometry import Topology, initial_state
from .frames import FrameState
from .elastic import elastic_energy_grad_hess
from . import forces
from . import observables
from .fields import make_field, make_flow


@dataclass
class Trajectory:
    """Recorded outputs of a run."""

    t: np.ndarray
    x: np.ndarray          # (n_t, n_nodes, 3)
    v: np.ndarray          # (n_t, n_nodes, 3)
    theta: np.ndarray      # (n_t, n_edges)
    edges: np.ndarray      # (n_edges, 2)
    thread_id: np.ndarray  # (n_nodes,)
    q_node: np.ndarray     # (n_nodes,)
    outcome: dict = dc_field(default_factory=dict)


def build_mass_matrix(P: Params, topo: Topology) -> np.ndarray:
    """Lumped diagonal mass matrix (returned as a 1-D array of diagonal entries).

    m at node 0, rho_t A dl_k at thread nodes (3 entries each); twist DOFs get the
    rotational inertia rho_t J l_k / 2 (standard DER choice; config via params).
    """
    diag = np.zeros(P.n_dof)
    dl = topo.node_voronoi_lengths()
    # node 0 (spider)
    diag[0:3] = P.m
    for node in range(1, topo.n_nodes):
        diag[3 * node: 3 * node + 3] = P.rho_t * P.A * dl[node]
    # twist DOFs: rho_t J l0 / 2 per edge
    twist_inertia = P.rho_t * P.J * P.l0 / 2.0
    diag[3 * topo.n_nodes:] = twist_inertia
    return diag


def _residual_and_jacobian(P, topo, frames_state, M_diag, xi, xi_n, xi_dot_n,
                           dt, field, flow, q, t):
    """Assemble Eq. (eq:residual) and Eq. (eq:jacobian) at the current iterate."""
    X = topo.positions(xi)
    thetas = topo.thetas(xi)

    # Alg.1 line 6: update reference frames by parallel transport; compute
    #               material frames from xi^(k)
    _d1, _d2, m1, m2, ref_twist = frames_state.evaluate(X, thetas)

    # elastic energy gradient and Hessian
    _E, grad_el, H_el = elastic_energy_grad_hess(P, topo, X, m1, m2, ref_twist, thetas)

    # Alg.1 line 7: v^(k) <- (xi^(k) - xi^n)/dt
    v = (xi - xi_n) / dt

    # Alg.1 line 8: f_ext <- W + F_v(xi^(k), v^(k), u) + F_r(xi^(k)) + F_l(xi^(k))
    f_ext = forces.assemble_fext(P, topo, xi, v, field, flow, t, q)

    # Alg.1 line 9: assemble residual F (Eq. (eq:residual)) and Jacobian J (Eq. (eq:jacobian))
    inertial = (M_diag / dt) * ((xi - xi_n) / dt - xi_dot_n)  # Eq. (eq:residual)
    F = inertial + grad_el - f_ext

    C = forces.resistance_matrix(P, topo, xi)                 # d F_v/d xi = -C/dt
    Jext = forces.external_position_jacobian(P, topo, xi, field, q)  # d(F_l+F_r)/d xi
    # d F_v/d xi from the spatial dependence of u (zero for Zero/Uniform flow).
    Jflow = forces.viscous_flow_jacobian(P, topo, xi, flow, t)
    M_over_dt2 = sp.diags(M_diag / dt ** 2)
    J = M_over_dt2 + H_el + C / dt - Jext - Jflow             # Eq. (eq:jacobian)
    return F, J.tocsr()


def _solve(J, F, dense: bool):
    """Linear solve J dxi = F (dense only when the uniform Coulomb Jacobian is on)."""
    if dense:
        return np.linalg.solve(J.toarray(), F)
    return spla.spsolve(J.tocsc(), F)


def _apply_dirichlet(F, J, fixed_dofs):
    """Impose dxi = 0 on ``fixed_dofs`` (Dirichlet) via row replacement.

    Sets F[d] = 0 and J row d to the unit row e_d, so the Newton solve returns
    dxi[d] = 0 and those DOFs stay at their initial (clamped) value.
    """
    if not fixed_dofs:
        return F, J
    F = F.copy()
    F[fixed_dofs] = 0.0
    J = J.tolil()
    for d in fixed_dofs:
        J.rows[d] = [d]
        J.data[d] = [1.0]
    return F, J.tocsr()


def simulate(P: Params, field=None, flow=None, xi0=None, xi_dot0=None,
             progress: bool = False, fixed_dofs=None) -> Trajectory:
    """Run Algorithm 1 and return the recorded trajectory.

    ``fixed_dofs`` (optional): DOF indices held fixed (Dirichlet) throughout the
    run -- used by the clamped-release mode to pin the spider node.
    """
    P.validate()
    # Clamped release (Alg.2 line 4): only when nothing is overridden by the caller.
    if (P.release_mode == "clamped" and flow is None and xi0 is None
            and fixed_dofs is None):
        return _simulate_clamped_release(P, field=field, progress=progress)
    if field is None:
        field = make_field(P)
    if flow is None:
        flow = make_flow(P)

    if xi0 is None:
        xi0, xi_dot0, topo = initial_state(P)
    else:
        topo = Topology(P)
        if xi_dot0 is None:
            xi_dot0 = np.zeros(P.n_dof)

    q = forces.node_charges(P, topo)
    dense_solve = (P.charge_model == "uniform")

    # Alg.1 line 2: compute lumped mass matrix M
    M_diag = build_mass_matrix(P, topo)

    frames_state = FrameState(topo, topo.positions(xi0))

    # Alg.1 line 1: n <- 0, t <- 0, dt <- dt0, c <- 0  (c: consecutive successes)
    n = 0
    t = 0.0
    dt = P.dt0
    c = 0

    xi_n = xi0.copy()
    xi_dot_n = xi_dot0.copy()

    # output buffers
    out_t: list[float] = []
    out_x: list[np.ndarray] = []
    out_v: list[np.ndarray] = []
    out_theta: list[np.ndarray] = []

    # diagnostics (used by the Stage A pilot cost report)
    diag = {"newton_failures": 0, "max_abs_u": 0.0}
    accepted_dts: list[float] = []

    def record(t_val, xi_val, xidot_val):
        out_t.append(t_val)
        X_rec = topo.positions(xi_val)
        out_x.append(X_rec.copy())
        out_v.append(topo.positions(xidot_val).copy())
        out_theta.append(topo.thetas(xi_val).copy())
        u_rec = flow.u(X_rec, t_val)
        if u_rec.size:
            diag["max_abs_u"] = max(diag["max_abs_u"], float(np.max(np.abs(u_rec))))

    record(t, xi_n, xi_dot_n)
    next_output = P.output_dt

    # steady-state / stopping bookkeeping
    zdot0_prev = xi_dot_n[2]
    steady_since = None
    outcome = {"status": "timeout", "exit_time": None, "entangled": False,
               "steady_state": False}

    # Alg.1 line 3: while t < t_end do
    while t < P.t_end:
        # Alg.1 line 4: xi^(0) <- xi^n; k <- 0  (Newton initial guess)
        xi_k = xi_n.copy()
        k = 0
        Fnorm = np.inf

        # Alg.1 line 5: repeat
        while True:
            F, J = _residual_and_jacobian(P, topo, frames_state, M_diag, xi_k,
                                          xi_n, xi_dot_n, dt, field, flow, q, t + dt)
            F, J = _apply_dirichlet(F, J, fixed_dofs)
            # Alg.1 line 10: solve J dxi = F; xi^(k+1) <- xi^(k) - dxi; k <- k+1
            dxi = _solve(J, F, dense_solve)
            xi_k = xi_k - dxi
            k += 1
            Fnorm = np.sum(np.abs(F))  # ||F||_1
            # Alg.1 line 11: until ||F||_1 < eps or k = K
            if Fnorm < P.eps or k >= P.K:
                break

        # Alg.1 line 12: if ||F||_1 >= eps then  (Newton failed)
        if Fnorm >= P.eps:
            diag["newton_failures"] += 1
            if not P.adaptive_dt:
                raise RuntimeError(
                    f"Newton failed at t={t:.6e} with fixed dt={dt:.3e} "
                    f"(||F||_1={Fnorm:.3e}, k={k})."
                )
            # Alg.1 line 13: dt <- dt/10; c <- 0; retry the step
            dt = dt / 10.0
            c = 0
            if dt < P.dt_min:
                raise RuntimeError(
                    f"Time step underflow: dt={dt:.3e} < dt_min={P.dt_min:.3e} "
                    f"at t={t:.6e} (Newton failed to converge)."
                )
            continue
        else:
            # Alg.1 line 14: else
            # Alg.1 line 15: xi^{n+1} <- xi^(k); xi_dot^{n+1} <- (xi^{n+1}-xi^n)/dt;
            #                t <- t + dt; n <- n + 1
            xi_np1 = xi_k
            xi_dot_np1 = (xi_np1 - xi_n) / dt
            frames_state.commit(topo.positions(xi_np1), topo.thetas(xi_np1))
            accepted_dts.append(dt)
            t = t + dt
            n = n + 1

            # Alg.1 line 16: c <- c+1; if c = 10 then dt <- min(10 dt, dt_max), c <- 0
            if P.adaptive_dt:
                c = c + 1
                if c == 10:
                    dt = min(10.0 * dt, P.dt_max)
                    c = 0

            # Alg.1 line 17: save (t, xi^n) at fixed output interval (HDF5)
            if t + 1e-12 >= next_output:
                record(t, xi_np1, xi_dot_np1)
                next_output += P.output_dt

            # Alg.1 line 18: if |zdot_0^n - zdot_0^{n-1}| / |zdot_0^n| < delta over
            #                a window t_w then
            zdot0 = xi_dot_np1[2]
            denom = abs(zdot0) if abs(zdot0) > 0 else 1.0
            rel = abs(zdot0 - zdot0_prev) / denom
            if rel < P.delta:
                if steady_since is None:
                    steady_since = t - dt
            else:
                steady_since = None

            # Alg.2 stopping rule: z_0 >= h -> "rise" / z_0 <= 0 -> "fall"
            if P.use_alg2_stopping:
                z0 = xi_np1[2]
                if z0 >= P.h:
                    outcome.update(status="rise", exit_time=t)
                    xi_n, xi_dot_n = xi_np1, xi_dot_np1
                    break
                if z0 <= 0.0:
                    outcome.update(status="fall", exit_time=t)
                    xi_n, xi_dot_n = xi_np1, xi_dot_np1
                    break

            if (P.stop_on_steady and steady_since is not None
                    and (t - steady_since) >= P.t_w):
                # Alg.1 line 19: return trajectory, steady-state flag = true
                outcome.update(status="steady", exit_time=t, steady_state=True)
                xi_n, xi_dot_n = xi_np1, xi_dot_np1
                break
            # Alg.1 line 20-21: end if / end if

            zdot0_prev = zdot0
            xi_n = xi_np1
            xi_dot_n = xi_dot_np1

            if progress and n % 50 == 0:
                print(f"  step n={n} t={t:.5e} dt={dt:.2e} zdot0={zdot0:.4e} "
                      f"Newton k={k}")

    # Alg.1 line 22: end while
    # Alg.1 line 23: return trajectory, steady-state flag = false (if not set above)

    # ensure the final state is recorded
    if not out_t or out_t[-1] != t:
        record(t, xi_n, xi_dot_n)

    # entanglement flag: threads in contact, d_min < entangle_contact_factor * r
    outcome["entangled"] = _entangled(P, topo, xi_n)

    if accepted_dts:
        diag["mean_dt"] = float(np.mean(accepted_dts))
        diag["min_dt"] = float(np.min(accepted_dts))
    else:
        diag["mean_dt"] = float("nan")
        diag["min_dt"] = float("nan")
    diag["n_steps"] = len(accepted_dts)
    outcome["diag"] = diag

    traj = Trajectory(
        t=np.array(out_t),
        x=np.stack(out_x),
        v=np.stack(out_v),
        theta=np.stack(out_theta),
        edges=topo.edges.copy(),
        thread_id=topo.thread_id.copy(),
        q_node=q.copy(),
        outcome=outcome,
    )
    return traj


def _simulate_clamped_release(P: Params, field=None, progress: bool = False) -> Trajectory:
    """Clamped release (Alg.2 line 4): hold node 0 fixed at z0 in still air until
    steady, then release at t=0 into the configured flow.

    Phase 1 pins the spider's 3 position DOFs (Dirichlet) with ZeroFlow and runs to
    the steady-state test (default), or for a fixed duration when
    ``P.clamp_relax_time`` is set (>0). Phase 2 restarts from that clamped shape
    (at rest), releases node 0, switches on the configured flow, and applies the
    run's stopping rules with t reset to 0.
    """
    from dataclasses import replace
    from .fields import ZeroFlow

    if field is None:
        field = make_field(P)

    # --- Phase 1: clamped, still air ---
    relax_t = P.clamp_relax_time
    if relax_t is not None and float(relax_t) > 0.0:
        # Fixed-duration still-air clamp (opt-in; default path unchanged).
        P1 = replace(P, flow_model="zero", release_mode="free",
                     use_alg2_stopping=False, stop_on_steady=False,
                     t_end=float(relax_t))
    else:
        P1 = replace(P, flow_model="zero", release_mode="free",
                     use_alg2_stopping=False, stop_on_steady=True)
    xi0, xi_dot0, topo = initial_state(P1)
    traj1 = simulate(P1, field=field, flow=ZeroFlow(), xi0=xi0,
                     xi_dot0=xi_dot0, fixed_dofs=[0, 1, 2], progress=progress)

    # clamped shape -> full DOF vector, released from rest
    X_rel = traj1.x[-1]
    theta_rel = traj1.theta[-1]
    xi_release = np.concatenate([X_rel.reshape(-1), theta_rel])
    xi_dot_release = np.zeros(P.n_dof)

    # --- Phase 2: released, configured flow, run's stopping rules, t reset ---
    P2 = replace(P, release_mode="free")
    traj2 = simulate(P2, field=field, flow=None, xi0=xi_release,
                     xi_dot0=xi_dot_release, progress=progress)
    # make_flow(P2) filled P2.turb; propagate so the caller's Params (and HDF5
    # /turb writer) see the Stage B modes / renorm_factor.
    P.turb = dict(P2.turb)
    traj2.outcome["clamped_release"] = True
    traj2.outcome["phase1_steady"] = bool(traj1.outcome.get("steady_state", False))
    traj2.outcome["phase1_t_exit"] = float(traj1.t[-1])
    traj2.outcome["clamp_relax_time"] = (
        float(relax_t) if relax_t is not None and float(relax_t) > 0.0 else None
    )
    return traj2


def _entangled(P: Params, topo: Topology, xi: np.ndarray) -> bool:
    """Flag entanglement iff d_min < entangle_contact_factor * r (default 2r)."""
    if topo.N < 2:
        return False
    dmin = observables.min_interthread_distance(topo, topo.positions(xi))
    return bool(dmin < P.entangle_contact_factor * P.r)
