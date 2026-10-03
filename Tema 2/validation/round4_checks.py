"""Round-4 diagnostics: D contact on arresting cases, failure-energy booking,
lateral drone drift vs offc_free.

Run:  python -m validation.round4_checks
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

import numpy as np

_REF = str(Path(__file__).resolve().parent.parent / "ref")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

from netsim.materials import get_material
from netsim.config import (SimConfig, MaterialConfig, NetConfig, DroneConfig,
                           NumericsConfig, ContactConfig, OutputConfig,
                           KinematicConfig)
from netsim.simulate import simulate_config, build_net, simulate
from netsim.topology import from_arrays
from netsim.discretize import discretize
from netsim.mmin import MminConfig, minimum_mass, analytical_s0, evaluate
from netsim.integrator import integrate
from netsim._kernels import segment_forces

# Default relative k_c for D after item-1 convergence (set by main if it passes).
K_C_FACTOR_D_DEFAULT = 8.0


def _d_cfg(s, *, k_c_factor=8.0, C=0.5, n_s=20, t_end=0.4, M=1.0, v0=15.0):
    return SimConfig(
        material=MaterialConfig(name="D"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
        drone=DroneConfig(M=M, r_d=0.15, v0=v0, p=(0.5, 0.0)),
        numerics=NumericsConfig(n_s=n_s, C=C, t_end=t_end, dt_out=2e-3,
                                area_scale=float(s), use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7, k_c_mode="relative",
                              k_c_factor=k_c_factor,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )


def item1_failure_energy_booking():
    """Pull a single thread at constant end speed until break + 10 transit times.

    Energy balance: W_prescribed - KE - U_el - U_fail must close within 1e-3
    relative to W.
    """
    print("\n=== Item 1.3: failure-energy bookkeeping (single thread) ===")
    results = {}
    for mat_name in ("S", "D"):
        mat = get_material(mat_name)
        L = 1.0
        n_nodes = 21
        xs = np.linspace(0.0, L, n_nodes)
        nodes = np.c_[xs, np.zeros(n_nodes)]
        edges = np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)]
        anchored = np.zeros(n_nodes, dtype=bool)
        anchored[0] = True
        h = L / (n_nodes - 1)
        rest = np.full(n_nodes - 1, h)
        A = np.full(n_nodes - 1, 1e-6)
        net = from_arrays(nodes, edges, anchored, np.zeros(n_nodes - 1), A,
                          rest_length=rest, R=L)
        disc = discretize(net, mat, 1, r_d=1.0, warn_ratio=1e9)
        # Constant end speed; transit time of a longitudinal wave.
        c = mat.c_tan_max
        t_transit = L / c
        speed = 0.5 * mat.eps_b * c / 5.0  # gentle enough to resolve the break
        # Time to break roughly: end displacement ~ eps_b * L => t_b ~ eps_b L / speed
        t_b_est = mat.eps_b * L / speed
        t_end = t_b_est + 10.0 * t_transit
        drone = DroneConfig(M=1.0, r_d=1.0, v0=0.0)
        numerics = NumericsConfig(n_s=1, C=0.25, t_end=t_end,
                                  dt_out=min(t_transit / 20.0, t_end / 2000),
                                  use_numba=False)
        contact = ContactConfig(mode="frictionless", k_c=1e7)
        output = OutputConfig(hdf5=None, R_max=1e9, k_max=10 ** 9)
        end = n_nodes - 1
        kin = KinematicConfig(enabled=True, node=end, mode="velocity",
                              direction=(1.0, 0.0, 0.0), amplitude=speed)
        traj = integrate(disc, mat, drone, numerics, contact, output,
                         kinematic=kin, net_R=L)
        # Work by the prescribed end: integral F_end · v_end dt.
        # F_end = -segment force on end node = mass * a is awkward; use
        # energy bookkeeping: W = Δ(KE + U_el + U_fail) would be tautological.
        # Instead accumulate from the recorded energy change plus residual.
        # Work = final (KE + U_el + U_fail) - initial U_el, since anchors do no
        # work and the prescribed velocity injects all the energy.
        e0 = traj.energy[0]
        e1 = traj.energy[-1]
        U0 = e0[2]  # elastic
        E_final = e1[0] + e1[1] + e1[2] + e1[3]  # KE_dr(=0) + KE_net + Uel + Ufail
        W = E_final - U0  # if perfectly conserved
        # Residual of the integrator's own balance (should be tiny).
        # Relative closure: |E_tot(t) - E_tot(0) - 0| / max(|W|,1e-30) —
        # for a driven end, E_tot grows by W, so the recorded energy_error is
        # already |max ΔE|/E_norm. Better: reconstruct W from force.
        # Work by trapezoidal rule on the end-node reaction.
        W_acc = 0.0
        f_prev = None
        for i in range(traj.t.size):
            x = traj.x[i]
            intact = traj.intact[i]
            f = segment_forces(x, disc.seg_edges, disc.seg_rest_length,
                               disc.seg_A, intact, mat.E0, mat.b)
            if f_prev is not None:
                dt_i = float(traj.t[i] - traj.t[i - 1])
                F_avg = -0.5 * (f_prev[end, 0] + f[end, 0])
                W_acc += float(F_avg * speed * dt_i)
            f_prev = f
        KE = float(e1[1])
        Uel = float(e1[2])
        Ufail = float(e1[3])
        residual = W_acc - (KE + Uel + Ufail - U0)
        rel = abs(residual) / max(abs(W_acc), 1e-30)
        broke = traj.failures.shape[0] > 0
        results[mat_name] = dict(W=W_acc, KE=KE, Uel=Uel, Ufail=Ufail, U0=U0,
                                 residual=residual, rel=rel, broke=broke,
                                 n_fail=int(traj.failures.shape[0]),
                                 energy_error=traj.energy_error)
        print(f"  {mat_name}: broke={broke} n_fail={traj.failures.shape[0]} "
              f"W={W_acc:.4g} KE={KE:.4g} Uel={Uel:.4g} Ufail={Ufail:.4g} "
              f"resid={residual:.4g} rel={rel:.3e}")
        if rel < 1e-3:
            print(f"  {mat_name}: PASS (rel < 1e-3)")
        else:
            print(f"  {mat_name}: FAIL booking (rel >= 1e-3) — late-run D "
                  f"error may come from failure overshoot")
    return results


def item1_d_arrest_convergence():
    """Convergence of D on arresting cases (not perforation)."""
    print("\n=== Item 1.2: D convergence on arresting cases ===")
    # Find s^A_min and a B-passing s with failures.
    cfg0 = _d_cfg(1.0, k_c_factor=8.0, C=0.5, n_s=10, t_end=0.5)
    mmA = MminConfig(criterion="A", tol=0.1, impact_points=[(0.5, 0.0)])
    rA = minimum_mass(cfg0, mmA)
    sA = rA.s_min
    print(f"  sA_min = {sA:.4g} (mA={1e3*rA.m_min:.3f} g) at {rA.worst_point}")

    mmB = MminConfig(criterion="B", tol=0.1, impact_points=[(0.5, 0.0)],
                     n_scan=16)
    rB = minimum_mass(cfg0, mmB)
    sB = rB.s_min
    print(f"  sB_min = {sB:.4g} (mB={1e3*rB.m_min:.3f} g) n_fail={rB.n_failures}")

    s_arrest = 1.1 * sA
    s_with_fail = sB if rB.n_failures > 0 else 0.9 * sA
    # If B == A (no sacrificial regime), probe slightly below sA for failures.
    if rB.n_failures == 0:
        # Scan down for a failing-but-arrested s.
        for fac in (0.95, 0.9, 0.85, 0.8, 0.7):
            info = evaluate(cfg0, fac * sA, (0.5, 0.0),
                            MminConfig(criterion="B"), 0)
            if info["arrested"] and info["n_failures"] > 0:
                s_with_fail = fac * sA
                print(f"  probing s={s_with_fail:.4g}: arrested with "
                      f"n_fail={info['n_failures']}")
                break
        else:
            print("  WARNING: no arrested-with-failures s found below sA; "
                  "using 0.85*sA anyway")
            s_with_fail = 0.85 * sA

    cases = {"1.1_sA": s_arrest, "sB_or_fail": s_with_fail}
    table = []
    for label, s in cases.items():
        print(f"\n  --- case {label}: s={s:.4g} ---")
        print(f"  {'kcf':>4} {'C':>5} {'outcome':>11} {'nfail':>5} "
              f"{'wmax':>8} {'eta':>8} {'Eerr':>9} {'cpu':>7}")
        for kcf in (4.0, 8.0, 16.0):
            for C in (0.5, 0.25):
                t0 = time.time()
                cfg = _d_cfg(s, k_c_factor=kcf, C=C, n_s=20, t_end=0.5)
                res = simulate_config(cfg, write=False)
                cpu = time.time() - t0
                segs = sorted(int(s_) for s_ in res.trajectory.failures[:, 0]
                              ) if res.trajectory.failures.size else []
                row = dict(label=label, s=s, kcf=kcf, C=C,
                           outcome=res.outcome, n_failed=res.n_failures,
                           segs=segs, w_max=res.w_max, eta=res.eta,
                           Eerr=res.energy_error, cpu=cpu)
                table.append(row)
                print(f"  {kcf:>4.0f} {C:>5.2f} {res.outcome:>11} "
                      f"{res.n_failures:>5} {res.w_max:>8.4f} {res.eta:>8.4f} "
                      f"{res.energy_error:>9.2e} {cpu:>7.2f}")

    # Convergence check: for each case, outcome+failure set identical and
    # w_max, eta change < 1% across kcf and C.
    ok_default = True
    for label in cases:
        sub = [r for r in table if r["label"] == label]
        outcomes = {r["outcome"] for r in sub}
        fails = {tuple(r["segs"]) for r in sub}
        w = [r["w_max"] for r in sub]
        e = [r["eta"] for r in sub]
        w_spread = (max(w) - min(w)) / max(abs(np.mean(w)), 1e-30)
        e_spread = (max(e) - min(e)) / max(abs(np.mean(e)), 1e-30)
        same = (len(outcomes) == 1 and len(fails) == 1
                and w_spread < 0.01 and e_spread < 0.01)
        print(f"  {label}: outcome_set={outcomes} fail_sets={len(fails)} "
              f"dw={100*w_spread:.2f}% deta={100*e_spread:.2f}% "
              f"{'PASS' if same else 'FAIL'}")
        # Specifically check k_c_factor=8 subset.
        sub8 = [r for r in sub if r["kcf"] == 8.0]
        if len({r["outcome"] for r in sub8}) > 1:
            ok_default = False
        # Compare kcf=8 vs 16 at C=0.5
        r8 = next(r for r in sub if r["kcf"] == 8 and r["C"] == 0.5)
        r16 = next(r for r in sub if r["kcf"] == 16 and r["C"] == 0.5)
        if (r8["outcome"] != r16["outcome"] or r8["segs"] != r16["segs"]
                or abs(r8["w_max"] - r16["w_max"]) / max(r16["w_max"], 1e-30) > 0.01
                or abs(r8["eta"] - r16["eta"]) / max(abs(r16["eta"]), 1e-30) > 0.01):
            ok_default = False
            print(f"    kcf=8 vs 16 @ C=0.5 not within 1%")
        else:
            print(f"    kcf=8 vs 16 @ C=0.5 within 1% (ok for default)")

    # CPU at n_s=40.
    print("\n  CPU at n_s=40, k_c_factor=8, C=0.5, s=1.1*sA:")
    t0 = time.time()
    res = simulate_config(_d_cfg(s_arrest, k_c_factor=8.0, C=0.5, n_s=40,
                                 t_end=0.5), write=False)
    cpu40 = time.time() - t0
    print(f"  outcome={res.outcome} n_fail={res.n_failures} "
          f"wmax={res.w_max:.4f} Eerr={res.energy_error:.3e} cpu={cpu40:.2f}s")

    return dict(sA=sA, sB=sB, table=table, ok_default_kcf8=ok_default,
                cpu40=cpu40)


def item2_lateral_and_free():
    print("\n=== Item 2: lateral drift / offc_free ===")
    import offc_free
    import offc
    from validation.offcentre import run_offcentre

    print("  offc_free vs free_lateral sim (S, ep=0):")
    print(f"  {'a/R':>5} {'eta_free_ref':>12} {'eta_free_num':>12} "
          f"{'px_ref':>8} {'px_num':>8} {'eta_fixed':>10}")
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        k, w, px_ref, eta_ref = offc_free.run("S", 8, a, 0.0)
        sim = run_offcentre("S", a_over_R=a, eps_p_frac=0.0, n_s=20,
                            free_lateral=True, continue_to_B=False)
        fixed = offc.run("S", 8, a, 0.0)[0][2]
        print(f"  {a:>5.1f} {eta_ref:>12.4f} {sim['eta_A']:>12.4f} "
              f"{px_ref:>8.4f} {sim['px_fail']:>8.4f} {fixed:>10.4f}")

    # Dynamic mA_min vs M at fixed E_kin.
    print("\n  Dynamic mA_min at (0.5,0), fixed E_kin, vary M:")
    Ekin = 0.5 * 1.0 * 15.0 ** 2
    print(f"  E_kin = {Ekin:.4g} J")
    print(f"  {'M':>6} {'v0':>7} {'mA_g':>8} {'px_arr':>8}")
    for M in (0.25, 1.0, 4.0):
        v0 = math.sqrt(2.0 * Ekin / M)
        cfg = SimConfig(
            material=MaterialConfig(name="S"),
            net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
            drone=DroneConfig(M=M, r_d=0.15, v0=v0, p=(0.5, 0.0)),
            numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.5, dt_out=5e-3,
                                    use_numba=False),
            contact=ContactConfig(mode="gripped", k_c=1e7,
                                  penetration_guard="reduce_dt"),
            output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
        )
        r = minimum_mass(cfg, MminConfig(criterion="A", tol=0.1,
                                         impact_points=[(0.5, 0.0)]))
        # One eval at s_min for lateral position.
        cfg2 = cfg
        from dataclasses import replace
        cfg2 = replace(cfg, numerics=replace(cfg.numerics, area_scale=r.s_min))
        res = simulate_config(cfg2, write=False)
        print(f"  {M:>6.2f} {v0:>7.2f} {1e3*r.m_min:>8.3f} "
              f"{res.drone_x_arrest:>8.4f}")


def main():
    book = item1_failure_energy_booking()
    conv = item1_d_arrest_convergence()
    if conv["ok_default_kcf8"]:
        print(f"\n*** Using k_c_factor={K_C_FACTOR_D_DEFAULT} as default for D ***")
    else:
        print("\n*** k_c_factor=8 did NOT fully pass the 1% criterion; "
              "report numbers and do not silently claim convergence ***")
    item2_lateral_and_free()
    print("\n=== done ===")
    return book, conv


if __name__ == "__main__":
    main()
