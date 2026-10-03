"""Round-3 diagnostics: report numbers for each item in cursor_prompt_round3_en.md.

Run:  python -m validation.round3_checks
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
                           NumericsConfig, ContactConfig, OutputConfig)
from netsim.simulate import simulate_config, build_net
from netsim.mmin import MminConfig, monotonicity_scan, evaluate, analytical_s0
from netsim.overload import run_overload, daf_sdof_nonlinear
from netsim.topology import star_with_rings


def item1_criterion_B():
    print("\n=== Item 1: criterion B diagnostics ===")
    from validation.export_paper import _mmin_cfg
    cfg = _mmin_cfg("D", 1.0, 15.0)
    mm = MminConfig(criterion="B", impact_points=[(0.5, 0.0)])
    sc = monotonicity_scan(cfg, mm, (0.5, 0.0), n=10)
    pattern = "".join("T" if p else "F" for p in sc["pattern"])
    print(f"D criterion B pattern (shared A-scale s_hi={sc['s_hi']:.4g}): {pattern}")
    print(f"monotone={sc['monotone']}")
    for d in sc["details"]:
        if not d["passed"]:
            print(f"  F s={d['s']:.4g}: reason={d['fail_reason']} "
                  f"n_failed={d['n_failures']} R_d={d['R_d']:.4g} "
                  f"outcome={d['outcome']} Eerr={d['energy_error']:.3e}")
    # Also A pattern on the same scale.
    mmA = MminConfig(criterion="A", impact_points=[(0.5, 0.0)])
    scA = monotonicity_scan(cfg, mmA, (0.5, 0.0), n=10, s_hi=sc["s_hi"])
    print(f"D criterion A pattern (same s_hi): "
          f"{''.join('T' if p else 'F' for p in scA['pattern'])}")
    # Cross-check: every A-pass must be a B-pass.
    for a_ok, b_ok, s in zip(scA["pattern"], sc["pattern"], sc["s_values"]):
        if a_ok and not b_ok:
            print(f"  BUG: A passes but B fails at s={s:.4g}")
    return sc


def item3_overload():
    print("\n=== Item 3: dynamic overload (constrained vs free hub) ===")
    for mat_name, eps0 in (("D", 0.01),
                           ("S", round(0.8 * get_material("S").eps_b, 4))):
        mat = get_material(mat_name)
        print(f"\n--- material {mat_name}, eps0={eps0} ---")
        print(f"  n_eff_local = dlnT/dlnε @ eps0 = "
              f"{((mat.E0*eps0 + 3*mat.b*eps0**3)/(mat.E0*eps0 + mat.b*eps0**3)):.4g}")
        for N in (4, 8, 16):
            for constr in (True, False):
                r = run_overload(N, mat, eps0=eps0, m_hub=0.01, n_s=2,
                                 constrain_hub_z=constr)
                tag = "z-only" if constr else "free "
                print(f"  N={N} {tag}: eps_s/eps0={r.eps_s_over_eps0:.4f} "
                      f"eps_m/eps0={r.eps_m_over_eps0:.4f} "
                      f"daf=eps_m/eps_s={r.daf:.4f} "
                      f"max_radial={r.max_strain_radial} "
                      f"drift_xy={r.hub_drift_xy:.3e} "
                      f"m_thread/m_hub={r.thread_to_hub_mass:.3e} "
                      f"n_eff(F)={r.n_eff:.3f}")
        # Convergence at N=8: n_s=8 and m_hub×10.
        for n_s, m_hub in ((8, 0.01), (2, 0.1)):
            r = run_overload(8, mat, eps0=eps0, m_hub=m_hub, n_s=n_s,
                             constrain_hub_z=True)
            print(f"  N=8 z-only n_s={n_s} m_hub={m_hub}: "
                  f"eps_m/eps0={r.eps_m_over_eps0:.4f} daf={r.daf:.4f}")


def item4_energy_D():
    print("\n=== Item 4: material D energy error scan ===")
    rows = []
    for kcf in (2.0, 4.0, 8.0):
        for C in (0.5, 0.25):
            t0 = time.time()
            cfg = SimConfig(
                material=MaterialConfig(name="D"),
                net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
                drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.5, 0.0)),
                numerics=NumericsConfig(n_s=20, C=C, t_end=0.25, dt_out=1e-3,
                                        use_numba=False),
                contact=ContactConfig(mode="gripped", k_c=1e7,
                                      k_c_mode="relative", k_c_factor=kcf,
                                      penetration_guard="reduce_dt"),
                output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
            )
            res = simulate_config(cfg, write=False)
            cpu = time.time() - t0
            row = dict(k_c_factor=kcf, C=C, energy_error=res.energy_error,
                       w_max=res.w_max, n_failed=res.n_failures, eta=res.eta,
                       cpu=cpu, outcome=res.outcome)
            rows.append(row)
            print(f"  kcf={kcf} C={C}: Eerr={res.energy_error:.4e} "
                  f"wmax={res.w_max:.4f} n_failed={res.n_failures} "
                  f"eta={res.eta:.4f} cpu={cpu:.2f}s outcome={res.outcome}")
            # Energy-component history: locate where the error grows.
            e = res.trajectory.energy
            Et = e.sum(axis=1)
            E0 = Et[0]
            i_max = int(np.argmax(np.abs(Et - E0)))
            print(f"    max|ΔE| at t={res.trajectory.t[i_max]:.4g}: "
                  f"KE_d={e[i_max,0]:.4g} KE_n={e[i_max,1]:.4g} "
                  f"Uel={e[i_max,2]:.4g} Ufail={e[i_max,3]:.4g} "
                  f"Uc={e[i_max,4]:.4g} Ucap={e[i_max,5]:.4g}")
    # Physical-output sensitivity to k_c_factor at C=0.5.
    subset = [r for r in rows if r["C"] == 0.5]
    nf = [r["n_failed"] for r in subset]
    print(f"  n_failed vs k_c_factor @ C=0.5: {list(zip([r['k_c_factor'] for r in subset], nf))}")
    if len(set(nf)) > 1:
        print("  WARNING: n_failed changes with k_c_factor -> contact not "
              "converged; D dynamic results cannot be used yet.")
    else:
        print("  n_failed insensitive to k_c_factor (contact converged on "
              "failure count).")
    return rows


def item5_mmin_example():
    print("\n=== Item 5: minimum-mass example vs quasi-static ===")
    import offc
    # Example from the prompt: S star, criterion A, m_min≈10.3 g, worst (0.5,0).
    # Recover M, v0 from a standard reference case.
    M, v0, R = 1.0, 15.0, 1.0
    mat = get_material("S")
    eps_p = 0.0
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    cfg = SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=R, eps_p=eps_p, A_hat=1e-6),
        drone=DroneConfig(M=M, r_d=0.15, v0=v0, p=(0.5, 0.0)),
        numerics=NumericsConfig(n_s=10, C=0.5, t_end=0.4, dt_out=5e-3,
                                use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )
    from netsim.mmin import minimum_mass
    mm = MminConfig(criterion="A", tol=0.1, impact_points=pts)
    r = minimum_mass(cfg, mm)
    Ekin = 0.5 * M * v0 ** 2
    print(f"  M={M} kg, v0={v0} m/s, eps_p={eps_p}, R={R} m")
    print(f"  impact points: {pts}")
    print(f"  mA_min={1e3*r.m_min:.3f} g at worst {r.worst_point} "
          f"(s_min={r.s_min:.4g})")
    # Quasi-static prediction at the worst point.
    a = r.worst_point[0]
    if abs(a) < 1e-12:
        etaA = 1.0 - float(mat.Phi(eps_p)) / float(mat.Phi(mat.eps_b))
    else:
        fails = offc.run("S", 8, a, eps_p / mat.eps_b if mat.eps_b else 0.0)
        etaA = fails[0][2]
    mA_qs = Ekin / (etaA * mat.e_mat)
    print(f"  etaA_qs(a={a})={etaA:.4f} -> mA_qs={1e3*mA_qs:.3f} g")
    print(f"  ratio dynamic/quasi-static = {r.m_min/mA_qs:.3f}")
    # Centre vs off-centre.
    s_c = r.per_point.get((0.0, 0.0), float("nan"))
    s_o = r.per_point.get((0.5, 0.0), float("nan"))
    m1 = build_net(cfg).net_mass(mat.rho, 1.0)
    print(f"  m_centre={1e3*s_c*m1:.3f} g, m_off={1e3*s_o*m1:.3f} g, "
          f"ratio off/centre={s_o/s_c if s_c else float('nan'):.2f}")
    fails_c = offc.run("S", 8, 0.05, 0.0)  # near-centre proxy
    # True centre eta from analytic.
    eta_c = 1.0
    fails_o = offc.run("S", 8, 0.5, 0.0)
    eta_o = fails_o[0][2]
    print(f"  quasi-static eta_centre≈{eta_c:.3f}, eta_off={eta_o:.4f}, "
          f"ratio m_off/m_c qs = {eta_c/eta_o:.2f}")


def item6b_extras():
    print("\n=== Item 6b: ring position, energy at arrest, gripped floor ===")
    m = get_material("S")
    net_fixed = star_with_rings(8, 1.0, [0.5, 1.0], 0.1 * m.eps_b, material=m,
                                q_ratio=0.5, fix_radii=True)
    net_free = star_with_rings(8, 1.0, [0.5, 1.0], 0.1 * m.eps_b, material=m,
                               q_ratio=0.5, fix_radii=False)
    r_fixed = float(np.mean(np.hypot(net_fixed.nodes[1:9, 0],
                                     net_fixed.nodes[1:9, 1])))
    r_free = float(np.mean(np.hypot(net_free.nodes[1:9, 0],
                                    net_free.nodes[1:9, 1])))
    print(f"  ring radius fix_radii=True:  {r_fixed:.4f} R (want 0.5)")
    print(f"  ring radius fix_radii=False: {r_free:.4f} R (was ~0.235)")
    print(f"  family_eps (fixed, q_ratio=0.5): {net_fixed.meta.get('family_eps')}")

    # Energy at arrest in Test 4 reference (gripped, S, n_s=40).
    from validation.test4_convergence import run_case
    # Use the same case as tab_conv gripped n_s=40 via simulate_config.
    cfg = SimConfig(
        material=MaterialConfig(name="S"),
        net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.1 * m.eps_b, A_hat=1e-6),
        drone=DroneConfig(M=1.0, r_d=0.15, v0=15.0, p=(0.5, 0.0)),
        numerics=NumericsConfig(n_s=40, C=0.5, t_end=0.4, dt_out=1e-3,
                                use_numba=False),
        contact=ContactConfig(mode="gripped", k_c=1e7,
                              penetration_guard="reduce_dt"),
        output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
    )
    res = simulate_config(cfg, write=False)
    traj = res.trajectory
    e = traj.energy[-1]
    Ekin0 = 0.5 * cfg.drone.M * cfg.drone.v0 ** 2
    vd = traj.drone[-1, 3:6]
    KE_d_v = 0.5 * (cfg.drone.M) * vd[2] ** 2  # vertical only approx
    print(f"  Test4-like arrest: outcome={res.outcome} n_failed={res.n_failures} "
          f"eta={res.eta:.4f} m_net={res.m_net:.4e} kg")
    print(f"  E_kin0={Ekin0:.4f} J")
    print(f"  at arrest: KE_drone={e[0]:.4g} KE_net={e[1]:.4g} "
          f"U_el={e[2]:.4g} U_fail={e[3]:.4g} U_contact={e[4]:.4g} "
          f"U_cap={e[5]:.4g}")
    print(f"  check: eta*m*e_mat + residual KE + U_fail+U_cap+Uc "
          f"= {res.eta*res.m_net*m.e_mat + e[0]+e[1]+e[3]+e[4]+e[5]:.4f} "
          f"vs E_kin0+U_pre={Ekin0+traj.U_prestress:.4f} "
          f"(Eerr={res.energy_error:.3e})")

    # Gripped energy floor vs C.
    print("  gripped energy_error vs C:")
    for C in (0.5, 0.25, 0.125):
        cfg2 = SimConfig(
            material=MaterialConfig(name="S"),
            net=NetConfig(kind="star", N=8, R=1.0, eps_p=0.0, A_hat=1e-6),
            drone=DroneConfig(M=0.2, r_d=0.15, v0=5.0, p=(0.5, 0.0),
                              gravity=False),
            numerics=NumericsConfig(n_s=20, C=C, t_end=0.06, dt_out=2e-4,
                                    use_numba=False),
            contact=ContactConfig(mode="gripped", k_c=1e7),
            output=OutputConfig(hdf5=None, R_max=0.5, k_max=10),
        )
        r2 = simulate_config(cfg2, write=False)
        print(f"    C={C}: Eerr={r2.energy_error:.4e} Ucap={r2.trajectory.energy[-1,5]:.4e}")


def item6_smith_plateau():
    print("\n=== Item 6: Smith longitudinal-front plateau (S) ===")
    from validation.test1_smith import run_smith
    for n_res in (200, 400, 800):
        r = run_smith(500.0, material_name="S", n_resolved=n_res)
        print(f"  n_seg≈{int(r['n_resolved'])}: cL_err={100*r['front_relerr']:.2f}% "
              f"cT_err={100*r['kink_relerr']:.2f}% eps_err={100*r['eps_relerr']:.2f}% "
              f"(n_times={r['n_front_times']})")


def main():
    item1_criterion_B()
    item3_overload()
    item4_energy_D()
    item5_mmin_example()
    item6b_extras()
    item6_smith_plateau()
    print("\n=== done ===")


if __name__ == "__main__":
    main()
