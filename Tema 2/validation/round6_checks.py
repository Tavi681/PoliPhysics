"""Round 6 diagnostics: A/B consistency and star-S eta at arrest (items 1–2).

Run:  python -m validation.round6_checks
"""

from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

import numpy as np

from netsim.materials import get_material
from netsim.mmin import (MminConfig, analytical_s0, evaluate, pass_from_info,
                         minimum_mass)
from netsim.simulate import build_net, simulate_config
from validation.export_paper import _mmin_cfg


def item1_star_ring_S():
    """Star+ring S, M=1 kg, v0=15 m/s at s_Bany worst B point."""
    print("=" * 72)
    print("Item 1: star+ring S, M=1, v0=15 — A/B at s_Bany")
    cfg = _mmin_cfg("S", 1.0, 15.0, net_kind="star+ring", n_s=10)
    m1 = build_net(cfg).net_mass(get_material("S").rho, 1.0)
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    mmA = MminConfig(criterion="A", tol=0.01, impact_points=pts, n_procs=1)
    rA = minimum_mass(cfg, mmA)
    mmB = MminConfig(criterion="B_any", tol=0.01, impact_points=pts,
                     n_procs=1, n_scan=24)
    rB = minimum_mass(cfg, mmB, s_cap=rA.s_min)
    print(f"  m1={m1*1e3:.6g} g  s0={analytical_s0(cfg):.6g}")
    print(f"  A:  s={rA.s_min:.6g} m={rA.m_min*1e3:.6g} g  worst={rA.worst_point} "
          f"per_point={{{', '.join(f'{p}:{s:.4f}' for p,s in rA.per_point.items())}}}")
    print(f"  Bany: s={rB.s_min:.6g} m={rB.m_min*1e3:.6g} g  worst={rB.worst_point} "
          f"n_failed={rB.n_failures}")
    print(f"  (CSV confusion: mA from A-worst, n_broken from B-worst; "
          f"points differ → ratio>1 with n_broken=0 is not A rejecting arrest.)")

    point = rB.worst_point
    s = rB.s_min
    cfg2 = replace(cfg)
    cfg2.numerics = replace(cfg.numerics, area_scale=float(s))
    cfg2.drone = replace(cfg.drone, p=point)
    res = simulate_config(cfg2, write=False)
    traj = res.trajectory
    dr = np.asarray(traj.drone[-1], dtype=float)
    vz_d = float(dr[5]) if dr.size >= 6 else float("nan")
    print(f"  Run at s_Bany={s:.6g} point={point}:")
    print(f"    arrested={res.arrested} outcome={res.outcome!r} "
          f"t_final={traj.t[-1]:.6g} t_end_cfg={cfg.numerics.t_end}")
    print(f"    n_failed={res.n_failures} perforated="
          f"{res.outcome == 'perforated'} R_d={res.R_d:.6g}")
    print(f"    drone_xy_arrest=({res.drone_x_arrest:.6g}, "
          f"{res.drone_y_arrest:.6g}) vz_final={vz_d:.6g}")
    print(f"    arrest rule fired: outcome=='arrested' "
          f"(mid-run vz reversal or timeout-after-vz-reversal)")
    info = evaluate(cfg, s, point, mmB, pts.index(point))
    for crit in ("A", "B_any", "B_loc"):
        ok, reason = pass_from_info(info, crit, 0.5, 10)
        print(f"    verdict {crit}: pass={ok} reason={reason!r}")
    print("  Arrest rule: same integrator path for A and B "
          "(timeout after vz reversal ⇒ arrested); same t_end=0.5 s.")


def item2_star_S():
    """Star S, M=1, v0=15: eta just above mA at each impact point."""
    print("=" * 72)
    print("Item 2: star S, M=1, v0=15 — eta / lateral / contact at mA+")
    cfg = _mmin_cfg("S", 1.0, 15.0, net_kind="star", n_s=10)
    m = get_material("S")
    m1 = build_net(cfg).net_mass(m.rho, 1.0)
    Ekin = 0.5 * 1.0 * 15.0 ** 2
    pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)]
    mmA = MminConfig(criterion="A", tol=0.01, impact_points=pts, n_procs=1)
    rA = minimum_mass(cfg, mmA)
    print(f"  mA={rA.m_min*1e3:.6g} g  worst={rA.worst_point}  "
          f"mA/Ekin={rA.m_min*1e3/Ekin:.6g} g/J")
    print(f"  per_point mA_g: "
          f"{ {p: round(s*m1*1e3, 4) for p, s in rA.per_point.items()} }")
    print("  (Round-5 CSV labelled worst_p=(0.25,0) from B; mA is from "
          f"{rA.worst_point}. Effective eta=Ekin/(mA e_mat)="
          f"{Ekin/(rA.m_min*m.e_mat):.4f} at a={rA.worst_point[0]}.)")
    rd = cfg.drone.r_d
    for p in pts:
        s = rA.per_point[p] * 1.01
        cfg2 = replace(cfg)
        cfg2.numerics = replace(cfg.numerics, area_scale=float(s))
        cfg2.drone = replace(cfg.drone, p=p)
        res = simulate_config(cfg2, write=False)
        traj = res.trajectory
        m_net = s * m1
        i = -1
        eta = ((float(traj.energy[i, 2]) + float(traj.energy[i, 3])
                - traj.U_prestress) / (m_net * m.e_mat))
        xd = np.asarray(traj.drone[i][:3], dtype=float)
        x = traj.x[i]
        d_hub = float(np.linalg.norm(x[0] - xd))
        dists = np.linalg.norm(x - xd[None, :], axis=1)
        touch = dists < rd - 1e-9
        n_touch = int(np.sum(touch))
        hub_touch = bool(touch[0]) if touch.size else False
        print(f"  p={p}: s={s:.5f} m={s*m1*1e3:.4f}g arrested={res.arrested} "
              f"n_fail={res.n_failures}")
        print(f"    eta_arrest={eta:.6g} drone_xy="
              f"({res.drone_x_arrest:.4f},{res.drone_y_arrest:.4f}) "
              f"d_hub={d_hub:.4f} hub_touch={hub_touch} "
              f"n_nodes_in_sphere={n_touch}")
        if p == (0.25, 0.0):
            # Other radials: nodes with |y| large or x not on radial 0.
            other = np.nonzero(touch & (np.arange(len(touch)) != 0))[0]
            print(f"    other touching node ids (excl hub)={other[:12]}")


def main():
    item1_star_ring_S()
    item2_star_S()
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
