"""Validation 5a: dynamic overload / dynamic amplification factor (DAF).

A flat N-star is loaded by a constant hub force giving a static radial strain
``eps0``; one radial is then removed and the remaining radials' response is
measured, statically (``eps_s``) and dynamically (``eps_m``). Two regimes are
covered: the linear material D (small ``eps0``) and the cubic-dominated material
S (``eps0 = 0.8 eps_b``). The effective exponent ``n_eff = d ln F / d ln w`` is
reported at the operating point, and the dynamic amplification is swept over a
finite release time ``t_f / T_n`` in {0, 0.25, 0.5, 1, 2, 4}.

Run:  python -m validation.test_daf
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/netsim_mpl")

from netsim.materials import get_material
from netsim.overload import run_overload

# Operating strain per regime: D linear (small), S cubic (0.8 eps_b).
REGIMES = {
    "D": 0.01,                         # linear regime
    "S": round(0.8 * get_material("S").eps_b, 4),  # 0.24, cubic-dominated
}
N_LIST = (4, 8, 16)
TF_OVER_TN = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
M_HUB = 0.01


def main():
    print("Dynamic overload test (remove one radial of an N-star)\n")
    print("Instantaneous removal (t_f = 0):")
    print(f"{'mat':>3} {'eps0':>6} {'N':>3} {'eps0_sim':>8} {'eps_s/eps0':>10} "
          f"{'eps_m/eps0':>10} {'DAF':>6} {'n_eff':>6} {'eps_m>eps_b':>11}")
    for mat_name, eps0 in REGIMES.items():
        mat = get_material(mat_name)
        for N in N_LIST:
            r = run_overload(N, mat, eps0=eps0, m_hub=M_HUB, n_s=2,
                             t_f_over_Tn=0.0)
            over = (r.eps_m > mat.eps_b)
            print(f"{mat_name:>3} {eps0:>6.3f} {N:>3} {r.eps0_sim:>8.4f} "
                  f"{r.eps_s/eps0:>10.3f} {r.eps_m/eps0:>10.3f} {r.daf:>6.3f} "
                  f"{r.n_eff:>6.2f} {str(over):>11}")

    print("\nFinite release time (N = 8):")
    print(f"{'mat':>3} {'t_f/T_n':>8} {'eps_m/eps0':>10} {'DAF':>6}")
    for mat_name, eps0 in REGIMES.items():
        mat = get_material(mat_name)
        for tf in TF_OVER_TN:
            r = run_overload(8, mat, eps0=eps0, m_hub=M_HUB, n_s=2,
                             t_f_over_Tn=tf, dyn_periods=16.0)
            print(f"{mat_name:>3} {tf:>8.2f} {r.eps_m/eps0:>10.3f} {r.daf:>6.3f}")

    print("\nNotes:")
    print("- DAF (= eps_m/eps_s) > 1 for instantaneous removal and decreases")
    print("  toward 1 as the release time t_f grows (quasi-static limit).")
    print("- The stiffening material S shows a smaller DAF than the linear D.")
    print("- eps_m can exceed eps_b (remaining radials would then fail): this is")
    print("  the cascade-onset condition; failure itself is not modelled here.")


if __name__ == "__main__":
    main()
