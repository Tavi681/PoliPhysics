"""Validation 1 (script): material-law numbers for S and D.

Reproduces the objective's material table:
    S: sigma_b=1.20 GPa, e_mat=103.8 kJ/kg, c_L0=1240, c_L(eb)=1754, c_T(eb)=843
    D: sigma_b=3.52 GPa, e_mat=57.8 kJ/kg, c_L0=10622
plus the tangent-speed correction c_tan(eb) used for the time step.
"""

from netsim.materials import get_material


def main():
    print(f"{'mat':>3} {'sig_b[GPa]':>10} {'e_mat[kJ/kg]':>12} "
          f"{'cL0':>7} {'cL(eb)':>7} {'cT(eb)':>7} {'ctan(eb)':>8}")
    for name in ("S", "D"):
        m = get_material(name)
        print(f"{name:>3} {m.sigma_b/1e9:>10.3f} {m.e_mat/1e3:>12.1f} "
              f"{m.c_L0:>7.0f} {float(m.c_L(m.eps_b)):>7.0f} "
              f"{float(m.c_T(m.eps_b)):>7.0f} {m.c_tan_max:>8.0f}")


if __name__ == "__main__":
    main()
