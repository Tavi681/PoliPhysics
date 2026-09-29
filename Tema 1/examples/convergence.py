"""Discretization convergence for the N=2, Fbar_l = 2 case.

Prints vbar_t, R/L and CPU time for N_t in {25, 50, 100, 200}.

Usage:
    python examples/convergence.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ballooning.studies import run_normalized


def main() -> None:
    print("N=2, E=7.41e3 V/m, Fbar_l=2 (Habchi & Jawed):")
    print(f"    {'N_t':>5} {'vbar_t':>10} {'R/L':>12} {'CPU [s]':>10} {'status':>10}")
    for N_t in (25, 50, 100, 200):
        out = run_normalized(N_t)
        print(f"    {N_t:>5} {out['vbar_t']:>10.4f} {out['R_over_L']:>12.6e} "
              f"{out['runtime_s']:>10.2f} {out['status']:>10}")


if __name__ == "__main__":
    main()
