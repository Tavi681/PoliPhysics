"""Run a single ballooning simulation, save an HDF5 file and print observables.

Usage:
    python examples/run_single.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ballooning.params import Params
from ballooning.integrator import simulate
from ballooning.io_hdf5 import write_trajectory
from ballooning.geometry import Topology
from ballooning import observables


def main() -> None:
    P = Params(
        N=2,
        N_t=50,
        L=0.5,
        Q_t=3e-9,
        charge_model="tip",
        field_model="constant",
        E_constant=2500.0,
        flow_model="zero",
        t_end=3.0,
        dt0=1e-4,
        dt_max=1e-2,
        output_dt=5e-3,
        delta=1e-6,
        t_w=0.1,
    )
    traj = simulate(P, progress=True)

    out = "run_single.h5"
    write_trajectory(out, P, traj)

    topo = Topology(P)
    obs = observables.summary(P, topo, traj.x[-1], traj.v[-1])
    print(f"\nSaved trajectory to {out}")
    print(f"  outcome : {traj.outcome['status']} (exit_time={traj.outcome['exit_time']})")
    print(f"  V (zdot_0) = {obs['V']:.6e} m/s")
    print(f"  R          = {obs['R']:.6e} m")
    print(f"  theta_L    = {obs['theta_L']:.6e} rad")
    print(f"  d_min      = {obs['d_min']:.6e} m")


if __name__ == "__main__":
    main()
