"""Command-line interface (cli.py).

Example:
    ballooning --N 2 --Nt 100 --field chamber --charge tip --t-end 0.3 \
        --out run.h5
"""

from __future__ import annotations

import argparse

from .params import Params
from .integrator import simulate
from .io_hdf5 import write_trajectory
from .geometry import Topology
from . import observables


def build_params(args) -> Params:
    return Params(
        N=args.N,
        N_t=args.Nt,
        L=args.L,
        Q_t=args.Qt,
        field_model=args.field,
        charge_model=args.charge,
        flow_model=args.flow,
        t_end=args.t_end,
        dt0=args.dt0,
        dt_max=args.dt_max,
        output_dt=args.output_dt,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DER ballooning spider (Algorithm 1)")
    p.add_argument("--N", type=int, default=1, help="number of threads")
    p.add_argument("--Nt", type=int, default=100, help="edges per thread")
    p.add_argument("--L", type=float, default=0.5, help="thread length [m]")
    p.add_argument("--Qt", type=float, default=1.28e-9, help="thread charge [C]")
    p.add_argument("--field", default="chamber",
                   choices=["gorham", "chamber", "constant"])
    p.add_argument("--charge", default="tip", choices=["tip", "uniform"])
    p.add_argument("--flow", default="zero", choices=["zero", "uniform"])
    p.add_argument("--t-end", type=float, default=0.3, help="final time [s]")
    p.add_argument("--dt0", type=float, default=1e-4)
    p.add_argument("--dt-max", type=float, default=1e-2)
    p.add_argument("--output-dt", type=float, default=1e-3)
    p.add_argument("--out", default="run.h5", help="output HDF5 path")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    P = build_params(args)
    traj = simulate(P, progress=args.progress)
    write_trajectory(args.out, P, traj)

    topo = Topology(P)
    obs = observables.summary(P, topo, traj.x[-1], traj.v[-1])
    print(f"Wrote {args.out}")
    print(f"  outcome : {traj.outcome.get('status')}  "
          f"exit_time={traj.outcome.get('exit_time')}")
    print(f"  V (zdot_0) = {obs['V']:.6e} m/s")
    print(f"  R          = {obs['R']:.6e} m")
    print(f"  theta_L    = {obs['theta_L']:.6e} rad")
    print(f"  d_min      = {obs['d_min']:.6e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
