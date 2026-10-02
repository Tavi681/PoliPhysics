"""Command-line interface.

Usage::

    python -m netsim.cli run config.yaml [--out run.h5] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .simulate import simulate_config

__all__ = ["main"]


def _run(args) -> int:
    cfg = load_config(args.config)
    if args.out is not None:
        cfg.output.hdf5 = args.out
    result = simulate_config(cfg, write=True)
    print("outcome      :", result.outcome)
    print("arrested     :", result.arrested)
    print("w_max        : %.6g m" % result.w_max)
    print("eta          : %.6g" % result.eta)
    print("n_failures   :", result.n_failures)
    print("R_d          : %.6g m" % result.R_d)
    print("cascade      :", result.cascade)
    print("energy_error : %.3e" % result.energy_error)
    if cfg.output.hdf5:
        print("written      :", cfg.output.hdf5)
    return 0


def _mmin(args) -> int:
    import yaml
    from .mmin import MminConfig, minimum_mass, monotonicity_scan

    cfg = load_config(args.config)
    with open(args.config, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    m = raw.get("mmin", {})
    pts = [tuple(p) for p in m.get("impact_points", [[0.0, 0.0]])]
    mmincfg = MminConfig(
        criterion=args.criterion or m.get("criterion", "A"),
        tol=m.get("tol", 0.05),
        impact_points=pts,
        s0_scale=m.get("s0_scale", 1.0),
        n_procs=args.procs or m.get("n_procs", 1),
        cache_dir=args.cache_dir or m.get("cache_dir"),
    )
    if args.scan:
        sc = monotonicity_scan(cfg, mmincfg, pts[0], n=args.scan)
        print("monotonicity scan at", sc["point"])
        for s, ok in zip(sc["s_values"], sc["pattern"]):
            print(f"  s={s:.4f}  {'PASS' if ok else 'fail'}")
        print("monotone     :", sc["monotone"])
        return 0
    res = minimum_mass(cfg, mmincfg)
    print("criterion    :", res.criterion)
    print("s_min        : %.6g" % res.s_min)
    print("m_min        : %.6g kg" % res.m_min)
    print("worst point  :", res.worst_point)
    print("n_failures   :", res.n_failures)
    print("broken segs  :", res.broken_segments)
    print("evaluations  :", res.evaluations)
    print("per point    :", {k: round(v, 5) for k, v in res.per_point.items()})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="netsim", description=__doc__)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase log verbosity")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a simulation from a YAML config")
    p_run.add_argument("config", help="path to the YAML config file")
    p_run.add_argument("--out", default=None, help="output HDF5 path")
    p_run.set_defaults(func=_run)

    p_mm = sub.add_parser("mmin", help="Algorithm 2: minimum-mass bisection")
    p_mm.add_argument("config", help="path to the YAML config file")
    p_mm.add_argument("--criterion", choices=["A", "B"], default=None)
    p_mm.add_argument("--procs", type=int, default=None, help="worker processes")
    p_mm.add_argument("--cache-dir", default=None, help="resumable HDF5 cache")
    p_mm.add_argument("--scan", type=int, default=0, metavar="N",
                      help="run an N-point monotonicity scan instead")
    p_mm.set_defaults(func=_mmin)

    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(name)s: %(message)s")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
