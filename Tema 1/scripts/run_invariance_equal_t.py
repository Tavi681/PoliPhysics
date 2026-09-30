"""Run the equal-time invariance study and patch results/meta.json.

Usage (from ``Tema 1``):
    python scripts/run_invariance_equal_t.py --dt 3e-5
    python scripts/run_invariance_equal_t.py --from-cache results/equal_t_cache --dt 3e-5
    python scripts/run_invariance_equal_t.py --smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ballooning.studies import (
    load_invariance_equal_t_cache,
    run_invariance_equal_t,
)


def _params_dict(P) -> dict:
    d = dataclasses.asdict(P)
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _eq_branch(branch: dict) -> dict:
    out = {
        "label": branch["label"],
        "per_t_dev_over_L": [float(x) for x in branch["per_t_dev_over_L"]],
        "max_shape_dev_equal_t": float(branch["max_shape_dev_equal_t"]),
        "dV_m_per_s": float(branch["dV_m_per_s"]),
        "status": {str(k): v for k, v in branch["status"].items()},
        "runtime_s": {str(k): float(v) for k, v in branch["runtime_s"].items()},
        "t_exit_s": {str(k): float(v) for k, v in branch["t_exit_s"].items()},
    }
    for key in ("t_s", "decay_time_s", "decay_definition", "t_used"):
        if key in branch:
            val = branch[key]
            out[key] = [float(x) for x in val] if key == "t_used" else val
    return out


def write_meta(eq: dict, meta_path: Path) -> Path:
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"studies": {}}
    fi = meta.setdefault("studies", {}).setdefault("fig_invariance", {})
    fi["max_shape_dev_equal_t"] = float(eq["comoving"]["max_shape_dev_equal_t"])
    fi["params_equal_t"] = {k: _params_dict(eq["params"][k])
                            for k in ("w0", "w_comoving", "w_transient")}
    fi["equal_t"] = {
        "dt": eq["dt"],
        "output_dt": eq["output_dt"],
        "t_end": eq["t_end"],
        "eps": eq["eps"],
        "K": eq["K"],
        "adaptive_dt": eq["adaptive_dt"],
        "stop_on_steady": eq["stop_on_steady"],
        "w": eq["w"],
        "t_compare": [float(t) for t in eq["t_compare"]],
        "comoving": _eq_branch(eq["comoving"]),
        "transient": _eq_branch(eq["transient"]),
    }
    base = fi.get("description", "").split(" equal_t.")[0].split(" max_shape_dev_equal_t")[0]
    fi["description"] = (
        base.rstrip() + " equal_t.comoving: fixed dt, default eps/K, w-run starts "
        "with v0=w z_hat on all nodes; equal_t.transient: both v0=0 (different "
        "relative IC), with decay_time vs Stokes t_s."
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, default=float) + "\n")
    return meta_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=None,
                    help="fixed dt (default: auto-select among candidates)")
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--output-dt", type=float, default=None)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--cache-dir", type=str, default=str(ROOT / "results" / "equal_t_cache"))
    ap.add_argument("--from-cache", type=str, default=None,
                    help="rebuild meta from cached .npz trajectories")
    ap.add_argument("--meta-out", type=str, default=str(ROOT / "results" / "meta.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="short end-to-end run (t_end=0.003, output_dt=0.001, dt=3e-5)")
    ns = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    dt = ns.dt
    t_end = ns.t_end
    output_dt = ns.output_dt
    cache_dir = ns.cache_dir
    meta_out = Path(ns.meta_out)

    if ns.smoke:
        dt = 3e-5 if dt is None else dt
        t_end = 0.003 if t_end is None else t_end
        output_dt = 0.001 if output_dt is None else output_dt
        cache_dir = str(ROOT / "results" / "equal_t_cache_smoke")
        if ns.meta_out == str(ROOT / "results" / "meta.json"):
            meta_out = ROOT / "results" / "meta_smoke.json"
        print(f"SMOKE: dt={dt} t_end={t_end} output_dt={output_dt} "
              f"cache={cache_dir} meta={meta_out}", flush=True)

    if t_end is None:
        from ballooning.studies import INVARIANCE_EQUAL_T_END
        t_end = INVARIANCE_EQUAL_T_END
    if output_dt is None:
        from ballooning.studies import INVARIANCE_EQUAL_T_OUTPUT
        output_dt = INVARIANCE_EQUAL_T_OUTPUT

    if ns.from_cache:
        if dt is None:
            raise SystemExit("--from-cache requires --dt")
        eq = load_invariance_equal_t_cache(
            ns.from_cache, dt, t_end=t_end, output_dt=output_dt,
        )
    else:
        eq = run_invariance_equal_t(
            parallel=not ns.serial, dt=dt, cache_dir=cache_dir,
            t_end=t_end, output_dt=output_dt,
        )

    # Require both branches and finite max before writing meta.
    assert "comoving" in eq and "transient" in eq
    assert np_isfinite(eq["comoving"]["max_shape_dev_equal_t"])
    assert np_isfinite(eq["transient"]["max_shape_dev_equal_t"])
    assert len(eq["t_compare"]) >= 1

    meta_path = write_meta(eq, meta_out)
    print(f"wrote {meta_path}", flush=True)
    print("comoving max", eq["comoving"]["max_shape_dev_equal_t"], flush=True)
    print("transient max", eq["transient"]["max_shape_dev_equal_t"],
          "decay", eq["transient"]["decay_time_s"], "t_s", eq["transient"]["t_s"],
          flush=True)

    if ns.smoke:
        # Round-trip from cache without re-integrating.
        eq2 = load_invariance_equal_t_cache(cache_dir, eq["dt"],
                                            t_end=eq["t_end"], output_dt=eq["output_dt"])
        assert abs(eq2["comoving"]["max_shape_dev_equal_t"]
                   - eq["comoving"]["max_shape_dev_equal_t"]) < 1e-15
        meta2 = json.loads(meta_path.read_text())
        assert "comoving" in meta2["studies"]["fig_invariance"]["equal_t"]
        assert "transient" in meta2["studies"]["fig_invariance"]["equal_t"]
        print("SMOKE OK", flush=True)
    return 0


def np_isfinite(x) -> bool:
    import math
    return math.isfinite(float(x))


if __name__ == "__main__":
    raise SystemExit(main())
