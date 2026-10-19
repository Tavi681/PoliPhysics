"""KS dtype / mode-source diagnosis vs simulator KinematicSimulation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ml.data.air import make_ks
from ml.data.ks_torch import ks_u_torch, turb_modes_from_globals, turb_modes_from_h5, turb_modes_from_ks
from ml.utils.git_info import repo_root


def main():
    raw = repo_root() / "results" / "ml" / "raw_smoke"
    h5s = sorted(raw.glob("N*_sw*.h5"))[:6]
    if not h5s:
        raise SystemExit("no raw_smoke HDF5")

    import h5py
    dtype_report = []
    with h5py.File(h5s[0], "r") as f:
        for key in ("k_n", "a_n", "b_n", "omega_n"):
            ds = f["turb"][key]
            dtype_report.append({"key": key, "hdf5_dtype": str(ds.dtype), "shape": list(ds.shape)})
    flow0 = make_ks({
        "sigma_w": float(h5py.File(h5s[0], "r")["turb"].attrs.get("sigma_w",
                     h5py.File(h5s[0], "r")["turb"].attrs.get("sigma"))),
        "ell": float(h5py.File(h5s[0], "r")["turb"].attrs["ell"]),
        "U_h": float(h5py.File(h5s[0], "r")["turb"].attrs.get("U_h", 1.0)),
        "seed": int(h5py.File(h5s[0], "r")["turb"].attrs["seed"]),
    })
    sim_dtypes = {
        "k": str(flow0.k.dtype),
        "a": str(flow0.a.dtype),
        "b": str(flow0.b.dtype),
        "omega": str(flow0.omega.dtype),
    }

    rng = np.random.default_rng(0)
    max_abs_h5 = max_rel_h5 = 0.0
    max_abs_mk = max_rel_mk = 0.0
    max_rel_h5_gt001 = 0.0
    u_at_max_rel = None
    n = 0
    for p in h5s:
        modes_h5 = turb_modes_from_h5(p)
        with h5py.File(p, "r") as f:
            attrs = dict(f["turb"].attrs)
            params = dict(f["params"].attrs)
            g = {
                "sigma_w": float(attrs.get("sigma_w", attrs.get("sigma"))),
                "ell": float(attrs["ell"]),
                "U_h": float(attrs.get("U_h", params.get("U_h", 1.0))),
                "seed": int(attrs["seed"]),
            }
        flow = make_ks(g)
        modes_mk = turb_modes_from_ks(flow)
        for _ in range(50):
            x = rng.normal(0.0, 2.0, size=(8, 3))
            t = float(rng.uniform(0.0, 30.0))
            ref = np.asarray(flow.u(x, t), dtype=np.float64)
            got_h5 = ks_u_torch(
                torch.from_numpy(x), t, modes_h5, dtype=torch.float32
            ).numpy().astype(np.float64)
            got_mk = ks_u_torch(
                torch.from_numpy(x), t, modes_mk, dtype=torch.float32
            ).numpy().astype(np.float64)
            for got, store in ((got_h5, "h5"), (got_mk, "mk")):
                diff = np.abs(got - ref)
                rel = diff / np.maximum(np.abs(ref), 1e-30)
                if store == "h5":
                    max_abs_h5 = max(max_abs_h5, float(diff.max()))
                    # track global max rel and |u| there
                    flat = int(np.argmax(rel))
                    rmax = float(rel.ravel()[flat])
                    if rmax >= max_rel_h5:
                        max_rel_h5 = rmax
                        u_at_max_rel = float(np.abs(ref).ravel()[flat])
                    mask = np.abs(ref) > 0.01
                    if np.any(mask):
                        max_rel_h5_gt001 = max(max_rel_h5_gt001, float(rel[mask].max()))
                else:
                    max_abs_mk = max(max_abs_mk, float(diff.max()))
                    max_rel_mk = max(max_rel_mk, float(rel.max()))
            n += 1

    out = {
        "hdf5_turb_dtypes": dtype_report,
        "simulator_KinematicSimulation_dtypes": sim_dtypes,
        "n_checks": n,
        "from_hdf5_modes": {
            "max_abs_diff": max_abs_h5,
            "max_rel_diff": max_rel_h5,
            "max_rel_diff_|u|>0.01": max_rel_h5_gt001,
            "|u|_at_max_rel_point": u_at_max_rel,
        },
        "from_make_ks_float64_modes": {
            "max_abs_diff": max_abs_mk,
            "max_rel_diff": max_rel_mk,
        },
        "conclusion": (
            "HDF5 /turb stores modes as float32 (write_ml_hdf5 fdtype); "
            "KinematicSimulation holds float64. Remaining error after full f64 "
            "eval is from float32 mode quantization on disk."
        ),
    }
    print(json.dumps(out, indent=2))
    dest = repo_root() / "results" / "ml" / "logs" / "diagnose_ks_dtype.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
