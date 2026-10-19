"""Batched KinematicSimulation field evaluation in torch (cuda/mps/cpu).

Exact formula as ``ballooning.fields.KinematicSimulation._eval``:

    u(x,t) = U_h x_hat + sum_n [ a_n cos(k_n·x' + ω_n t) + b_n sin(...) ]
    x' = x - U_h t x_hat

Modes come from HDF5 ``/turb`` datasets when available, otherwise from the
same deterministic ``KinematicSimulation(..., seed=...)`` constructor used by
preprocess / ``make_ks`` (identical modes; no fresh random draws).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ml.data.air import make_ks


@dataclass
class TurbModes:
    """One run's KS modes (float64 numpy for fidelity; cast on device at eval)."""

    k: np.ndarray       # (N_k, 3)
    a: np.ndarray       # (N_k, 3)
    b: np.ndarray       # (N_k, 3)
    omega: np.ndarray   # (N_k,)
    U_h: float
    seed: int
    sigma_w: float
    ell: float

    @property
    def N_k(self) -> int:
        return int(self.k.shape[0])


def turb_modes_from_ks(flow) -> TurbModes:
    """Extract modes from a constructed KinematicSimulation (matches /turb)."""
    return TurbModes(
        k=np.asarray(flow.k, dtype=np.float64),
        a=np.asarray(flow.a, dtype=np.float64),
        b=np.asarray(flow.b, dtype=np.float64),
        omega=np.asarray(flow.omega, dtype=np.float64),
        U_h=float(flow.U_h),
        seed=int(flow.seed),
        sigma_w=float(flow.sigma),
        ell=float(flow.ell),
    )


def turb_modes_from_globals(globals_: dict[str, Any]) -> TurbModes:
    """Deterministic modes via make_ks (same RNG stream as the simulator)."""
    return turb_modes_from_ks(make_ks(globals_))


def turb_modes_from_h5(path: str | Path) -> TurbModes:
    """Load stored modes from HDF5 ``/turb`` datasets (no RNG)."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as f:
        turb = f["turb"]
        attrs = dict(turb.attrs)
        return TurbModes(
            k=np.asarray(turb["k_n"][:], dtype=np.float64),
            a=np.asarray(turb["a_n"][:], dtype=np.float64),
            b=np.asarray(turb["b_n"][:], dtype=np.float64),
            omega=np.asarray(turb["omega_n"][:], dtype=np.float64),
            U_h=float(attrs.get("U_h", 0.0)),
            seed=int(attrs.get("seed", -1)),
            sigma_w=float(attrs.get("sigma_w", attrs.get("sigma", 0.0))),
            ell=float(attrs.get("ell", 0.0)),
        )


class TurbModeCache:
    """Cache TurbModes by (seed, sigma_w, ell, U_h)."""

    def __init__(self):
        self._cache: dict[tuple, TurbModes] = {}

    def get(self, globals_: dict[str, Any], *, h5_path: str | Path | None = None) -> TurbModes:
        key = (
            int(globals_["seed"]),
            float(globals_["sigma_w"]),
            float(globals_["ell"]),
            float(globals_["U_h"]),
        )
        if key in self._cache:
            return self._cache[key]
        if h5_path is not None and Path(h5_path).exists():
            modes = turb_modes_from_h5(h5_path)
        else:
            modes = turb_modes_from_globals(globals_)
        self._cache[key] = modes
        return modes


def _as_modes_tensors(
    modes: TurbModes | Sequence[TurbModes],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(modes, TurbModes):
        k = torch.as_tensor(modes.k, device=device, dtype=dtype)
        a = torch.as_tensor(modes.a, device=device, dtype=dtype)
        b = torch.as_tensor(modes.b, device=device, dtype=dtype)
        omega = torch.as_tensor(modes.omega, device=device, dtype=dtype)
        U_h = torch.tensor(modes.U_h, device=device, dtype=dtype)
        return k, a, b, omega, U_h
    ks, asas, bs, omegas, uhs = [], [], [], [], []
    for m in modes:
        ks.append(m.k)
        asas.append(m.a)
        bs.append(m.b)
        omegas.append(m.omega)
        uhs.append(m.U_h)
    k = torch.as_tensor(np.stack(ks), device=device, dtype=dtype)
    a = torch.as_tensor(np.stack(asas), device=device, dtype=dtype)
    b = torch.as_tensor(np.stack(bs), device=device, dtype=dtype)
    omega = torch.as_tensor(np.stack(omegas), device=device, dtype=dtype)
    U_h = torch.as_tensor(np.asarray(uhs), device=device, dtype=dtype)
    return k, a, b, omega, U_h


def ks_u_torch(
    x: torch.Tensor,
    t: torch.Tensor | float,
    modes: TurbModes | Sequence[TurbModes],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Evaluate KS air velocity.

    Phase ``k·x' + ω t`` is always computed in float64 (absolute positions from
    spider_abs float64 + x_rel); ``cos``/``sin`` results are cast to ``dtype``
    (default float32) before the amplitude sum.

    Shapes
    ------
    Single run:
      x: (M, 3) or (3,)  →  u same shape
      t: scalar
      modes: TurbModes
    Batched runs (same N_k, padded M):
      x: (B, M, 3)
      t: (B,) or scalar broadcast
      modes: sequence of B TurbModes
    """
    device = x.device
    out_dtype = dtype
    single_point = x.ndim == 1
    single_run = isinstance(modes, TurbModes) and x.ndim <= 2

    if single_point:
        x = x.unsqueeze(0)

    # Modes may be float32 on disk (/turb); promote everything to float64 for
    # the full field evaluation, then cast the result to ``out_dtype``.
    # MPS has no float64: evaluate the field on CPU and move the result back.
    cdev = torch.device("cpu") if device.type == "mps" else device
    k64, a64, b64, omega64, U_h64 = _as_modes_tensors(
        modes, cdev, dtype=torch.float64
    )
    x64 = x.to(device=cdev, dtype=torch.float64)

    if single_run or (isinstance(modes, TurbModes)):
        # x: (M, 3); k: (N_k, 3)
        if not torch.is_tensor(t):
            t64 = torch.tensor(float(t), device=cdev, dtype=torch.float64)
        else:
            t64 = t.to(device=cdev, dtype=torch.float64)
        xp = x64.clone()
        xp[..., 0] = xp[..., 0] - U_h64 * t64
        phase = xp @ k64.T + omega64 * t64  # (M, N_k) float64
        c = torch.cos(phase)
        s = torch.sin(phase)
        up = c @ a64 + s @ b64  # (M, 3) float64
        u = up.clone()
        u[..., 0] = u[..., 0] + U_h64
        u = u.to(device=device, dtype=out_dtype)
        return u[0] if single_point else u

    # Batched runs: x (B, M, 3), k (B, N_k, 3)
    if not torch.is_tensor(t):
        t64 = torch.full((x64.shape[0],), float(t), device=cdev, dtype=torch.float64)
    else:
        t64 = t.to(device=cdev, dtype=torch.float64)
        if t64.ndim == 0:
            t64 = t64.expand(x64.shape[0])
    t64 = t64.view(-1, 1)  # (B, 1)
    xp = x64.clone()
    xp[..., 0] = xp[..., 0] - U_h64.view(-1, 1) * t64
    phase = torch.einsum("bmi,bni->bmn", xp, k64) + omega64.unsqueeze(1) * t64.unsqueeze(-1)
    c = torch.cos(phase)
    s = torch.sin(phase)
    up = torch.einsum("bmn,bni->bmi", c, a64) + torch.einsum("bmn,bni->bmi", s, b64)
    u = up.clone()
    u[..., 0] = u[..., 0] + U_h64.view(-1, 1)
    return u.to(device=device, dtype=out_dtype)


def ks_u_numpy(
    x_abs: np.ndarray,
    t: float,
    modes: TurbModes,
) -> np.ndarray:
    """CPU convenience wrapper returning float32 (M,3)."""
    x = torch.from_numpy(np.asarray(x_abs, dtype=np.float64))
    u = ks_u_torch(x, t, modes, dtype=torch.float32)
    return u.numpy().astype(np.float32)
