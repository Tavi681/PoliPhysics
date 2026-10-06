"""Configuration dataclasses and YAML loader.

The simulation is driven by a set of small dataclasses that group the physical
and numerical parameters. A whole run can be described by a :class:`SimConfig`,
which can be loaded from YAML with :func:`load_config`.

Parameters marked ``# TODO: fix before production runs`` in the shipped YAML
files (``k_c``, ``r_d``, ``R_max``, ``k_max``, ...) are working values only:
they are *not* fixed by the paper, which leaves them as "?".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np

from .materials import Material, get_material, register_material

__all__ = [
    "MaterialConfig",
    "NetConfig",
    "DroneConfig",
    "NumericsConfig",
    "ContactConfig",
    "OutputConfig",
    "KinematicConfig",
    "SimConfig",
    "load_config",
]


@dataclass
class MaterialConfig:
    """Material selection: a registry name, or explicit parameters."""

    name: str = "S"
    E0: Optional[float] = None
    b: Optional[float] = None
    eps_b: Optional[float] = None
    rho: Optional[float] = None

    def resolve(self) -> Material:
        """Return the :class:`Material`, registering a custom one if needed."""
        if self.E0 is None:
            return get_material(self.name)
        mat = Material(name=self.name, E0=float(self.E0), b=float(self.b),
                       eps_b=float(self.eps_b), rho=float(self.rho))
        register_material(mat, overwrite=True)
        return mat


@dataclass
class NetConfig:
    """Net topology description."""

    kind: str = "star"  # star | star_with_rings
    N: int = 8
    R: float = 1.0
    eps_p: float = 0.0
    A_hat: float = 1e-6
    radii: Optional[list] = None  # for star_with_rings
    q_ratio: float = 1.0  # q_ring / q_radial for star_with_rings (FDM prestress)
    # Optional prestress of coarse thread 0 (star only); others keep eps_p.
    eps_p_thread0: Optional[float] = None
    # If True (default for paper nets), keep ring nodes at the prescribed
    # radii and set different radial force densities on the inner/outer
    # segments from ring-node equilibrium. If False, free FDM may pull the
    # rings inward.
    fix_radii: bool = True

    def validate(self) -> None:
        if self.kind not in ("star", "star_with_rings"):
            raise ValueError(f"unknown net kind {self.kind!r}")
        if self.kind == "star_with_rings" and not self.radii:
            raise ValueError("star_with_rings requires 'radii'")


@dataclass
class DroneConfig:
    """Rigid-sphere drone."""

    M: float = 1.0
    r_d: float = 0.15  # TODO: fix before production runs
    v0: float = 15.0
    p: tuple = (0.5, 0.0)  # in-plane impact point (px, py)
    gravity: bool = False
    # If True, zero in-plane drone force and velocity every step (z free).
    lock_xy: bool = False

    def validate(self) -> None:
        if self.r_d <= 0:
            raise ValueError("r_d must be positive")


@dataclass
class ContactConfig:
    """Contact model."""

    mode: str = "gripped"  # frictionless | gripped
    k_c: float = 1e7  # TODO: fix before production runs (absolute k_c_mode)
    k_c_mode: str = "absolute"  # absolute | relative
    # When relative: k_c is chosen so the contact frequency at delta_ref equals
    # k_c_factor times the axial frequency of a segment, sqrt((E A/l_s)/m_node).
    k_c_factor: float = 2.0  # TODO: fix before production runs
    segment_contact: bool = False  # optional sphere-segment contact
    delta_ref_frac: float = 0.01  # delta_ref = delta_ref_frac * r_d
    # Runtime penetration guard: if delta > delta_ref at any step, either abort
    # (default) or shrink dt from the actual delta (reduce_dt).
    penetration_guard: str = "abort"  # abort | reduce_dt | off
    penetration_margin: float = 0.5  # dt safety factor for reduce_dt

    def validate(self) -> None:
        if self.mode not in ("frictionless", "gripped"):
            raise ValueError(f"unknown contact mode {self.mode!r}")
        if self.k_c_mode not in ("absolute", "relative"):
            raise ValueError(f"unknown k_c_mode {self.k_c_mode!r}")
        if self.penetration_guard not in ("abort", "reduce_dt", "off"):
            raise ValueError(
                f"unknown penetration_guard {self.penetration_guard!r}")
        if self.k_c <= 0:
            raise ValueError("k_c must be positive")
        if self.k_c_factor <= 0:
            raise ValueError("k_c_factor must be positive")


@dataclass
class NumericsConfig:
    """Time-integration and discretization parameters."""

    n_s: int = 40
    C: float = 0.5  # CFL factor (<= 0.5)
    t_end: float = 0.2
    dt_out: float = 1e-4
    damping: float = 0.0  # viscous damping coefficient (per unit mass), default 0
    area_scale: float = 1.0
    seed: Optional[int] = None
    use_numba: bool = True
    # Compact diagnostic output (round 8): skip full mesh; record per-segment
    # elastic energy on radial 0 plus the rest lumped.
    store_mesh: bool = True
    energy_groups: bool = False

    def validate(self) -> None:
        if self.n_s < 1:
            raise ValueError("n_s must be >= 1")
        if self.C <= 0:
            raise ValueError("C must be positive")


@dataclass
class OutputConfig:
    """Output and cascade-criterion parameters."""

    hdf5: Optional[str] = None
    hdf5_light: bool = False
    hdf5_max_frames: int = 50
    R_max: float = 0.5  # TODO: fix before production runs
    k_max: int = 10  # TODO: fix before production runs

    def validate(self) -> None:
        if self.R_max <= 0 or self.k_max <= 0:
            raise ValueError("R_max and k_max must be positive")


@dataclass
class KinematicConfig:
    """Kinematic forcing (used instead of the drone, for validation).

    A single node receives a prescribed velocity ``v(t)`` or displacement
    ``w(t)`` along a fixed direction. For simple cases the velocity is constant;
    arbitrary time laws can be supplied programmatically via ``func``.

    If ``free_lateral`` is True, only the component along ``direction`` is
    prescribed; the two orthogonal (in-plane / lateral) components remain free
    and respond to the net forces. This matches ``ref/offc_free.py``.
    """

    enabled: bool = False
    node: int = 0
    mode: str = "velocity"  # velocity | displacement
    direction: tuple = (0.0, 0.0, 1.0)
    amplitude: float = 0.0  # constant velocity or displacement magnitude
    func: Optional[Callable[[float], float]] = None  # overrides amplitude
    free_lateral: bool = False
    # After this many time-separated failure events, switch to free_lateral
    # (criterion-B free-in-plane). None = never switch.
    free_lateral_after_failures: Optional[int] = None
    # Constant extra force [N] applied to ``extra_force_node`` (hub = 0).
    extra_force_node: Optional[int] = None
    extra_force: tuple = (0.0, 0.0, 0.0)

    def value(self, t: float) -> float:
        if self.func is not None:
            return float(self.func(t))
        return float(self.amplitude)

    def unit_direction(self) -> np.ndarray:
        d = np.asarray(self.direction, dtype=float)
        n = np.linalg.norm(d)
        if n == 0:
            raise ValueError("kinematic direction must be non-zero")
        return d / n

    def validate(self) -> None:
        if self.mode not in ("velocity", "displacement"):
            raise ValueError(f"unknown kinematic mode {self.mode!r}")


@dataclass
class SimConfig:
    """Full run configuration."""

    material: MaterialConfig = field(default_factory=MaterialConfig)
    net: NetConfig = field(default_factory=NetConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    kinematic: KinematicConfig = field(default_factory=KinematicConfig)

    def validate(self) -> None:
        self.net.validate()
        self.drone.validate()
        self.numerics.validate()
        self.contact.validate()
        self.output.validate()
        self.kinematic.validate()

    def to_flat_dict(self) -> dict:
        """Flatten for HDF5 /params attributes (JSON-serializable scalars)."""
        out: dict[str, Any] = {}
        for group_name, group in asdict(self).items():
            if not isinstance(group, dict):
                continue
            for k, v in group.items():
                if callable(v):
                    continue
                if isinstance(v, (list, tuple)):
                    v = np.asarray(v)
                out[f"{group_name}.{k}"] = v
        return out


def _build_group(cls, data: dict):
    """Instantiate a config dataclass from a dict, ignoring unknown keys."""
    if not data:
        return cls()
    fields = {f for f in cls.__dataclass_fields__}
    kwargs = {k: v for k, v in data.items() if k in fields}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**kwargs)


def load_config(path: str) -> SimConfig:
    """Load a :class:`SimConfig` from a YAML file."""
    import yaml

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = SimConfig(
        material=_build_group(MaterialConfig, raw.get("material", {})),
        net=_build_group(NetConfig, raw.get("net", {})),
        drone=_build_group(DroneConfig, raw.get("drone", {})),
        numerics=_build_group(NumericsConfig, raw.get("numerics", {})),
        contact=_build_group(ContactConfig, raw.get("contact", {})),
        output=_build_group(OutputConfig, raw.get("output", {})),
        kinematic=_build_group(KinematicConfig, raw.get("kinematic", {})),
    )
    # Normalize tuple-typed fields that YAML loads as lists.
    if isinstance(cfg.drone.p, list):
        cfg.drone.p = tuple(cfg.drone.p)
    if isinstance(cfg.kinematic.direction, list):
        cfg.kinematic.direction = tuple(cfg.kinematic.direction)
    if isinstance(cfg.kinematic.extra_force, list):
        cfg.kinematic.extra_force = tuple(cfg.kinematic.extra_force)
    cfg.validate()
    return cfg
