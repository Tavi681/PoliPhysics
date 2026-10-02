"""netsim: explicit mass-spring simulator for drone impact on a prestressed net.

Implements Algorithm 1 of the paper ``net_students_en.tex`` (Section "Numerical
method"): an explicit velocity-Verlet mass-spring integrator for the impact of a
rigid drone on a prestressed planar net, with permanent thread failure.

All quantities are in SI units.
"""

from .materials import Material, MATERIALS, get_material, register_material
from .topology import Net, star, star_with_rings, from_arrays
from .config import (
    MaterialConfig,
    NetConfig,
    DroneConfig,
    NumericsConfig,
    ContactConfig,
    OutputConfig,
    KinematicConfig,
    SimConfig,
    load_config,
)
from .simulate import simulate, Result

__all__ = [
    "Material",
    "MATERIALS",
    "get_material",
    "register_material",
    "Net",
    "star",
    "star_with_rings",
    "from_arrays",
    "MaterialConfig",
    "NetConfig",
    "DroneConfig",
    "NumericsConfig",
    "ContactConfig",
    "OutputConfig",
    "KinematicConfig",
    "SimConfig",
    "load_config",
    "simulate",
    "Result",
]

__version__ = "0.1.0"
