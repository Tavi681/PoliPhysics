"""Physical and numerical parameters (params.py).

All defaults are taken from ``tema1_objective.md`` and Habchi & Jawed (2022).
Any numerical choice not stated in the objective is a named parameter here and is
tagged with a ``# TODO`` comment (see the final summary / README for the list).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Params:
    """Container for all physical and numerical parameters of a run."""

    # --- Discretisation --------------------------------------------------
    N: int = 1  # number of threads (N_t threads in the paper, here `N`)
    N_t: int = 100  # number of edges (and interior segments) per thread

    # --- Geometry --------------------------------------------------------
    L: float = 0.5  # thread length [m]
    r: float = 300e-9  # thread radius [m]
    d0: float = 100e-6  # initial spacing of thread bases on a circle [m]
    z0: float = 0.0  # initial spider altitude [m]

    # --- Spider body -----------------------------------------------------
    m: float = 1e-6  # spider mass [kg]  (1 mg)
    r_s: float = 1e-3  # spider radius [m]
    Q_s: float = 3e-12  # spider body charge [C]  (3 pC)

    # --- Material --------------------------------------------------------
    rho_t: float = 1200.0  # thread density [kg/m^3]
    Y: float = 25e9  # Young's modulus [Pa]
    nu: float = 0.5  # Poisson ratio (config)

    # --- Fluid / environment --------------------------------------------
    mu: float = 1.837e-5  # dynamic viscosity of air [Pa s]
    g: float = 9.81  # gravitational acceleration [m/s^2]
    k_e: float = 8.9875517923e9  # Coulomb constant [N m^2 / C^2]

    # --- Electrostatics --------------------------------------------------
    Q_t: float = 1.28e-9  # total charge per thread [C] (chamber validation ~1.28 nC/mg)
    charge_model: str = "tip"  # "tip" or "uniform"

    # --- Field / flow model selection -----------------------------------
    field_model: str = "chamber"  # "gorham" | "chamber" | "constant"
    flow_model: str = "zero"  # "zero" | "uniform"
    E_constant: float = 1.2e5  # E for the "constant" field model [V/m]  # TODO: config
    flow_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)  # UniformFlow vector [m/s]

    # --- Time stepping (Algorithm 1) ------------------------------------
    dt0: float = 1e-4  # initial time step [s]
    dt_max: float = 1e-2  # maximum time step [s]
    dt_min: float = 1e-9  # abort threshold [s]
    eps: float = 1e-10  # Newton tolerance on ||F||_1 [N]
    K: int = 20  # max Newton iterations
    t_end: float = 1.0  # final time [s]
    output_dt: float = 1e-3  # output interval [s]

    # --- Steady-state detection (Alg.1 line 18) -------------------------
    delta: float = 1e-6  # relative velocity change threshold
    t_w: float = 0.05  # steady-state window [s]

    # --- Algorithm 2 stopping rules (off by default) --------------------
    use_alg2_stopping: bool = False
    h: float = 10.0  # "rise" altitude threshold [m]

    # --- Numerical options ----------------------------------------------
    lag_tangent: bool = True  # lag RFT node tangent in the Jacobian (documented default)
    fd_eps: float = 1e-6  # finite-difference step for tests  # TODO: config
    coulomb_softening: float = 0.0  # soft-floor added to |r| in Coulomb (0 = off)  # TODO: config
    # False: Habchi & Jawed Eq. (eq:coulomb), sum over i != 0 and i != k
    # (the spider is never a source). True: every pair i != k, including the spider.
    coulomb_include_spider: bool = False
    # Entangled iff d_min < entangle_contact_factor * r (default 2r: threads in contact).
    entangle_contact_factor: float = 2.0
    # exponent beta in the first integral T^beta sin(theta). If None, the physical
    # value eta_perp/eta_par is used (RFT steady-thread invariant).  # TODO: config
    first_integral_beta: float | None = None

    # --- Reserved turbulence metadata (written to /turb, all optional) ---
    turb: dict = field(default_factory=dict)

    # =====================================================================
    # Derived quantities
    # =====================================================================
    @property
    def A(self) -> float:
        """Cross-sectional area A = pi r^2."""
        return math.pi * self.r ** 2

    @property
    def I(self) -> float:  # noqa: E743 - matches paper symbol
        """Second moment of area I = pi r^4 / 4."""
        return math.pi * self.r ** 4 / 4.0

    @property
    def J(self) -> float:
        """Polar moment J = pi r^4 / 2."""
        return math.pi * self.r ** 4 / 2.0

    @property
    def G(self) -> float:
        """Shear modulus G = Y / (2 (1 + nu))."""
        return self.Y / (2.0 * (1.0 + self.nu))

    @property
    def l0(self) -> float:
        """Reference (undeformed) edge length l0 = L / N_t."""
        return self.L / self.N_t

    @property
    def eta_par(self) -> float:
        """RFT parallel drag coefficient eta_par = 2 pi mu / (ln(L/r) - 1/2)."""
        return 2.0 * math.pi * self.mu / (math.log(self.L / self.r) - 0.5)

    @property
    def eta_perp(self) -> float:
        """RFT perpendicular drag coefficient eta_perp = 4 pi mu / (ln(L/r) + 1/2)."""
        return 4.0 * math.pi * self.mu / (math.log(self.L / self.r) + 0.5)

    @property
    def zeta_s(self) -> float:
        """Stokes drag coefficient for the spider sphere: 6 pi mu r_s."""
        return 6.0 * math.pi * self.mu * self.r_s

    @property
    def beta_first_integral(self) -> float:
        """Exponent for T^beta sin(theta); defaults to eta_perp/eta_par."""
        if self.first_integral_beta is not None:
            return self.first_integral_beta
        return self.eta_perp / self.eta_par

    @property
    def n_nodes(self) -> int:
        """Total number of nodes n = N * N_t + 1."""
        return self.N * self.N_t + 1

    @property
    def n_edges(self) -> int:
        """Total number of edges N * N_t."""
        return self.N * self.N_t

    @property
    def n_dof(self) -> int:
        """Total DOF count: 3(N N_t + 1) + N N_t."""
        return 3 * self.n_nodes + self.n_edges

    def validate(self) -> None:
        """Basic sanity checks on parameter combinations."""
        if self.charge_model not in ("tip", "uniform"):
            raise ValueError(f"charge_model must be 'tip' or 'uniform', got {self.charge_model!r}")
        if self.field_model not in ("gorham", "chamber", "constant"):
            raise ValueError(f"unknown field_model {self.field_model!r}")
        if self.flow_model not in ("zero", "uniform"):
            raise ValueError(f"unknown flow_model {self.flow_model!r}")
        if self.N < 1 or self.N_t < 1:
            raise ValueError("N and N_t must be >= 1")
