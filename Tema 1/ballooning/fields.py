"""Electric field and air-velocity models (fields.py).

Electric field E(z) [V/m] acting along +z, and air velocity u(x, t) [m/s].
All fields expose an analytic derivative dE/dz for the Jacobian of the lift force.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class ElectricField(Protocol):
    """Vertical atmospheric electric field E(z)."""

    def E(self, z: float) -> float:  # noqa: N802 - matches paper symbol
        ...

    def dE(self, z: float) -> float:  # noqa: N802
        ...


class GorhamField:
    """Gorham flat-earth model: E = E0 exp(-alpha z) (Gorham flat-earth field)."""

    def __init__(self, E0: float = 120.0, alpha: float = 3e-4):
        self.E0 = E0
        self.alpha = alpha

    def E(self, z: float) -> float:  # noqa: N802
        return self.E0 * math.exp(-self.alpha * z)

    def dE(self, z: float) -> float:  # noqa: N802
        return -self.alpha * self.E0 * math.exp(-self.alpha * z)


class ChamberField:
    """Morley & Gorham chamber fit: E1 e^{-z/z1} + E2 e^{-z/z2} + Einf (chamber field fit)."""

    def __init__(self, E1: float = 2.52e5, z1: float = 1.51e-3,
                 E2: float = 5.07e4, z2: float = 7.93e-3, Einf: float = 7.41e3):
        self.E1 = E1
        self.z1 = z1
        self.E2 = E2
        self.z2 = z2
        self.Einf = Einf

    def E(self, z: float) -> float:  # noqa: N802
        return (self.E1 * math.exp(-z / self.z1)
                + self.E2 * math.exp(-z / self.z2) + self.Einf)

    def dE(self, z: float) -> float:  # noqa: N802
        return (-self.E1 / self.z1 * math.exp(-z / self.z1)
                - self.E2 / self.z2 * math.exp(-z / self.z2))


class ConstantField:
    """Uniform field E(z) = E0."""

    def __init__(self, E0: float):
        self.E0 = E0

    def E(self, z: float) -> float:  # noqa: N802
        return self.E0

    def dE(self, z: float) -> float:  # noqa: N802
        return 0.0


class AirFlow(Protocol):
    """Ambient air velocity field u(x, t).

    ``u`` accepts a single point ``(3,)`` or a batch ``(M, 3)`` and returns the
    matching shape. ``grad_u`` returns the velocity gradient ``d u_i / d x_j`` with
    shape ``(3, 3)`` (single point) or ``(M, 3, 3)`` (batch).
    """

    def u(self, x: np.ndarray, t: float) -> np.ndarray:
        ...

    def grad_u(self, x: np.ndarray, t: float) -> np.ndarray:
        ...


class ZeroFlow:
    """Still air, u = 0 everywhere."""

    def u(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.zeros(3) if x.ndim == 1 else np.zeros_like(x)

    def grad_u(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.zeros((3, 3)) if x.ndim == 1 else np.zeros((x.shape[0], 3, 3))


class UniformFlow:
    """Spatially uniform, steady flow u = U (grad u = 0)."""

    def __init__(self, U):
        self.U = np.asarray(U, dtype=float)

    def u(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return self.U.copy()
        return np.broadcast_to(self.U, x.shape).copy()

    def grad_u(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.zeros((3, 3)) if x.ndim == 1 else np.zeros((x.shape[0], 3, 3))


# von Karman spectral integral I = int_0^inf s^4 / (1+s^2)^(17/6) ds
# = (1/2) B(5/2, 1/3); normalises E(k) so that int_0^inf E dk = (3/2) sigma^2.
_VK_INTEGRAL = 0.5 * math.gamma(2.5) * math.gamma(1.0 / 3.0) / math.gamma(17.0 / 6.0)
# Log each unique renormalization factor once (factor is seed-independent).
_RENORM_LOGGED: set[tuple] = set()

# Stage B band is independent of the rod mesh. Reference: L=0.5 m, N_t=100
# so dl = 0.005 m and k_max = 2 pi / (5 dl) = 251.327... 1/m. k_min = 0.1/ell.
# Modes then depend only on (seed, sigma, ell, U_h, N_k, lambda).
KS_REF_DL = 0.005
KS_K_MAX = 2.0 * math.pi / (5.0 * KS_REF_DL)


def von_karman_E(k, sigma: float, ell: float) -> np.ndarray:
    """von Karman energy spectrum E(k), Eq. (eq:vk).

    E(k) = C sigma^2 ell (k ell)^4 / (1 + (k ell)^2)^(17/6), with C fixed so that
    int_0^inf E dk = (3/2) sigma^2 (i.e. C = 1.5 / _VK_INTEGRAL).
    """
    k = np.asarray(k, dtype=float)
    C = 1.5 / _VK_INTEGRAL
    x = k * ell
    return C * sigma ** 2 * ell * x ** 4 / (1.0 + x ** 2) ** (17.0 / 6.0)


class KinematicSimulation:
    """Synthetic homogeneous isotropic turbulence, Eq. (eq:ks) with spectrum (eq:vk).

        u(x, t) = U_h x_hat
                  + sum_n [ a_n cos(k_n . x' + omega_n t)
                          + b_n sin(k_n . x' + omega_n t) ],
        x' = x - U_h t x_hat.

    Modes are ``N_k`` geometric wavenumber shells from ``k_min = 0.1/ell`` to
    ``k_max = 2 pi / (5 * 0.005 m) = 251.327 1/m`` (fixed; independent of the
    rod mesh ``N_t``). Directions ``k_hat`` are uniform on the sphere; ``a_n, b_n``
    are perpendicular to ``k_n`` (so div u = 0 exactly) with random in-plane
    orientation and ``|a_n| = |b_n| = sqrt(2 E(k_n) dk_n)`` so that
    ``(1/2) <|u - U_h x_hat|^2> = sum_n E(k_n) dk_n = (3/2) sigma^2`` (variance
    ``sigma^2`` per velocity component). ``omega_n = lambda sqrt(k_n^3 E(k_n))``.

    All randomness comes from one ``numpy.random.Generator(seed)``. The discrete
    modes therefore depend only on ``(seed, sigma, ell, U_h, N_k, lambda)``.
    ``mesh_limited=True`` restores the older ``k_max = 2 pi / (5 L/N_t)`` band
    (used only to reconstruct pre-fix T2 fields).
    """

    def __init__(self, sigma: float, ell: float, U_h: float = 0.0,
                 N_k: int = 100, seed: int = 0, L: float = 0.5, N_t: int = 100,
                 lambda_: float = 0.5, k_min_factor: float = 0.1,
                 k_max_dl_factor: float = 5.0, energy_capture_min: float = 0.98,
                 renormalize: bool = False, mesh_limited: bool = False):
        self.sigma = float(sigma)
        self.ell = float(ell)
        self.U_h = float(U_h)
        self.N_k = int(N_k)
        self.seed = int(seed)
        self.lambda_ = float(lambda_)
        self.renormalize = bool(renormalize)
        self.mesh_limited = bool(mesh_limited)

        k_min = k_min_factor / ell
        if self.mesh_limited:
            dl = L / N_t
            k_max = 2.0 * math.pi / (k_max_dl_factor * dl)
        else:
            k_max = KS_K_MAX

        target = 1.5 * self.sigma ** 2

        def _fraction(kmin):
            km, dkm = self._shells(kmin, k_max, self.N_k)
            if target <= 0:
                return 1.0, km, dkm
            frac = float(np.sum(von_karman_E(km, self.sigma, self.ell) * dkm)
                         / target)
            return frac, km, dkm

        fraction, k_mag, dk = _fraction(k_min)
        # Mesh-limited (legacy) path only: lower k_min if the missing energy is
        # actually at low k. The Stage B default never does this, so the band
        # stays k_min = 0.1/ell, k_max = KS_K_MAX for every N_t.
        if self.mesh_limited:
            k_floor = 1e-4 / ell
            floor_fraction, _, _ = _fraction(k_floor)
            if fraction < energy_capture_min and floor_fraction >= energy_capture_min:
                while fraction < energy_capture_min and k_min > k_floor:
                    k_min *= 0.5
                    fraction, k_mag, dk = _fraction(k_min)
        self.k_min_used = float(k_min)
        self.k_max = float(k_max)
        self.energy_fraction = fraction

        rng = np.random.default_rng(self.seed)
        # Uniform directions on the sphere (normalised Gaussians).
        g = rng.standard_normal((self.N_k, 3))
        k_hat = g / np.linalg.norm(g, axis=1, keepdims=True)
        self.k = k_hat * k_mag[:, None]  # (N_k, 3)

        # Orthonormal basis (e1, e2) in the plane perpendicular to k_hat.
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (self.N_k, 1))
        near_z = np.abs(k_hat[:, 2]) > 0.9
        ref[near_z] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, k_hat)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(k_hat, e1)  # unit, since k_hat _|_ e1

        amp = np.sqrt(2.0 * von_karman_E(k_mag, self.sigma, self.ell) * dk)  # (N_k,)
        psi_a = rng.uniform(0.0, 2.0 * math.pi, self.N_k)
        psi_b = rng.uniform(0.0, 2.0 * math.pi, self.N_k)
        self.a = amp[:, None] * (np.cos(psi_a)[:, None] * e1
                                 + np.sin(psi_a)[:, None] * e2)
        self.b = amp[:, None] * (np.cos(psi_b)[:, None] * e1
                                 + np.sin(psi_b)[:, None] * e2)
        self.omega = self.lambda_ * np.sqrt(
            k_mag ** 3 * von_karman_E(k_mag, self.sigma, self.ell))
        self.k_mag = k_mag
        self.dk = dk

        # Stage B option: rescale a_n, b_n by a common factor so the resolved
        # discrete energy sum_n E(k_n) dk_n equals (3/2) sigma^2 exactly.
        # Amplitudes scale as sqrt(E dk), so the energy scales as factor^2.
        resolved = float(np.sum(von_karman_E(k_mag, self.sigma, self.ell) * dk))
        self.renorm_factor = 1.0
        if self.renormalize and resolved > 0.0 and target > 0.0:
            self.renorm_factor = float(np.sqrt(target / resolved))
            self.a *= self.renorm_factor
            self.b *= self.renorm_factor
            self.energy_fraction = 1.0
            key = (self.N_k, self.sigma, self.ell, self.k_min_used, self.k_max)
            if key not in _RENORM_LOGGED:
                _RENORM_LOGGED.add(key)
                print(f"KinematicSimulation: turb_renormalize factor={self.renorm_factor:.6f} "
                      f"(resolved/target was {resolved / target:.6f}, N_k={self.N_k})",
                      flush=True)

    @staticmethod
    def _shells(k_min: float, k_max: float, N_k: int):
        """Geometric shell centres (geometric mean of edges) and widths."""
        edges = k_min * (k_max / k_min) ** (np.arange(N_k + 1) / N_k)
        k_mag = np.sqrt(edges[:-1] * edges[1:])
        dk = np.diff(edges)
        return k_mag, dk

    def _eval(self, X: np.ndarray, t: float, want_grad: bool):
        X2 = np.atleast_2d(np.asarray(X, dtype=float))  # (M, 3)
        xp = X2.copy()
        xp[:, 0] -= self.U_h * t  # advected frame x' = x - U_h t x_hat
        phase = xp @ self.k.T + self.omega * t  # (M, N_k)
        c = np.cos(phase)
        s = np.sin(phase)
        up = c @ self.a + s @ self.b  # (M, 3)
        u = up.copy()
        u[:, 0] += self.U_h
        grad = None
        if want_grad:
            # d u_i / d x_j = sum_n (-a_ni sin phi + b_ni cos phi) k_nj
            cc = (-s)[:, :, None] * self.a[None, :, :] \
                + c[:, :, None] * self.b[None, :, :]  # (M, N_k, 3)
            grad = np.einsum("mni,nj->mij", cc, self.k)  # (M, 3, 3)
        return u, grad

    def u(self, x: np.ndarray, t: float) -> np.ndarray:  # Eq. (eq:ks)
        x = np.asarray(x, dtype=float)
        u, _ = self._eval(x, t, want_grad=False)
        return u[0] if x.ndim == 1 else u

    def grad_u(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        _, g = self._eval(x, t, want_grad=True)
        return g[0] if x.ndim == 1 else g

    def largest_scale_index(self) -> int:
        """Index of the smallest-|k| shell (largest spatial scale)."""
        return int(np.argmin(self.k_mag))

    def largest_scale_w(self, x: np.ndarray, t: float) -> float:
        """Contribution of the largest-scale mode to w = u_z at ``x, t``."""
        i = self.largest_scale_index()
        x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(3)
        xp = x.copy()
        xp[0] -= self.U_h * t
        phase = float(np.dot(self.k[i], xp) + self.omega[i] * t)
        return float(self.a[i, 2] * math.cos(phase) + self.b[i, 2] * math.sin(phase))

    def modes_equal(self, other: "KinematicSimulation") -> bool:
        """Exact equality of (k_n, a_n, b_n, omega_n)."""
        return (np.array_equal(self.k, other.k)
                and np.array_equal(self.a, other.a)
                and np.array_equal(self.b, other.b)
                and np.array_equal(self.omega, other.omega))

    def turb_metadata(self) -> dict:
        """Everything needed to reconstruct the field, for /turb in the HDF5 output."""
        return {
            "sigma_w": self.sigma,
            "ell": self.ell,
            "U_h": self.U_h,
            "lambda": self.lambda_,
            "seed": self.seed,
            "N_k": self.N_k,
            "k_min_used": self.k_min_used,
            "k_max": self.k_max,
            "energy_fraction": self.energy_fraction,
            "turb_renormalize": self.renormalize,
            "renorm_factor": self.renorm_factor,
            "k_n": self.k.copy(),
            "a_n": self.a.copy(),
            "b_n": self.b.copy(),
            "omega_n": self.omega.copy(),
        }


def make_field(P) -> ElectricField:
    """Construct the electric field selected in ``P.field_model``."""
    if P.field_model == "gorham":
        return GorhamField()
    if P.field_model == "chamber":
        return ChamberField()
    if P.field_model == "constant":
        return ConstantField(P.E_constant)
    raise ValueError(f"unknown field_model {P.field_model!r}")


def make_flow(P) -> AirFlow:
    """Construct the air-flow model selected in ``P.flow_model``.

    For the ``"kinematic"`` model this also records the turbulence metadata into
    ``P.turb`` (consumed by ``io_hdf5`` to write the ``/turb`` group).
    """
    if P.flow_model == "zero":
        return ZeroFlow()
    if P.flow_model == "uniform":
        return UniformFlow(P.flow_velocity)
    if P.flow_model == "kinematic":
        ks = KinematicSimulation(
            sigma=P.sigma_w, ell=P.ell, U_h=P.U_h, N_k=P.turb_N_k,
            seed=P.turb_seed, L=P.L, N_t=P.N_t, lambda_=P.turb_lambda,
            renormalize=P.turb_renormalize,
        )
        P.turb = ks.turb_metadata()
        return ks
    raise ValueError(f"unknown flow_model {P.flow_model!r}")
