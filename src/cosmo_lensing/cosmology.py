"""Flat Lambda-CDM cosmological distance calculator."""

from dataclasses import dataclass
import numpy as np
from scipy import integrate
from .constants import C, G, MPC


@dataclass(frozen=True)
class FlatLambdaCDM:
    """Cosmological parameters for flat Lambda-CDM model.

    Attributes:
        H0: Hubble constant in km/s/Mpc (default 70.0)
        omega_m: Matter density parameter (default 0.3)
    """

    H0: float = 70.0
    omega_m: float = 0.3

    @property
    def omega_lambda(self) -> float:
        """Dark energy density parameter in flat universe."""
        return 1.0 - self.omega_m

    @property
    def hubble_distance_m(self) -> float:
        """Hubble distance c / H0 in meters."""
        h0_si = (self.H0 * 1e3) / MPC  # s^-1
        return C / h0_si

    def ez(self, z: float | np.ndarray) -> float | np.ndarray:
        """Dimensionless Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(self.omega_m * (1.0 + z) ** 3 + self.omega_lambda)

    def comoving_distance(self, z: float) -> float:
        """Comoving line-of-sight distance in meters."""
        if z <= 0.0:
            return 0.0
        val, _ = integrate.quad(lambda zp: 1.0 / self.ez(zp), 0.0, z)
        return self.hubble_distance_m * val

    def angular_diameter_distance(self, z: float) -> float:
        """Angular diameter distance D_A(z) in meters."""
        if z <= 0.0:
            return 0.0
        dc = self.comoving_distance(z)
        return dc / (1.0 + z)

    def angular_diameter_distance_z1_z2(self, z1: float, z2: float) -> float:
        """Angular diameter distance D_A(z1, z2) between lens and source in meters."""
        if z2 <= z1:
            return 0.0
        dc1 = self.comoving_distance(z1)
        dc2 = self.comoving_distance(z2)
        return (dc2 - dc1) / (1.0 + z2)

    def critical_surface_density(self, z_lens: float, z_source: float) -> float:
        """Critical surface mass density Sigma_cr in kg / m^2.

        Sigma_cr = (c^2 / (4 * pi * G)) * (D_s / (D_l * D_ls))
        """
        d_l = self.angular_diameter_distance(z_lens)
        d_s = self.angular_diameter_distance(z_source)
        d_ls = self.angular_diameter_distance_z1_z2(z_lens, z_source)

        if d_l <= 0.0 or d_ls <= 0.0:
            raise ValueError(f"Invalid lens/source distances: D_l={d_l}, D_ls={d_ls}")

        factor = (C**2) / (4.0 * np.pi * G)
        return factor * (d_s / (d_l * d_ls))
