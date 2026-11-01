"""Fermat potential and gravitational lensing time delays."""

from typing import Tuple
import numpy as np
from .constants import C, ARCSEC_TO_RAD, DAY_SECONDS
from .cosmology import FlatLambdaCDM
from .models.base import BaseLens


class TimeDelayCalculator:
    """Calculates geometric, gravitational, and relative image time delays."""

    def __init__(self, lens: BaseLens, cosmology: FlatLambdaCDM, z_lens: float, z_source: float):
        self.lens = lens
        self.cosmology = cosmology
        self.z_lens = float(z_lens)
        self.z_source = float(z_source)

        d_l = cosmology.angular_diameter_distance(z_lens)
        d_s = cosmology.angular_diameter_distance(z_source)
        d_ls = cosmology.angular_diameter_distance_z1_z2(z_lens, z_source)

        # Time-delay distance D_delta_t = (1 + z_l) * (D_l * D_s / D_ls) in meters
        self.d_dt = (1.0 + self.z_lens) * (d_l * d_s / d_ls)

    def fermat_potential(
        self,
        theta_x: float,
        theta_y: float,
        beta_x: float,
        beta_y: float,
    ) -> float:
        """Dimensionless Fermat potential tau = 0.5 * |theta - beta|^2 - psi(theta) in arcsec^2."""
        geom_term = 0.5 * ((theta_x - beta_x) ** 2 + (theta_y - beta_y) ** 2)
        grav_term = self.lens.potential(theta_x, theta_y)
        return float(geom_term - grav_term)

    def time_delay_days(
        self,
        theta_x: float,
        theta_y: float,
        beta_x: float,
        beta_y: float,
    ) -> float:
        """Physical time delay in days relative to an undeflected ray."""
        tau_arcsec2 = self.fermat_potential(theta_x, theta_y, beta_x, beta_y)
        tau_rad2 = tau_arcsec2 * (ARCSEC_TO_RAD**2)
        dt_seconds = (self.d_dt / C) * tau_rad2
        return float(dt_seconds / DAY_SECONDS)

    def relative_time_delay_days(
        self,
        img1: Tuple[float, float],
        img2: Tuple[float, float],
        beta: Tuple[float, float],
    ) -> float:
        """Relative time delay Delta t_12 = t(img1) - t(img2) in days."""
        t1 = self.time_delay_days(img1[0], img1[1], beta[0], beta[1])
        t2 = self.time_delay_days(img2[0], img2[1], beta[0], beta[1])
        return float(t1 - t2)
