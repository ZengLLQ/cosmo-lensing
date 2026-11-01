"""Singular Isothermal Sphere (SIS) gravitational lens model."""

from typing import Tuple
import numpy as np
from .base import BaseLens
from ..constants import C, ARCSEC_TO_RAD, RAD_TO_ARCSEC


class SISLens(BaseLens):
    """Singular Isothermal Sphere lens with velocity dispersion sigma_v or direct theta_e.

    Deflection angle is constant in magnitude: |alpha| = theta_e
    Convergence: kappa(theta) = theta_e / (2 * theta)
    """

    def __init__(
        self,
        theta_e: float | None = None,
        sigma_v_kms: float | None = None,
        d_ls_over_d_s: float | None = None,
        xc: float = 0.0,
        yc: float = 0.0,
    ):
        self.xc = float(xc)
        self.yc = float(yc)

        if theta_e is not None:
            if theta_e <= 0:
                raise ValueError(f"theta_e must be positive, got {theta_e}")
            self.theta_e = float(theta_e)
        elif sigma_v_kms is not None and d_ls_over_d_s is not None:
            # theta_e = 4 * pi * (sigma_v / c)^2 * (D_ls / D_s) in radians, then converted to arcsec
            sigma_si = sigma_v_kms * 1e3
            theta_e_rad = 4.0 * np.pi * (sigma_si / C) ** 2 * d_ls_over_d_s
            self.theta_e = float(theta_e_rad * RAD_TO_ARCSEC)
        else:
            raise ValueError("Must specify either theta_e or (sigma_v_kms and d_ls_over_d_s)")

    def deflection_angle(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.where(r == 0, 1e-12, r)

        alpha_x = self.theta_e * (dx / r_safe)
        alpha_y = self.theta_e * (dy / r_safe)

        if np.isscalar(x) and np.isscalar(y):
            return float(alpha_x), float(alpha_y)
        return alpha_x, alpha_y

    def potential(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r = np.sqrt(dx**2 + dy**2)
        psi = self.theta_e * r
        if np.isscalar(x) and np.isscalar(y):
            return float(psi)
        return psi

    def convergence(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.where(r == 0, 1e-12, r)
        kappa = self.theta_e / (2.0 * r_safe)
        if np.isscalar(x) and np.isscalar(y):
            return float(kappa)
        return kappa

    def analytic_images(self, beta_x: float, beta_y: float) -> list[Tuple[float, float, float]]:
        """Compute exact analytic images for SIS lens.

        If beta < theta_e: 2 images formed at theta_pm = beta pm theta_e
        If beta >= theta_e: 1 image formed at theta_+ = beta + theta_e
        """
        bx = beta_x - self.xc
        by = beta_y - self.yc
        beta = np.sqrt(bx**2 + by**2)

        if beta == 0:
            return [(self.xc, self.yc, np.inf)]

        ux = bx / beta
        uy = by / beta

        images = []
        # Image 1 (outer)
        t_plus = beta + self.theta_e
        mu_plus = 1.0 + (self.theta_e / beta)
        images.append((self.xc + t_plus * ux, self.yc + t_plus * uy, float(mu_plus)))

        # Image 2 (inner, if beta < theta_e)
        if beta < self.theta_e:
            t_minus = beta - self.theta_e
            mu_minus = 1.0 - (self.theta_e / beta)
            images.append((self.xc + t_minus * ux, self.yc + t_minus * uy, float(mu_minus)))

        return images
