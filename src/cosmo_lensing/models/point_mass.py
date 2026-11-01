"""Point mass / Schwarzschild gravitational lens model."""

from typing import Tuple
import numpy as np
from .base import BaseLens


class PointMassLens(BaseLens):
    """Point mass gravitational lens characterized by its Einstein radius theta_e."""

    def __init__(self, theta_e: float, xc: float = 0.0, yc: float = 0.0):
        if theta_e <= 0:
            raise ValueError(f"Einstein radius theta_e must be strictly positive, got {theta_e}")
        self.theta_e = float(theta_e)
        self.xc = float(xc)
        self.yc = float(yc)

    def deflection_angle(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r2 = dx**2 + dy**2
        r2_safe = np.where(r2 == 0, 1e-12, r2)

        alpha_x = (self.theta_e**2 * dx) / r2_safe
        alpha_y = (self.theta_e**2 * dy) / r2_safe

        if np.isscalar(x) and np.isscalar(y):
            return float(alpha_x), float(alpha_y)
        return alpha_x, alpha_y

    def potential(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r = np.sqrt(dx**2 + dy**2)
        r_safe = np.where(r == 0, 1e-12, r)
        psi = self.theta_e**2 * np.log(r_safe)
        if np.isscalar(x) and np.isscalar(y):
            return float(psi)
        return psi

    def convergence(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        """Convergence is zero everywhere except at the origin singularity."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        dx = x_arr - self.xc
        dy = y_arr - self.yc
        r = np.sqrt(dx**2 + dy**2)
        kappa = np.where(r < 1e-6, np.inf, 0.0)
        if np.isscalar(x) and np.isscalar(y):
            return float(kappa)
        return kappa

    def analytic_images(self, beta_x: float, beta_y: float) -> list[Tuple[float, float, float]]:
        """Compute exact analytic image positions (theta_x, theta_y) and magnifications for point source.

        theta_pm = 0.5 * (beta pm sqrt(beta^2 + 4 * theta_e^2))
        """
        bx = beta_x - self.xc
        by = beta_y - self.yc
        beta = np.sqrt(bx**2 + by**2)

        if beta == 0:
            # Einstein ring
            return [(self.xc, self.yc, np.inf)]

        term = np.sqrt(beta**2 + 4.0 * self.theta_e**2)
        theta_plus = 0.5 * (beta + term)
        theta_minus = 0.5 * (beta - term)

        # Unit vector along beta
        ux = bx / beta
        uy = by / beta

        img_plus = (self.xc + theta_plus * ux, self.yc + theta_plus * uy)
        img_minus = (self.xc + theta_minus * ux, self.yc + theta_minus * uy)

        # Magnification: mu_pm = (1 - (theta_e / theta)^4)^(-1)
        mu_plus = 1.0 / (1.0 - (self.theta_e / theta_plus) ** 4)
        mu_minus = 1.0 / (1.0 - (self.theta_e / abs(theta_minus)) ** 4)

        return [
            (img_plus[0], img_plus[1], float(mu_plus)),
            (img_minus[0], img_minus[1], float(mu_minus)),
        ]
