"""Abstract base class for gravitational lens mass models."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseLens(ABC):
    """Abstract interface defining the gravitational lens deflection and potential."""

    @abstractmethod
    def deflection_angle(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        """Compute deflection angle components (alpha_x, alpha_y) in arcseconds."""
        pass

    @abstractmethod
    def potential(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        """Compute 2D effective gravitational lensing potential psi(x, y) in arcsec^2."""
        pass

    @abstractmethod
    def convergence(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        """Compute dimensionless surface mass density / convergence kappa(x, y)."""
        pass

    def shear(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        """Compute shear components (gamma_1, gamma_2) via numerical finite difference.

        gamma_1 = 0.5 * (psi_xx - psi_yy)
        gamma_2 = psi_xy
        """
        eps = 1e-5
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        psi_c = self.potential(x_arr, y_arr)
        psi_xp = self.potential(x_arr + eps, y_arr)
        psi_xm = self.potential(x_arr - eps, y_arr)
        psi_yp = self.potential(x_arr, y_arr + eps)
        psi_ym = self.potential(x_arr, y_arr - eps)

        psi_xx = (psi_xp - 2.0 * psi_c + psi_xm) / (eps**2)
        psi_yy = (psi_yp - 2.0 * psi_c + psi_ym) / (eps**2)

        psi_xpyp = self.potential(x_arr + eps, y_arr + eps)
        psi_xpym = self.potential(x_arr + eps, y_arr - eps)
        psi_xmyp = self.potential(x_arr - eps, y_arr + eps)
        psi_xmym = self.potential(x_arr - eps, y_arr - eps)

        psi_xy = (psi_xpyp - psi_xpym - psi_xmyp + psi_xmym) / (4.0 * eps**2)

        gamma_1 = 0.5 * (psi_xx - psi_yy)
        gamma_2 = psi_xy

        if np.isscalar(x) and np.isscalar(y):
            return float(gamma_1), float(gamma_2)
        return gamma_1, gamma_2

    def magnification(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        """Compute total point magnification mu = 1 / ((1 - kappa)^2 - gamma_1^2 - gamma_2^2)."""
        kappa = self.convergence(x, y)
        gamma_1, gamma_2 = self.shear(x, y)
        gamma_sq = gamma_1**2 + gamma_2**2
        det_a = (1.0 - kappa) ** 2 - gamma_sq
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / det_a
