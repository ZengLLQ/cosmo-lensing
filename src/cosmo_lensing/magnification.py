"""Jacobian analysis, critical curves, and caustic mapping."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .models.base import BaseLens


@dataclass(frozen=True)
class JacobianComponents:
    """Components of the inverse magnification tensor (Jacobian matrix A)."""

    kappa: np.ndarray | float
    gamma_1: np.ndarray | float
    gamma_2: np.ndarray | float
    gamma_total: np.ndarray | float
    det_a: np.ndarray | float
    lambda_tangential: np.ndarray | float
    lambda_radial: np.ndarray | float
    magnification: np.ndarray | float


class CriticalCurveAnalyzer:
    """Analyzes magnification topology, critical curves, and source-plane caustics."""

    def __init__(self, lens: BaseLens):
        self.lens = lens

    def compute_jacobian(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> JacobianComponents:
        """Compute all linear perturbation quantities on the coordinate grid."""
        kappa = self.lens.convergence(x, y)
        gamma_1, gamma_2 = self.lens.shear(x, y)
        gamma = np.sqrt(gamma_1**2 + gamma_2**2)

        lambda_t = 1.0 - kappa - gamma
        lambda_r = 1.0 - kappa + gamma
        det_a = lambda_t * lambda_r

        with np.errstate(divide="ignore", invalid="ignore"):
            mu = 1.0 / det_a

        return JacobianComponents(
            kappa=kappa,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            gamma_total=gamma,
            det_a=det_a,
            lambda_tangential=lambda_t,
            lambda_radial=lambda_r,
            magnification=mu,
        )

    def extract_critical_curves(
        self,
        grid_extent: float = 4.0,
        grid_res: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute critical curve contours in lens plane and map them to caustics in source plane.

        Returns:
            (theta_x_crit, theta_y_crit, beta_x_caustic, beta_y_caustic)
        """
        x = np.linspace(-grid_extent, grid_extent, grid_res)
        y = np.linspace(-grid_extent, grid_extent, grid_res)
        xx, yy = np.meshgrid(x, y)

        jac = self.compute_jacobian(xx, yy)
        det_a = np.asarray(jac.det_a)

        # Detect zero-crossings of det A
        # Mask near zero
        mask = np.abs(det_a) < 0.15
        crit_x = xx[mask]
        crit_y = yy[mask]

        if len(crit_x) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        ax, ay = self.lens.deflection_angle(crit_x, crit_y)
        caustic_x = crit_x - ax
        caustic_y = crit_y - ay

        return crit_x, crit_y, caustic_x, caustic_y
