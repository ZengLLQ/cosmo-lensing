"""Navarro-Frenk-White (NFW) dark matter halo gravitational lens model."""

from typing import Tuple
import numpy as np
from .base import BaseLens


class NFWLens(BaseLens):
    """Navarro-Frenk-White profile lens with scale radius and characteristic convergence.

    rho(r) = rho_s / ((r/r_s) * (1 + r/r_s)^2)
    """

    def __init__(
        self,
        kappa_s: float,
        r_s_arcsec: float,
        xc: float = 0.0,
        yc: float = 0.0,
    ):
        if kappa_s <= 0 or r_s_arcsec <= 0:
            raise ValueError("kappa_s and r_s_arcsec must be strictly positive")
        self.kappa_s = float(kappa_s)
        self.r_s = float(r_s_arcsec)
        self.xc = float(xc)
        self.yc = float(yc)

    def _f_func(self, x: np.ndarray) -> np.ndarray:
        """Helper function F(x) for NFW lens projections."""
        res = np.empty_like(x, dtype=float)
        tol = 1e-6

        # x < 1 - tol
        m_inner = x < (1.0 - tol)
        if np.any(m_inner):
            xi = x[m_inner]
            xi_safe = np.maximum(xi, 1e-12)
            res[m_inner] = (2.0 / np.sqrt(1.0 - xi_safe**2)) * np.arctanh(
                np.sqrt((1.0 - xi_safe) / (1.0 + xi_safe))
            )

        # x > 1 + tol
        m_outer = x > (1.0 + tol)
        if np.any(m_outer):
            xo = x[m_outer]
            res[m_outer] = (2.0 / np.sqrt(xo**2 - 1.0)) * np.arctan(
                np.sqrt((xo - 1.0) / (xo + 1.0))
            )

        # |x - 1| <= tol
        m_trans = (~m_inner) & (~m_outer)
        if np.any(m_trans):
            res[m_trans] = 1.0

        return res

    def _g_func(self, x: np.ndarray) -> np.ndarray:
        """Helper function g(x) = ln(x/2) + F(x)."""
        x_safe = np.maximum(x, 1e-12)
        f = self._f_func(x)
        return np.log(x_safe / 2.0) + f

    def deflection_angle(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r = np.sqrt(dx**2 + dy**2)
        x_dim = np.maximum(r / self.r_s, 1e-12)

        g = self._g_func(x_dim)
        # alpha_mag = (4 * kappa_s * r_s / x_dim) * g
        alpha_mag = (4.0 * self.kappa_s * self.r_s / x_dim) * g

        r_safe = np.where(r == 0, 1e-12, r)
        alpha_x = alpha_mag * (dx / r_safe)
        alpha_y = alpha_mag * (dy / r_safe)

        if np.isscalar(x) and np.isscalar(y):
            return float(alpha_x), float(alpha_y)
        return alpha_x, alpha_y

    def convergence(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r = np.sqrt(dx**2 + dy**2)
        x_dim = np.asarray(r / self.r_s, dtype=float)

        res = np.empty_like(x_dim)
        tol = 1e-4

        m_inner = x_dim < (1.0 - tol)
        if np.any(m_inner):
            xi = np.maximum(x_dim[m_inner], 1e-12)
            f = self._f_func(xi)
            res[m_inner] = 2.0 * self.kappa_s * (1.0 - f) / (xi**2 - 1.0)

        m_outer = x_dim > (1.0 + tol)
        if np.any(m_outer):
            xo = x_dim[m_outer]
            f = self._f_func(xo)
            res[m_outer] = 2.0 * self.kappa_s * (1.0 - f) / (xo**2 - 1.0)

        m_trans = (~m_inner) & (~m_outer)
        if np.any(m_trans):
            # Limit as x -> 1 is 2/3 * kappa_s
            res[m_trans] = (2.0 / 3.0) * self.kappa_s

        if np.isscalar(x) and np.isscalar(y):
            return float(res)
        return res

    def potential(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        dx = np.asarray(x, dtype=float) - self.xc
        dy = np.asarray(y, dtype=float) - self.yc
        r = np.sqrt(dx**2 + dy**2)
        x_dim = np.asarray(r / self.r_s, dtype=float)
        x_safe = np.maximum(x_dim, 1e-12)

        res = np.empty_like(x_dim)
        tol = 1e-4

        log_sq = 0.5 * (np.log(x_safe / 2.0)) ** 2

        m_inner = x_dim < (1.0 - tol)
        if np.any(m_inner):
            xi = x_safe[m_inner]
            atanh_term = np.arctanh(np.sqrt((1.0 - xi) / (1.0 + xi)))
            res[m_inner] = 4.0 * self.kappa_s * (self.r_s**2) * (log_sq[m_inner] - 2.0 * atanh_term**2)

        m_outer = x_dim > (1.0 + tol)
        if np.any(m_outer):
            xo = x_safe[m_outer]
            atan_term = np.arctan(np.sqrt((xo - 1.0) / (xo + 1.0)))
            res[m_outer] = 4.0 * self.kappa_s * (self.r_s**2) * (log_sq[m_outer] + 2.0 * atan_term**2)

        m_trans = (~m_inner) & (~m_outer)
        if np.any(m_trans):
            res[m_trans] = 4.0 * self.kappa_s * (self.r_s**2) * log_sq[m_trans]

        if np.isscalar(x) and np.isscalar(y):
            return float(res)
        return res
