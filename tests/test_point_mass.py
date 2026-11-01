"""Unit tests for Point Mass / Schwarzschild lens model."""

import pytest
import numpy as np
from cosmo_lensing.models.point_mass import PointMassLens


def test_point_mass_deflection_and_potential():
    theta_e = 1.5
    lens = PointMassLens(theta_e=theta_e)

    # On x-axis at x=3.0, alpha_x should be theta_e^2 / x = 2.25 / 3.0 = 0.75
    ax, ay = lens.deflection_angle(3.0, 0.0)
    assert ax == pytest.approx(0.75)
    assert ay == pytest.approx(0.0)

    # Test numerical gradient of potential equals deflection angle
    x, y = 2.0, 1.5
    eps = 1e-6
    dpsi_dx = (lens.potential(x + eps, y) - lens.potential(x - eps, y)) / (2.0 * eps)
    dpsi_dy = (lens.potential(x, y + eps) - lens.potential(x, y - eps)) / (2.0 * eps)

    ax_eval, ay_eval = lens.deflection_angle(x, y)
    assert ax_eval == pytest.approx(dpsi_dx, rel=1e-4)
    assert ay_eval == pytest.approx(dpsi_dy, rel=1e-4)


def test_point_mass_analytic_images():
    theta_e = 1.0
    lens = PointMassLens(theta_e=theta_e)

    beta_x, beta_y = 0.5, 0.0
    images = lens.analytic_images(beta_x, beta_y)

    assert len(images) == 2
    # Verify lens equation: theta - alpha(theta) = beta
    for ix, iy, mag in images:
        ax, ay = lens.deflection_angle(ix, iy)
        assert (ix - ax) == pytest.approx(beta_x, abs=1e-5)
        assert (iy - ay) == pytest.approx(beta_y, abs=1e-5)
