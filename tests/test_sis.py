"""Unit tests for Singular Isothermal Sphere (SIS) lens model."""

import pytest
import numpy as np
from cosmo_lensing.models.sis import SISLens


def test_sis_properties():
    theta_e = 1.2
    lens = SISLens(theta_e=theta_e)

    # Deflection angle magnitude is always theta_e
    ax, ay = lens.deflection_angle(3.0, 4.0)
    alpha_mag = np.hypot(ax, ay)
    assert alpha_mag == pytest.approx(theta_e)

    # Convergence at radius r: kappa = theta_e / (2 * r)
    r = 2.0
    kappa = lens.convergence(r, 0.0)
    assert kappa == pytest.approx(theta_e / (2.0 * r))


def test_sis_analytic_images():
    theta_e = 1.0
    lens = SISLens(theta_e=theta_e)

    # Source inside Einstein radius (beta < theta_e) -> 2 images
    beta_inside = 0.4
    images_inside = lens.analytic_images(beta_inside, 0.0)
    assert len(images_inside) == 2

    # Check lens equation for both images
    for ix, iy, mag in images_inside:
        ax, ay = lens.deflection_angle(ix, iy)
        assert (ix - ax) == pytest.approx(beta_inside, abs=1e-5)
        assert (iy - ay) == pytest.approx(0.0, abs=1e-5)

    # Source outside Einstein radius (beta > theta_e) -> 1 image
    beta_outside = 1.5
    images_outside = lens.analytic_images(beta_outside, 0.0)
    assert len(images_outside) == 1
