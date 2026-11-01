"""Unit tests for Navarro-Frenk-White (NFW) lens model."""

import pytest
import numpy as np
from cosmo_lensing.models.nfw import NFWLens


def test_nfw_convergence_continuity():
    kappa_s = 0.3
    r_s = 10.0
    lens = NFWLens(kappa_s=kappa_s, r_s_arcsec=r_s)

    # Test kappa continuity around x = 1.0 (r = r_s = 10.0)
    kappa_at_1 = lens.convergence(10.0, 0.0)
    assert kappa_at_1 == pytest.approx((2.0 / 3.0) * kappa_s, rel=1e-2)

    # Inner region x < 1 (higher density)
    kappa_inner = lens.convergence(5.0, 0.0)
    assert kappa_inner > kappa_at_1

    # Outer region x > 1 (lower density)
    kappa_outer = lens.convergence(20.0, 0.0)
    assert kappa_outer < kappa_at_1


def test_nfw_deflection_and_potential_consistency():
    kappa_s = 0.25
    r_s = 5.0
    lens = NFWLens(kappa_s=kappa_s, r_s_arcsec=r_s)

    # Test potential gradient matches deflection angle
    x, y = 3.5, 4.0
    eps = 1e-5
    dpsi_dx = (lens.potential(x + eps, y) - lens.potential(x - eps, y)) / (2.0 * eps)
    dpsi_dy = (lens.potential(x, y + eps) - lens.potential(x, y - eps)) / (2.0 * eps)

    ax, ay = lens.deflection_angle(x, y)
    assert ax == pytest.approx(dpsi_dx, rel=1e-3)
    assert ay == pytest.approx(dpsi_dy, rel=1e-3)
