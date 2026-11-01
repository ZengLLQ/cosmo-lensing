"""Unit tests for cosmological distance calculations."""

import pytest
import numpy as np
from cosmo_lensing.cosmology import FlatLambdaCDM


def test_cosmological_distances():
    cosmo = FlatLambdaCDM(H0=70.0, omega_m=0.3)

    assert cosmo.omega_lambda == pytest.approx(0.7)
    assert cosmo.ez(0.0) == pytest.approx(1.0)
    assert cosmo.hubble_distance_m > 0

    # Distances at z=0 must be zero
    assert cosmo.comoving_distance(0.0) == 0.0
    assert cosmo.angular_diameter_distance(0.0) == 0.0

    # Distances at z > 0 must be positive and monotonic
    dc1 = cosmo.comoving_distance(0.5)
    dc2 = cosmo.comoving_distance(1.0)
    assert dc2 > dc1 > 0

    da1 = cosmo.angular_diameter_distance(0.5)
    da2 = cosmo.angular_diameter_distance(1.0)
    assert da1 > 0
    assert da2 > 0

    # Angular diameter distance between lens (z=0.3) and source (z=1.0)
    d_ls = cosmo.angular_diameter_distance_z1_z2(0.3, 1.0)
    assert d_ls > 0

    # Critical surface mass density (SI units: kg / m^2)
    sigma_cr = cosmo.critical_surface_density(z_lens=0.3, z_source=1.0)
    assert sigma_cr > 0
    assert 1.0 < sigma_cr < 20.0
