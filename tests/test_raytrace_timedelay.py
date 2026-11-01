"""Unit tests for Ray-tracing, Jacobian, and Time Delay calculations."""

import pytest
import numpy as np
from cosmo_lensing.models.point_mass import PointMassLens
from cosmo_lensing.models.sis import SISLens
from cosmo_lensing.raytrace import RayTracer
from cosmo_lensing.magnification import CriticalCurveAnalyzer
from cosmo_lensing.cosmology import FlatLambdaCDM
from cosmo_lensing.timedelay import TimeDelayCalculator


def test_ray_tracer_point_mass_images():
    theta_e = 1.0
    lens = PointMassLens(theta_e=theta_e)
    tracer = RayTracer(lens)

    beta_x, beta_y = 0.6, 0.0
    images = tracer.find_images(beta_x, beta_y, grid_extent=3.0, grid_res=100)

    assert len(images) == 2
    # Analytic expected positions
    analytic_imgs = lens.analytic_images(beta_x, beta_y)
    pos_x = sorted([img.x for img in images])
    expected_x = sorted([img[0] for img in analytic_imgs])

    assert pos_x[0] == pytest.approx(expected_x[0], abs=1e-3)
    assert pos_x[1] == pytest.approx(expected_x[1], abs=1e-3)


def test_jacobian_and_critical_curves():
    theta_e = 1.0
    lens = SISLens(theta_e=theta_e)
    analyzer = CriticalCurveAnalyzer(lens)

    # For SIS, tangential critical curve is at theta = theta_e = 1.0 where lambda_t = 1 - 2*kappa = 0
    jac = analyzer.compute_jacobian(1.0, 0.0)
    assert jac.lambda_tangential == pytest.approx(0.0, abs=1e-2)
    assert abs(jac.det_a) < 0.05

    # Test critical curve extraction
    crit_x, crit_y, caust_x, caust_y = analyzer.extract_critical_curves(grid_extent=2.5, grid_res=100)
    assert len(crit_x) > 0
    # For SIS, the caustic maps to the central point (0, 0)
    assert np.all(np.hypot(caust_x, caust_y) < 0.2)


def test_time_delay_calculation():
    theta_e = 1.0
    lens = PointMassLens(theta_e=theta_e)
    cosmo = FlatLambdaCDM(H0=70.0, omega_m=0.3)
    td_calc = TimeDelayCalculator(lens, cosmo, z_lens=0.5, z_source=1.5)

    beta_x, beta_y = 0.5, 0.0
    analytic_imgs = lens.analytic_images(beta_x, beta_y)
    img_a = (analytic_imgs[0][0], analytic_imgs[0][1])
    img_b = (analytic_imgs[1][0], analytic_imgs[1][1])

    delta_t = td_calc.relative_time_delay_days(img_a, img_b, (beta_x, beta_y))
    # Time delay between multiple images should be non-zero and finite (typically tens of days)
    assert delta_t != 0.0
    assert np.isfinite(delta_t)


def test_extended_sersic_source_rendering():
    theta_e = 1.2
    lens = SISLens(theta_e=theta_e)
    tracer = RayTracer(lens)

    # Gaussian source profile
    def source_profile(bx, by):
        return np.exp(-((bx - 0.2) ** 2 + by**2) / (2 * 0.1**2))

    x = np.linspace(-2.0, 2.0, 50)
    y = np.linspace(-2.0, 2.0, 50)
    xx, yy = np.meshgrid(x, y)

    surface_brightness = tracer.render_extended_source(xx, yy, source_profile)
    assert surface_brightness.shape == (50, 50)
    assert np.all(surface_brightness >= 0.0)
    assert np.max(surface_brightness) > 0.5
