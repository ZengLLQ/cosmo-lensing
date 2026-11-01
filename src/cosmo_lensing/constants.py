"""Physical and astronomical constants for gravitational lensing."""

import numpy as np

# Gravitational constant [m^3 kg^-1 s^-2]
G: float = 6.67430e-11

# Speed of light in vacuum [m s^-1]
C: float = 299792458.0

# Solar mass [kg]
M_SUN: float = 1.98847e30

# Megaparsec [m]
MPC: float = 3.085677581e22

# Kiloparsec [m]
KPC: float = MPC * 1e-3

# Angular conversions
ARCSEC_TO_RAD: float = np.pi / (180.0 * 3600.0)
RAD_TO_ARCSEC: float = (180.0 * 3600.0) / np.pi

# Time units
DAY_SECONDS: float = 86400.0
YEAR_SECONDS: float = 365.25 * DAY_SECONDS
