"""Ray-tracing engine and lens equation solver."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy import optimize
from .models.base import BaseLens


@dataclass(frozen=True)
class LensedImage:
    """Represents a resolved gravitational lens image.

    Attributes:
        x: Image position on x-axis [arcsec]
        y: Image position on y-axis [arcsec]
        magnification: Signed image magnification
        time_delay_days: Relative or absolute geometric+gravitational time delay [days]
    """

    x: float
    y: float
    magnification: float
    time_delay_days: float = 0.0


class RayTracer:
    """Ray-tracing simulator mapping image plane theta to source plane beta."""

    def __init__(self, lens: BaseLens):
        self.lens = lens

    def ray_shoot(
        self, theta_x: np.ndarray | float, theta_y: np.ndarray | float
    ) -> Tuple[np.ndarray | float, np.ndarray | float]:
        """Map lens plane coordinates (theta_x, theta_y) to source plane coordinates (beta_x, beta_y).

        beta = theta - alpha(theta)
        """
        ax, ay = self.lens.deflection_angle(theta_x, theta_y)
        beta_x = theta_x - ax
        beta_y = theta_y - ay
        return beta_x, beta_y

    def find_images(
        self,
        beta_x: float,
        beta_y: float,
        grid_extent: float = 5.0,
        grid_res: int = 150,
        tolerance: float = 1e-4,
    ) -> List[LensedImage]:
        """Find all discrete lensed images for a point source at (beta_x, beta_y)."""
        x_lin = np.linspace(-grid_extent, grid_extent, grid_res)
        y_lin = np.linspace(-grid_extent, grid_extent, grid_res)
        xx, yy = np.meshgrid(x_lin, y_lin)

        bx_grid, by_grid = self.ray_shoot(xx, yy)
        dist_sq = (bx_grid - beta_x) ** 2 + (by_grid - beta_y) ** 2

        # Identify local minima on the grid
        found_images: List[LensedImage] = []
        visited_candidates: List[Tuple[float, float]] = []

        # Find cells where distance is below threshold
        candidates = np.argwhere(dist_sq < (grid_extent / grid_res * 2.0) ** 2)

        def objective(theta: np.ndarray) -> np.ndarray:
            ax, ay = self.lens.deflection_angle(theta[0], theta[1])
            return np.array([theta[0] - ax - beta_x, theta[1] - ay - beta_y])

        for iy, ix in candidates:
            init_x = float(xx[iy, ix])
            init_y = float(yy[iy, ix])

            # Check if close to already tested candidates
            if any(np.hypot(init_x - cx, init_y - cy) < tolerance * 10 for cx, cy in visited_candidates):
                continue

            visited_candidates.append((init_x, init_y))

            sol = optimize.root(objective, [init_x, init_y], method="hybr", tol=tolerance)
            if sol.success:
                pos_x, pos_y = float(sol.x[0]), float(sol.x[1])
                residual = np.hypot(
                    pos_x - self.lens.deflection_angle(pos_x, pos_y)[0] - beta_x,
                    pos_y - self.lens.deflection_angle(pos_x, pos_y)[1] - beta_y,
                )

                if residual < tolerance:
                    # Check for duplicates among confirmed images
                    if not any(np.hypot(pos_x - img.x, pos_y - img.y) < tolerance * 5 for img in found_images):
                        mag = float(self.lens.magnification(pos_x, pos_y))
                        found_images.append(LensedImage(x=pos_x, y=pos_y, magnification=mag))

        # Sort images by decreasing absolute magnification
        found_images.sort(key=lambda img: abs(img.magnification), reverse=True)
        return found_images

    def render_extended_source(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        source_profile_fn,
    ) -> np.ndarray:
        """Render lensed surface brightness of an extended source."""
        bx, by = self.ray_shoot(grid_x, grid_y)
        return source_profile_fn(bx, by)
