"""Backward-compatible names for the consolidated ResolutionPlotter API."""
import numpy as np
from src.utils.plotting import ResolutionPlotter


def radians_to_degrees(angle):
    """Convert radians to degrees (legacy helper)."""
    return 0 if angle == 0 else angle * 180.0 / np.pi


def polar_plot(erf, use_valid=False, save_path=None, show=False):
    """Compatibility wrapper; use ResolutionPlotter.sfsc_polar directly."""
    return ResolutionPlotter.sfsc_polar(erf, use_valid=use_valid, save_path=save_path, show=show)


def polar_plot_V0(xy_resolution, z_resolution, save_path=None, show=False):
    """Compatibility wrapper for legacy scalar XY/Z polar plots."""
    return ResolutionPlotter.sfsc_polar_legacy(xy_resolution, z_resolution,
                                               save_path=save_path, show=show)
