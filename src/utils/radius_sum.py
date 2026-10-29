"""Geometry-cached radial/spherical Fourier bin summation.

Coordinates and NumPy's round-to-even rule intentionally match the original
implementation. Shared immutable bin arrays avoid repeated meshgrid creation.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=12)
def _radial_bins(height, width, radial_step):
    y = np.arange(1, height + 1) - height / 2
    x = np.arange(1, width + 1) - width / 2
    bins = (np.rint(np.hypot(y[:, None], x[None, :]) / radial_step)
            .astype(np.int32) + 1).ravel()
    bins.flags.writeable = False
    return bins, int(bins.max())


def _make_spherical_bins(depth, height, width, radial_step):
    z = np.arange(1, depth + 1) - depth / 2
    y = np.arange(1, height + 1) - height / 2
    x = np.arange(1, width + 1) - width / 2
    radii2 = (z[:, None, None] ** 2 + y[None, :, None] ** 2
              + x[None, None, :] ** 2)
    bins = (np.rint(np.sqrt(radii2) / radial_step)
            .astype(np.int32) + 1).ravel()
    bins.flags.writeable = False
    return bins, int(bins.max())


@lru_cache(maxsize=2)
def _cached_spherical_bins(depth, height, width, radial_step):
    return _make_spherical_bins(depth, height, width, radial_step)


def spherical_bins(shape, radial_step=1):
    """Return centered, rounded 3D shell indices (1 based) and max index."""
    shape = tuple(map(int, shape))
    if len(shape) != 3:
        raise ValueError('Expected a 3D shape')
    # Do not keep multiple giant volumes in the global cache.
    if np.prod(shape) > 2_000_000:
        return _make_spherical_bins(*shape, float(radial_step))
    return _cached_spherical_bins(*shape, float(radial_step))


def _sum_binned(data, indices, max_idx):
    values = np.asarray(data).ravel()
    real = np.bincount(indices, weights=np.real(values), minlength=max_idx + 1)
    if np.iscomplexobj(values):
        imag = np.bincount(indices, weights=np.imag(values), minlength=max_idx + 1)
        return (real + 1j * imag)[1:max_idx + 1]
    # Preserve complex result type of original utility for real inputs.
    return real[1:max_idx + 1].astype(np.complex128)


def radial_sum(data, radial_step=1):
    """Sum a 2D array into original round-to-even radial bins."""
    bins, end = _radial_bins(*data.shape, float(radial_step))
    return _sum_binned(data, bins, end)


def spherical_sum(data, radial_step=1):
    """Sum a 3D array into original round-to-even spherical shells."""
    bins, end = spherical_bins(data.shape, radial_step)
    return _sum_binned(data, bins, end)
