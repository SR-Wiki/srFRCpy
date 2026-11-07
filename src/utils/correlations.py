import math
import os
from functools import lru_cache
import numpy as np
import tifffile
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
from typing import Tuple, Dict, Optional
from src.utils.radius_sum import radial_sum, spherical_sum, _radial_bins, spherical_bins
from src.utils.data_smoothing import mean_filter, refine
from src.utils.fourier_correlation_data import FourierCorrelationData
from src.utils.angle_mask import AngleMask
from tqdm import tqdm
import matplotlib.pyplot as plt


def perform_frc(ffta, fftb, pixel_size=32.5, single_frame=False, mean_filter_width=3, extrapolation=False):
    """Build a Fourier ring correlation result from two centered 2D FFTs.
    
    Parameters
    ----------
    ffta : numpy.ndarray
        Centered complex Fourier transform of the first input.
    fftb : numpy.ndarray
        Centered complex Fourier transform of the second input.
    pixel_size : float
        Positive calibrated pixel spacing in nm. {default: 32.5}
    single_frame : bool
        Use the correlation correction for samples derived from one acquisition.
        {default: False}
    mean_filter_width : int
        Moving-mean neighborhood radius in frequency bins; must be at least 1.
        {default: 3}
    extrapolation : bool
        Internal permission to attempt curve extrapolation for the fixed 1/7 threshold.
        {default: False}
    
    Returns
    -------
    FourierCorrelationData
        Measured/corrected curves, 1/7 and sigma thresholds, and frequency coordinates.
        Call calculate_resolution to evaluate crossings.
    """
    h, w = ffta.shape
    yc, xc = math.ceil((h + 1) / 2), math.ceil((w + 1) / 2)


    max_r = min(w - xc, h - yc) + 1
    bins, max_bin = _radial_bins(h, w, 1.0)
    num_pixels = np.bincount(bins, minlength=max_bin + 1)[1:max_r + 1].astype(float)

    numerator = ffta * np.conj(fftb)
    de1 = radial_sum(np.abs(ffta) ** 2, 1)
    de2 = radial_sum(np.abs(fftb) ** 2, 1)
    top = radial_sum(numerator, 1)
    bot = np.sqrt(de1.real * de2.real)
    frc = np.abs(top) / (bot + 1e-4)
    frc = refine(frc[:max_r])
    smoothed_frc = mean_filter(frc, mean_filter_width)
    smoothed_frc_corrected = apply_correction(smoothed_frc)

    k = 1000 / (pixel_size * 2) / max_r
    frequency = np.arange(max_r) * k
    fixed_threshold = 1 / 7 * np.ones(max_r)

    return FourierCorrelationData(frequency, smoothed_frc, smoothed_frc_corrected, fixed_threshold,
                                  pixel_size, num_pixels, single_frame=single_frame, extrapolation=extrapolation)


class FSCWorkspace:
    """Reuse Fourier products and spherical geometry across directional sectors."""
    def __init__(self, ffta, fftb):
        if ffta.shape != fftb.shape or ffta.ndim != 3:
            raise ValueError('FSCWorkspace needs matching 3D Fourier volumes')
        self.shape = ffta.shape
        d, h, w = self.shape
        self.max_r = min(w - math.ceil((w + 1) / 2),
                         h - math.ceil((h + 1) / 2),
                         d - math.ceil((d + 1) / 2)) + 1
        self.bins, self.max_bin = spherical_bins(self.shape)
        self.counts = np.bincount(self.bins, minlength=self.max_bin + 1)[1:self.max_r + 1].astype(float)
        a = np.asarray(ffta).ravel()
        b = np.asarray(fftb).ravel()
        self.cross = a * np.conj(b)
        self.power_a = np.abs(a) ** 2
        self.power_b = np.abs(b) ** 2

    def curves(self, mask=None):
        if mask is None:
            bins = self.bins
            cross = self.cross
            power_a = self.power_a
            power_b = self.power_b
        else:
            selected = np.asarray(mask, dtype=bool).ravel()
            bins = self.bins[selected]
            cross = self.cross[selected]
            power_a = self.power_a[selected]
            power_b = self.power_b[selected]
        count = self.max_bin + 1
        numerator_real = np.bincount(bins, weights=cross.real, minlength=count)
        numerator_imag = np.bincount(bins, weights=cross.imag, minlength=count)
        de1 = np.bincount(bins, weights=power_a, minlength=count)
        de2 = np.bincount(bins, weights=power_b, minlength=count)
        top = np.hypot(numerator_real[1:], numerator_imag[1:])
        bot = np.sqrt(de1[1:] * de2[1:])
        return top / (bot + 1e-4)


def perform_fsc(ffta, fftb, pixel_size=32.5, single_frame=False, z_sampling=False,
                mean_filter_width=3, extrapolation=False, mask=None, workspace=None):
    """Perform FSC; an optional shared workspace avoids per-cone 3D geometry/FFT copies.

    The mask is applied to Fourier products before their spherical accumulation.
    The original all-volume shell population is retained for sigma thresholds.
    """
    work = workspace if workspace is not None else FSCWorkspace(ffta, fftb)
    if work.shape != ffta.shape or fftb.shape != ffta.shape:
        raise ValueError('FSC workspace shape does not match input FFTs')
    max_r = work.max_r
    frc = refine(work.curves(mask)[:max_r])
    smoothed_frc = mean_filter(frc, mean_filter_width)
    smoothed_frc_corrected = apply_correction(smoothed_frc)
    if z_sampling:
        smoothed_frc_corrected = z_correction(smoothed_frc_corrected)
    frequency = np.arange(max_r) * (1000 / (pixel_size * 2) / max_r)
    fixed_threshold = 1 / 7 * np.ones(max_r)
    return FourierCorrelationData(frequency, smoothed_frc, smoothed_frc_corrected,
                                  fixed_threshold, pixel_size, work.counts,
                                  single_frame=single_frame, extrapolation=extrapolation)

def refine(pre):
    """Repair the initial correlation segment and isolated near-zero samples.
    
    Parameters
    ----------
    pre : numpy.ndarray
        One-dimensional correlation samples before refinement.
    
    Returns
    -------
    numpy.ndarray
        Refined floating-point correlation samples.
    """
    refined = np.array(pre, dtype=float)
    max_val = np.max(pre)
    max_indices = np.where(pre > 1e-8)[0]
    if len(max_indices) > 0:
        idx = max_indices[0]
        if idx > 0:
            k = (1 - max_val) / idx
            x = np.arange(0, idx)  # Indices from 0 to idx - 1.
            refined[:idx] = 1 - k * x
    n = len(refined)
    for i in range(1, n - 1):
        if refined[i] < 1e-4:
            if refined[i - 1] > 0.9 and refined[i + 1] > 0.9:
                refined[i] = 1.0
    return refined


def z_correction(single_frc):
    """Apply the single-volume Z-sampling correlation correction.
    
    Parameters
    ----------
    single_frc : numpy.ndarray
        One-dimensional measured correlation curve.
    
    Returns
    -------
    numpy.ndarray
        Corrected samples computed as 2*correlation/(1+correlation).
    """
    return 2 * single_frc / (1.0 + single_frc)


# The default geometry is the same rounded Euclidean radius used by Java:
#     round(sqrt(X**2 + Y**2)) == r, r=1..n.
# Keep explicit legacy modes for regression only; all ordinary calls (including
# multiprocessing workers) use the rounded-radius selection by default.
DEFAULT_CORRECTION_GEOMETRY = os.getenv("SRFRC_CORRECTION_GEOMETRY", "rounded_radius").strip().lower()
if DEFAULT_CORRECTION_GEOMETRY not in {"legacy_disk", "matlab_squared", "rounded_radius"}:
    raise ValueError("Invalid SRFRC_CORRECTION_GEOMETRY: " + DEFAULT_CORRECTION_GEOMETRY)

@lru_cache(maxsize=16)
def _correction_factors(n, geometry="rounded_radius"):
    """Precompute correction factors in O(n**2), with explicit radial geometry.

    ``matlab_squared`` reproduces the MATLAB *selection rule*
        round(X**2 + Y**2) == r**2, r = 1, ..., n.
    Coordinates are integers, so MATLAB's round and NumPy's rint act as
    identities here. This selects exact discrete circles (not filled disks).

    ``rounded_radius`` (default) selects round(sqrt(X**2 + Y**2)) == r,
    r = 1, ..., n, as in Java and the updated MATLAB implementation. NumPy's ties-to-even
    rounding does not matter: the squared distances are integers and cannot
    produce a nonzero half-integer radius.

    ``legacy_disk`` is an explicit backward-compatibility option preserving the previous Python cumulative disk
        X**2 + Y**2 <= r*(r+1), r = 0, ..., n-1.

    Only the sampling region changes between modes. The Python numerical
    response and correction/tail formula deliberately remain unchanged; this
    DOES NOT make the outputs of MATLAB/Python/Java mathematically identical.
    """
    if geometry not in {"matlab_squared", "rounded_radius", "legacy_disk"}:
        raise ValueError("geometry must be matlab_squared, rounded_radius, or legacy_disk")
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    w = 2 * n
    x = np.arange(-n + 1, n + 1, dtype=np.int64)
    xx, yy = x[None, :], x[:, None]
    radius_squared = xx * xx + yy * yy
    theta1 = xx * np.pi / w
    theta2 = yy * np.pi / w
    e1 = np.exp(-1j * theta1)
    e2 = np.exp(-1j * theta2)
    e12 = np.exp(-1j * (theta1 + theta2))
    real_response = np.real(e1 + e2)
    x1 = np.abs(e12 + 1) ** 2
    x2 = np.abs(e12 - 1) ** 2
    y1 = np.abs(e1 + e2) ** 2
    y2 = np.abs(e1 - e2) ** 2
    correction_factor = np.sqrt(x1 * y1 + x2 * y2 + x1 * y2 + x2 * y1)
    with np.errstate(divide='ignore', invalid='ignore'):
        div = correction_factor / real_response
    div[~np.isfinite(div)] = 0

    if geometry == "legacy_disk":
        largest_disk = (n - 1) * n
        selected = radius_squared <= largest_disk
        squared_radii = radius_squared[selected]
        responses = div[selected]
        shell_sums = np.bincount(
            squared_radii, weights=responses, minlength=largest_disk + 1)
        shell_counts = np.bincount(
            squared_radii, minlength=largest_disk + 1)
        prefix_sums = np.cumsum(shell_sums)
        prefix_counts = np.cumsum(shell_counts)
        r = np.arange(n, dtype=np.int64)
        limits = r * (r + 1)
        ad = prefix_sums[limits] / prefix_counts[limits]
    elif geometry == "matlab_squared":
        # MATLAB: round(X.^2+Y.^2)==r.^2.  X,Y are integer coordinates.
        # No need to materialize one 2D mask per radius or evaluate sqrt.
        max_sq = n * n
        selected = radius_squared <= max_sq
        integer_squared_radii = radius_squared[selected]  # == np.rint(...)
        weights = div[selected]
        sums = np.bincount(
            integer_squared_radii, weights=weights, minlength=max_sq + 1)
        counts = np.bincount(
            integer_squared_radii, minlength=max_sq + 1)
        r = np.arange(1, n + 1, dtype=np.int64)
        ad = sums[r * r] / counts[r * r]
    else:  # rounded_radius, like MATLAB radial_sum and the previous Java
        selected = radius_squared <= (n + 1) * (n + 1)
        bins = np.rint(np.sqrt(radius_squared[selected])).astype(np.intp)
        sums = np.bincount(bins, weights=div[selected], minlength=n + 1)
        counts = np.bincount(bins, minlength=n + 1)
        ad = sums[1:n + 1] / counts[1:n + 1]
    ad.flags.writeable = False
    return ad


def apply_correction(single_frc, geometry=None):
    """Apply the Python correction/tail formula with selectable radial geometry.

    Default rounded_radius uses the common Python/Java/MATLAB selection
    round(sqrt(X**2 + Y**2)) == r. Explicit legacy_disk/matlab_squared
    options are retained only to compare with prior releases.

    Operations are evaluated before the first cutoff only in the original
    function, but the vectorized result restores all samples after it. The
    input dtype is retained, including the float32 rounding before threshold
    detection. Cached factors are read-only and shareable across calls.
    """
    if geometry is None:
        geometry = DEFAULT_CORRECTION_GEOMETRY
    values = np.asarray(single_frc)
    if values.size == 0:
        return np.zeros_like(values)
    ad = _correction_factors(len(values), geometry)
    corrected = np.array(
        (ad * values + 1e-4) / (1 + (ad-1) * values + 1e-4),
        dtype=values.dtype,
    )
    crossing = np.flatnonzero(corrected <= 1 / 7)
    if crossing.size:
        start = int(crossing[0]) + 1
        corrected[start:] = values[start:]
    corrected[corrected > 1] = 1
    corrected[corrected < 0] = -corrected[corrected < 0]
    return corrected
