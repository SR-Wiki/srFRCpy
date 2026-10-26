import math
import numpy as np
import tifffile
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
from typing import Tuple, Dict, Optional
from src.utils.radius_sum import radial_sum, spherical_sum
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

    y, x = np.meshgrid(np.arange(-h/2+1, h/2+1), np.arange(-w/2+1, w/2+1))

    max_r = min(w - xc, h - yc) + 1
    num_pixels = radial_sum(np.ones_like(ffta.real), 1).real
    num_pixels = num_pixels[:max_r]

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


def perform_fsc(ffta, fftb, pixel_size=32.5, single_frame=False, z_sampling=False,
                mean_filter_width=3, extrapolation=False):
    """Build a Fourier shell correlation result from two centered 3D FFTs.
    
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
    z_sampling : bool
        Whether odd/even Z sampling and its associated correction are enabled.
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
        Shell correlation data and frequency coordinates for a scalar or masked
        directional analysis.
    """
    d, h, w = ffta.shape
    zc, yc, xc = math.ceil((d + 1) / 2), math.ceil((h + 1) / 2), math.ceil((w + 1) / 2)
    z, y, x = np.meshgrid(np.arange(-d/2+1, d/2+1), np.arange(-h/2+1, h/2+1), np.arange(-w/2+1, w/2+1))

    max_r = min(w - xc, h - yc, d - zc) + 1
    num_pixels = spherical_sum(np.ones_like(ffta.real), 1).real
    num_pixels = num_pixels[:max_r]

    numerator = ffta * np.conj(fftb)
    de1 = spherical_sum(np.abs(ffta) ** 2, 1)
    de2 = spherical_sum(np.abs(fftb) ** 2, 1)
    top = np.abs(spherical_sum(numerator, 1))
    bot = np.sqrt(de1.real * de2.real)
    frc = top / (bot + 1e-4)
    # plt.plot(np.arange(0, d), frc)
    # plt.plot(np.arange(0, d), 1.0/7*np.ones_like(frc))
    # plt.show()
    frc = refine(frc[:max_r])
    smoothed_frc = mean_filter(frc, mean_filter_width)
    smoothed_frc_corrected = apply_correction(smoothed_frc)
    if z_sampling:
        smoothed_frc_corrected = z_correction(smoothed_frc_corrected)

    k = 1000 / (pixel_size * 2) / max_r
    frequency = np.arange(max_r) * k
    fixed_threshold = 1 / 7 * np.ones(max_r)

    return FourierCorrelationData(frequency, smoothed_frc, smoothed_frc_corrected,
                                  fixed_threshold, pixel_size, num_pixels,
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


def apply_correction(single_frc):
    """Correct diagonal single-frame correlations using the sampling transfer factors.
    
    Parameters
    ----------
    single_frc : numpy.ndarray
        One-dimensional measured correlation curve.
    
    Returns
    -------
    numpy.ndarray
        Corrected correlation values, with the measured tail retained after the
        corrected curve reaches 1/7.
    """
    w = 2 * len(single_frc)
    corrected = np.zeros_like(single_frc)
    ad = np.zeros_like(single_frc)
    x = np.arange(-w // 2 + 1, w // 2 + 1)
    y = np.arange(-w // 2 + 1, w // 2 + 1)
    x_grid, y_grid = np.meshgrid(x, y)
    radii = np.sqrt(x_grid*x_grid+y_grid*y_grid)
    theta1 = x_grid * np.pi / w
    theta2 = y_grid * np.pi / w
    real_response = np.real(np.exp(-1j * theta1) + np.exp(-1j * theta2))
    x1 = np.exp(-1j * (theta1 + theta2)) + 1
    x2 = np.exp(-1j * (theta1 + theta2)) - 1
    y1 = np.exp(-1j * theta1) + np.exp(-1j * theta2)
    y2 = np.exp(-1j * theta1) - np.exp(-1j * theta2)
    x1 = np.abs(x1) ** 2
    x2 = np.abs(x2) ** 2
    y1 = np.abs(y1) ** 2
    y2 = np.abs(y2) ** 2
    correction_factor = np.sqrt(x1 * y1 + x2 * y2 + x1 * y2 + x2 * y1)
    div = correction_factor / real_response
    div[np.isnan(div)] = 0
    div[np.isinf(div)] = 0
    div[radii > w // 2] = 0

    for r in range(w // 2):
        diff1 = np.abs(x_grid ** 2 + y_grid ** 2 - r ** 2)
        diff = np.abs(x_grid ** 2 + y_grid ** 2 - (r + 1) ** 2)
        diff2 = np.abs(x_grid ** 2 + y_grid ** 2 - (r + 2) ** 2)
        mask1 = diff1 <= diff
        mask2 = diff <= diff2
        ring_set = mask1 * mask2
        cur_div = np.mean(div[ring_set])
        ad[r] = cur_div
        tmp = single_frc[r]
        # if cur_div > 1:
        #     corrected[r] = (cur_div * tmp + 1e-4) / (1 + (cur_div - 1) * tmp + 1e-4)
        # else:
        #     corrected[r] = single_frc[r]
        corrected[r] = (1 + cur_div + 1e-4) * tmp / (1 + cur_div * tmp + 1e-4)
        if corrected[r] <= 1/7:
            corrected[r + 1:] = single_frc[r + 1:]
            # print(f'Ending factor: {ad[r]}')
            # print(f'Ending r: {r}')
            break
    corrected[corrected > 1] = 1
    corrected[corrected < 0] = -corrected[corrected < 0]
    return corrected
