"""SN2N Fourier interpolation adapted for quantitative 2D resolution maps.

Retained as a standalone reference utility. The maintained srFRC map-restoration
path uses cubic splines and does not call this module.

Source: uploaded SN2N-main(1).zip, SN2N/SN2N/datagen.py,
generator2D.fourier_inter (lines 341-388) and interpft (lines 390-435).
Source SHA256: 5b49367d5eb7003c30af525c6a52dd2ce0596c1fd2b57280eca94b0f589a487d.
Attribution and the supplied license are in third_party/SN2N/.

The FFT assembly, original even-length branch, irfft and ny/m scaling are
retained. The wrapper uses each axis's own symmetric padding for rectangular
maps and at least one padding pixel for singleton axes. It returns a 2D map
rather than allocating generator training patches or adding a stack axis.
"""
import numpy as np

BACKEND = 'SN2N_fourier_inter'


def interpft(x, ny, dim=0):
    """Apply the SN2N Fourier interpolation kernel along one array axis.
    
    Parameters
    ----------
    x : numpy.ndarray
        Input 1D or 2D samples for Fourier interpolation.
    ny : int
        Target sample count along the interpolated axis; must not be smaller than its
        input length.
    dim : int
        Axis to interpolate: 0 or 1 for a 2D array. {default: 0}
    
    Returns
    -------
    numpy.ndarray
        Real-valued interpolated samples at the requested length.
    
    Notes
    -----
    The existing SN2N FFT padding, even-length Nyquist handling, inverse real FFT, and
    amplitude scaling are retained. A 1D input is supported along axis 0.
    """
    if dim >= 1:
        x = np.swapaxes(x, 0, dim)
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    siz = x.shape
    m, n = x.shape
    a = np.fft.fft(x, m, 0)
    nyqst = int(np.ceil((m + 1) / 2))
    b = np.concatenate((a[0:nyqst, :], np.zeros(shape=(ny - m, n)), a[nyqst:m, :]), 0)
    if np.remainder(m, 2) == 0:
        b[nyqst, :] = b[nyqst, :] / 2
        b[nyqst + ny - m, :] = b[nyqst, :]
    y = np.fft.irfft(b, b.shape[0], 0)
    y = y * ny / m
    y = np.reshape(y, [y.shape[0], siz[1]])
    y = np.squeeze(y)
    if dim >= 1:
        y = np.swapaxes(y, 0, dim)
    return y


def fourier_inter(image):
    """Upsample a 2D map by two using the SN2N Fourier interpolation procedure.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image data to process.
    
    Returns
    -------
    numpy.ndarray
        Map with exactly twice the input height and width; negative interpolation values
        are set to zero.
    
    Notes
    -----
    Uses symmetric extension, successive axis interpolation, and the SN2N crop.
    Singleton-axis handling supports very small srFRC maps.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 1:
        raise ValueError('SN2N Fourier interpolation expects a nonempty 2D map.')
    imgsz = np.array(image.shape)
    # Preserve the source's sz/idx calculation, including its float division.
    tem1 = np.divide(imgsz, 2)
    tem2 = np.multiply(tem1, 2)
    tem3 = np.subtract(imgsz, tem2)
    b = tem3 == np.array([0, 0])
    sz = imgsz - 1 if b[0] else imgsz
    n = np.array([2, 2])
    idx = (np.ceil(np.divide(sz, 2)) + 1
           + np.floor(np.divide(sz, 2)) * (n - 1)).astype(int)
    padsize = np.maximum(1, np.array(imgsz / 2, dtype=int))
    padded = np.pad(image, ((padsize[0], padsize[0]),
                            (padsize[1], padsize[1])), mode='symmetric')
    newsz = np.round(n * np.array(padded.shape) - (n - 1)).astype(int)
    result = interpft(padded, newsz[0], 0)
    result = interpft(result, newsz[1], 1)
    # One-pixel source axes need the crop offset for their nonzero padding.
    idx = np.where(imgsz == 1, 2 * padsize + 1, idx)
    start = idx - 1
    result = result[start[0]:start[0] + 2 * imgsz[0],
                    start[1]:start[1] + 2 * imgsz[1]].copy()
    if result.shape != tuple(2 * imgsz):
        raise ValueError('SN2N Fourier crop did not produce the requested 2x map.')
    result[result < 0] = 0  # Retain SN2N's negative-value cleanup.
    return result


def restore_size(image, target_shape):
    """Restore an srFRC map to the original acquisition dimensions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image data to process.
    target_shape : tuple of int
        Positive target image dimensions (Y,X).
    
    Returns
    -------
    numpy.ndarray
        Map with shape target_shape; unchanged-size input is returned as a copy.
    
    Notes
    -----
    A changed-size input must equal the half-size grid of the target. Odd target
    dimensions are handled by symmetric extension before the 2x SN2N interpolation and
    final crop.
    """
    image = np.asarray(image, dtype=float)
    target = tuple(target_shape)
    if image.ndim != 2 or len(target) != 2 or min(image.shape) < 1:
        raise ValueError('Expected a nonempty 2D map and a 2D target shape.')
    if any(not isinstance(size, (int, np.integer)) or size < 1 for size in target):
        raise ValueError('Target dimensions must be positive integers.')
    if image.shape == target:
        return image.copy()
    expected = tuple(size // 2 for size in target)
    if image.shape != expected:
        raise ValueError('SN2N restoration expects a half-size srFRC map or an unchanged paired map.')
    # For an odd original size, mirror one final low-grid row/column before
    # applying the same 2x interpolation, then crop only the high edge.
    extra = tuple(size % 2 for size in target)
    if any(extra):
        image = np.pad(image, ((0, extra[0]), (0, extra[1])), mode='symmetric')
    return fourier_inter(image)[:target[0], :target[1]]
