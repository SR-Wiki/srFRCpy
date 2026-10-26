"""Floating-point preparation of image arrays; no file I/O is performed here."""
import numpy as np


def normalize_image(image):
    """Copy an image, stack or volume to float32 and divide by its maximum.

    Parameters
    ----------
    image : array_like
        Nonempty, finite, real image data. A stack or volume shares one maximum,
        so relative intensities between its slices are preserved.

    Returns
    -------
    numpy.ndarray
        Independent float32 data. A zero maximum leaves the values unchanged,
        keeping an all-zero input finite. The caller's array is never modified.
    """
    source = np.asarray(image)
    if source.dtype.kind not in 'buif' or source.size == 0:
        raise ValueError('Provide a nonempty, real numeric image, stack or volume.')
    data = np.array(source, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(data)):
        raise ValueError('Image intensities must be finite; remove NaN or infinite values before analysis.')
    maximum = np.max(data)
    if maximum != 0:
        data /= maximum
    return data
