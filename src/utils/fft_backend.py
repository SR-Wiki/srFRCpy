"""FFT backend used by the numerical sFRC/sFSC/srFRC implementations.

The SciPy backend is the default for performance and uses its *native* input
precision: float32 -> complex64 and float64 -> complex128. In particular, no
promotion is performed to mimic older NumPy FFT dtype behavior.

Set ``SRFRC_FFT_BACKEND=numpy`` before importing this module to use the NumPy
implementation instead. Different FFT implementations and input precisions can
cause small rounding differences near an FRC/FSC threshold crossing.
"""
import os
import numpy as np
from scipy import fft as _scipy_fft

_BACKEND = os.environ.get('SRFRC_FFT_BACKEND', 'scipy').strip().lower()
if _BACKEND not in ('numpy', 'scipy'):
    raise ValueError('SRFRC_FFT_BACKEND must be numpy or scipy')


def fft2(array):
    if _BACKEND == 'numpy':
        return np.fft.fft2(array)
    return _scipy_fft.fft2(array)


def fftn(array):
    if _BACKEND == 'numpy':
        return np.fft.fftn(array)
    return _scipy_fft.fftn(array)


def fftshift(array):
    return np.fft.fftshift(array) if _BACKEND == 'numpy' else _scipy_fft.fftshift(array)
