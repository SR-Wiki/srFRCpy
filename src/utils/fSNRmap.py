# -*- coding: utf-8 -*-
import numpy as np
import math
import numpy.fft as ft
from sys import float_info
from src.utils.window_functions import hanning_2d
from src.utils.drift_correct import drift_detect
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2, ifftshift
from src.utils.correlations import perform_frc


def fSNRmap(stack, pixel_size=30.25, background_intensity=5, skip=1, block_size=64,
            single_frame=False, correct_drift=False, amedianfilter=True):
    """Describe the retained legacy paired-image rolling quality-map routine.
    
    Parameters
    ----------
    stack : numpy.ndarray
        Paired images with shape (2,Y,X).
    pixel_size : float
        Positive calibrated pixel spacing in nm. {default: 30.25}
    background_intensity : float
        Background intensity on the 0-255 scale. {default: 5}
    skip : int
        Positive spatial step between rolling windows, in pixels. {default: 1}
    block_size : int
        Side length of each rolling correlation window in pixels. {default: 64}
    single_frame : bool
        Use the correlation correction for samples derived from one acquisition.
        {default: False}
    correct_drift : bool
        Enable the legacy drift-correction path. {default: False}
    amedianfilter : bool
        Apply the legacy adaptive median-filter step. {default: True}
    
    Returns
    -------
    numpy.ndarray
        Legacy resolution-map array in nm, when the historical routine completes.
    
    Notes
    -----
    This module is retained for compatibility with the original code. The maintained
    srFRC API is src.srFRC.RollingFourierRingCorrelation; the legacy routine is not the
    stack/restoration entry point.
    """
    image1 = stack[0]
    image2 = stack[1]
    h, w = image1.shape
    if correct_drift:
        # detect drift of imB
        drift, _ = drift_detect(image1, image2, pixel_size)
        # correct drift
        fft_a = ft.fftshift(ft.fft2(image1))
        fft_b = ft.fftshift(ft.fft2(image2))
        x = np.linspace(-math.floor(w / 2), -math.floor(w / 2) + w, w)
        y = np.linspace(-math.floor(h / 2), -math.floor(h / 2) + h, h)
        [x_frequency, y_frequency] = np.meshgrid(x, y)
        fft_b = fft_b * np.exp(-1j * 2 * np.pi * ((-x_frequency * drift[0]) / h + (-y_frequency * drift[1]) / w)) + float_info.epsilon
        image2 = abs(np.real(ft.ifft2(ft.ifftshift(fft_b))))
        image2 /= np.max(image2)
        print('Subpixel shift: \n %.4f pixels in x;\n %.4f pixels in x;\n' % (drift[0], drift[1]))

    if block_size % 2 != 0:
        block_size += 1
    tem = (np.array(block_size / 2)).astype('int')
    skip = skip if skip < tem else tem

    [w, h] = image1.shape

    padw = w + block_size
    padh = h + block_size

    image1_pad = createPaddedImage(image1, padw, padh)
    image2_pad = createPaddedImage(image2, padw, padh)

    bi = (np.array(background_intensity * 2 / 255)).astype('float32')
    xstart_center = np.array((block_size / 2 - 1), dtype='int')
    rfrc_map = np.zeros((w, h), dtype='float32')
    size_padded = 128
    hm = hanning_2d(size_padded, size_padded)

    for xstart in range(0, w, skip):
        for ystart in range(0, h, skip):
            image1_roi = np.zeros((block_size, block_size), dtype='float32')
            image1_roi = image1_pad[xstart:xstart + block_size, ystart:ystart + block_size]
            image2_roi = np.zeros((block_size, block_size), dtype='float32')
            image2_roi = image2_pad[xstart:xstart + block_size, ystart:ystart + block_size]
            summing = np.mean(image1_roi[xstart_center:xstart_center + skip, xstart_center:xstart_center + skip])
            summing += np.mean(image2_roi[xstart_center:xstart_center + skip, xstart_center:xstart_center + skip])
            if summing > bi:
                flag = True
                image1_roi = image1_roi / np.max(image1_roi)
                image2_roi = image2_roi / np.max(image2_roi)
                pad_factor = math.sqrt(size_padded / block_size)
                image1_roi = createPaddedImage(image1_roi, size_padded, size_padded)
                image2_roi = createPaddedImage(image2_roi, size_padded, size_padded)
                ffta = ft.fftshift(ft.fft2(image1_roi * hm))
                fftb = ft.fftshift(ft.fft2(image2_roi * hm))
                frc_data = perform_frc(ffta, fftb, pixel_size, single_frame=single_frame)
                frc_data.calculate_resolution(additional_sigma=True, pad_factor=pad_factor)

                if not math.isnan(threesigma_resolution) and not math.isinf(
                        threesigma_resolution) and not threesigma_resolution < 0:
                    resolution = threesigma_resolution
                elif not math.isnan(fivesigma_resolution) and not math.isinf(
                        fivesigma_resolution) and not fivesigma_resolution < 0:
                    resolution = fivesigma_resolution / (np.array(25 / 9))
                elif not math.isnan(twosigma_resolution) and not math.isinf(
                        twosigma_resolution) and not twosigma_resolution < 0:
                    resolution = twosigma_resolution / (np.array(4 / 9))
                else:
                    flag = False
                    resolution = 0

                if resolution < pixel_size * 2.3:
                    resolution = pixel_size * 2.3
                if not flag:
                    resolution = 0
                for xxsum in range(skip):
                    for yysum in range(skip):
                        rfrc_map[xstart + xxsum, ystart + yysum] = resolution
    if amedianfilter:
        rfrc_map = AMF(rfrc_map)
    return rfrc_map


def createPaddedImage(image1, padded_width, padded_height):
    """Apply the legacy image normalization and symmetric padding operation.
    
    Parameters
    ----------
    image1 : numpy.ndarray
        First image or stack of images.
    paddedWidth : int
        Requested padded width in pixels.
    paddedHeight : int
        Requested padded height in pixels.
    
    Returns
    -------
    numpy.ndarray
        Padded floating-point image used by the historical fSNRmap routine.
    """
    image_width = image1.shape[0]
    image_height = image1.shape[0]
    extra_width = padded_width - image_width
    extra_height = padded_height - image_height
    image1 = image1.astype('float32') / 255
    image_tem = np.array(image1)
    extracol = np.array((extra_width / 2), dtype='int')
    extrarow = np.array((extra_height / 2), dtype='int')
    padded_array = np.pad(image_tem, (extracol, extrarow), 'symmetric')
    return padded_array
