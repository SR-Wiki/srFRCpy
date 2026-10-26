# -*- coding: utf-8 -*-
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy.fft as ft
from sys import float_info


def drift_correct(im1, im2, pixel_size, verbose=False):
    """Align two square images using a fitted cross-correlation shift.
    
    Parameters
    ----------
    im1 : numpy.ndarray
        First image or volume; its shape must match the paired input.
    im2 : numpy.ndarray or None
        Second image or volume for paired-acquisition analysis.
    pixel_size : float
        Positive calibrated pixel spacing in nm.
    verbose : bool
        Print diagnostics or display diagnostic figures. {default: False}
    
    Returns
    -------
    tuple of numpy.ndarray
        First image and aligned second image, each normalized to a maximum of 255.
    """
    h, w = im1.shape
    drift, _ = drift_detect(im1, im2, pixel_size)
    # correct drift
    fftb = ft.fftshift(ft.fft2(im2))
    x = np.linspace(-math.floor(w / 2), -math.floor(w / 2) + w, w)
    y = np.linspace(-math.floor(h / 2), -math.floor(h / 2) + h, h)
    [x_frequency, y_frequency] = np.meshgrid(x, y)
    fftb = fftb * np.exp(-1j * 2 * np.pi * ((-x_frequency * drift[0]) / h + (-y_frequency * drift[1]) / w)) + float_info.epsilon
    im2 = abs(np.real(ft.ifft2(ft.ifftshift(fftb))))
    im1 = 255 * im1 / np.max(im1)
    im2 = 255 * im2 / np.max(im2)
    if verbose:
        print('Subpixel shift: \n %.4f pixels in x;\n %.4f pixels in x;\n' % (drift[0], drift[1]))
    return im1, im2
    

def drift_detect(im1, im2, pixel_size, p0=(-1, -1, -1, -1, -1, -1, -1), verbose=False, fit_window=10):
    """Estimate subpixel image displacement from a Gaussian cross-correlation fit.
    
    Parameters
    ----------
    im1 : numpy.ndarray
        First image or volume; its shape must match the paired input.
    im2 : numpy.ndarray or None
        Second image or volume for paired-acquisition analysis.
    pixel_size : float
        Positive calibrated pixel spacing in nm.
    p0 : tuple of float
        Initial Gaussian parameters: amplitude, x center, y center, x width, y width,
        angle, offset. A first value of -1 selects automatic initialization.
        {default: (-1, -1, -1, -1, -1, -1, -1)}
    verbose : bool
        Print diagnostics or display diagnostic figures. {default: False}
    fit_window : int
        Half-width of the cross-correlation region used for Gaussian fitting, in pixels.
        {default: 10}
    
    Returns
    -------
    tuple of numpy.ndarray
        Two-component displacement and its fitted uncertainty, both in pixels.
    
    Notes
    -----
    Inputs must be square and equally sized. pixel_size is used only for the
    physical-unit diagnostic.
    """
    assert (im1.shape == im2.shape and im1.shape[0] == im1.shape[1]), "im1ges non-square or non-equal sized"
    imcenter = int(im1.shape[0] / 2)

    # perform autocorrelation
    corr = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(im1) * np.conj(np.fft.fft2(im2))))
    # take center region for Gaussian fitting
    corr = corr[imcenter - fit_window:imcenter + fit_window, imcenter - fit_window:imcenter + fit_window]

    # Create x and y indices
    x = np.arange(corr.shape[0])
    y = np.arange(corr.shape[0])
    coord = np.meshgrid(x, y)

    # if parameter guess is unset, make intelligent estim1te.
    if p0[0] == -1:
        p0 = (np.max(abs(corr)), fit_window, fit_window, 1, 1, 0, 0)

    # perform 2D gauss fit
    popt, pcov = curve_fit(gaussian_2d, coord, abs(corr).ravel(), p0=p0, full_output=False)

    # take relevant parameters
    drift = popt[1:3] - [fit_window, fit_window]
    drift_sigma = np.sqrt([pcov[1, 1], pcov[2, 2]])

    # plt and print
    if verbose:
        plt.imshow(abs(corr))
        plt.plot([0, fit_window * 2], [fit_window, fit_window], 'r', linewidth=0.5)
        plt.plot([fit_window, fit_window], [0, fit_window * 2], 'r', linewidth=0.5)
        plt.colorbar()
        plt.scatter(popt[1], popt[2], s=10)
        plt.show()

        print("using initial parameters (Amplitude, xpos, ypos, xsigma, ysigma, theta, background) equal to " + str(p0))

        print(u"shift in x is  {:.3f} \u00B1 {:.3f}  pixels\n".format(drift[0], drift_sigma[0])
              + u"shift in y is  {:.3f} \u00B1 {:.3f} pixels\n".format(drift[1], drift_sigma[1])
              + u"total shift is {:.3f} \u00B1 {:.3f} nm for pixel size of {:d} nm \n".format(
            np.linalg.norm(drift - [fit_window, fit_window]) * pixel_size,
            np.linalg.norm(drift_sigma * pixel_size),
            pixel_size))
    return drift, drift_sigma


def gaussian_2d(coord, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """Evaluate a rotated 2D Gaussian with a constant background.
    
    Parameters
    ----------
    coord : sequence of numpy.ndarray
        X and Y coordinate grids for the Gaussian model.
    amplitude : float
        Gaussian amplitude.
    xo : float
        Gaussian center along X, in pixels.
    yo : float
        Gaussian center along Y, in pixels.
    sigma_x : float
        Gaussian width along its first principal axis, in pixels.
    sigma_y : float
        Gaussian width along its second principal axis, in pixels.
    theta : float
        Rotation of the Gaussian principal axes in radians.
    offset : float
        Additive Gaussian background.
    
    Returns
    -------
    numpy.ndarray
        Flattened Gaussian model matching the coordinate grids.
    """
    x = coord[0]
    y = coord[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                       + c * ((y - yo) ** 2)))
    return g.ravel()
