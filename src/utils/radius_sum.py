import numpy as np


def radial_sum(data, radial_step=1):
    """Sum complex 2D Fourier values in rounded radial bins.
    
    Parameters
    ----------
    data : numpy.ndarray
        Numerical samples to process.
    radial_step : float
        Width of each radial bin in Fourier pixels. {default: 1}
    
    Returns
    -------
    numpy.ndarray
        One-dimensional complex sums ordered by radial bin.
    """
    h, w = data.shape
    x = np.arange(1, w + 1) - w / 2
    y = np.arange(1, h + 1) - h / 2
    x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
    radii = np.sqrt(x_grid ** 2 + y_grid ** 2)
    radial_bins = np.round(radii / radial_step).astype(int) + 1
    indices = radial_bins.ravel()
    values = data.ravel()
    max_idx = np.max(indices)
    real = np.bincount(indices, weights=np.real(values), minlength=max_idx+1)
    imag = np.bincount(indices, weights=np.imag(values), minlength=max_idx+1)
    result = real + 1j * imag
    # result = np.bincount(indices, weights=values, minlength=max_idx+1)
    return result[1:max_idx + 1]


def spherical_sum(data, radial_step=1):
    """Sum complex 3D Fourier values in rounded spherical shells.
    
    Parameters
    ----------
    data : numpy.ndarray
        Numerical samples to process.
    radial_step : float
        Width of each radial bin in Fourier pixels. {default: 1}
    
    Returns
    -------
    numpy.ndarray
        One-dimensional complex sums ordered by shell radius.
    """
    d, h, w = data.shape
    x = np.arange(1, w + 1) - w / 2
    y = np.arange(1, h + 1) - h / 2
    z = np.arange(1, d + 1) - d / 2
    z_grid, y_grid, x_grid = np.meshgrid(z, y, x, indexing='ij')
    radii = np.sqrt(x_grid ** 2 + y_grid ** 2 + z_grid ** 2)
    radial_bins = np.round(radii / radial_step).astype(int) + 1
    indices = radial_bins.ravel()
    values = data.ravel()
    max_idx = np.max(indices)
    real = np.bincount(indices, weights=np.real(values), minlength=max_idx + 1)
    imag = np.bincount(indices, weights=np.imag(values), minlength=max_idx + 1)
    result = real + 1j * imag
    # result = np.bincount(indices, weights=values, minlength=max_idx + 1)
    return result[1:max_idx + 1]


if __name__ == '__main__':
    data = np.ones((128, 128, 128))
    result = spherical_sum(data, 1)
    print(result.astype('uint16'))
