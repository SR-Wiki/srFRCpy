import numpy as np


def mean_filter(data, width):
    """Smooth one-dimensional samples with a local moving mean.
    
    Parameters
    ----------
    data : numpy.ndarray
        Numerical samples to process.
    width : int
        Moving-mean neighborhood radius in samples.
    
    Returns
    -------
    numpy.ndarray
        Smoothed samples with truncated neighborhoods near the ends.
    """
    d = data.shape[0]
    smooth_data = np.zeros(d)
    for i in range(d):
        if i <= width:
            smooth_data[i] = np.mean(data[:i + width + 1])
        elif width < i < d - width:
            smooth_data[i] = np.mean(data[i - width: i + width + 1])
        elif i >= d - width:
            smooth_data[i] = np.mean(data[i - width:])
        else:
            print("error found in meanFilter program")
    return smooth_data


def amf(rfrc_array):
    """Suppress high local resolution-map outliers with a 3x3 median comparison.
    
    Parameters
    ----------
    rfrc_array : numpy.ndarray
        Two-dimensional resolution map to filter.
    
    Returns
    -------
    numpy.ndarray
        Copied float32 map; a center value above twice the local median is replaced by
        that median.
    """
    [width, height] = rfrc_array.shape
    window_size = 3
    threshold = float(2)
    rfrc_array_copy = np.array(rfrc_array, dtype='float32')
    new_array = np.array(rfrc_array_copy, dtype='float32')
    temp = np.zeros((window_size * window_size), np.float32)
    win = int((3 - 1) / 2)
    for x in range(win, width - win):
        for y in range(win, height - win):
            flage = 0
            for i in range(-win, win + 1, 1):
                for j in range(-win, win + 1, 1):
                    temp[flage] = rfrc_array_copy[x + i, y + j]
                    flage = flage + 1
            temp.sort()
            if threshold * temp[int(flage / 2)] < rfrc_array_copy[x, y]:
                new_array[x, y] = temp[int(flage / 2)]
    return new_array


def remove_outlier(abnormal):
    """Replace non-finite samples with one.
    
    Parameters
    ----------
    abnormal : numpy.ndarray
        Array that may contain NaN or infinite values.
    
    Returns
    -------
    numpy.ndarray
        Copied array with NaN and infinite entries set to 1.
    """
    normal = abnormal.copy()
    mask = np.isnan(normal) | np.isinf(normal)
    normal[mask] = 1
    return normal


def refine(pre):
    """Connect the correlation origin to the first maximum with a linear ramp.
    
    Parameters
    ----------
    pre : numpy.ndarray
        One-dimensional correlation samples before refinement.
    
    Returns
    -------
    numpy.ndarray
        Refined copy of the input samples.
    """
    refined = pre.copy()
    max_val = np.max(pre)
    idx = np.argmax(pre == max_val)
    if idx > 0:
        k = (1 - max_val) / idx
        x = np.arange(idx)
        refined[:idx] = 1 - k * x

    return refined
