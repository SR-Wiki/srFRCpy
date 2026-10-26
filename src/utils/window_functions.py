import numpy as np


def hanning_2d(height, width):
    """Construct a separable Hann window.
    
    Parameters
    ----------
    height : int
        Output height, in pixels.
    width : int
        Output width, in pixels.
    
    Returns
    -------
    numpy.ndarray
        Window with dimensions (Y,X) specified by the arguments.
    """
    han_y = np.hanning(height)
    han_x = np.hanning(width)
    return np.outer(han_y, han_x)


def hanning_3d(depth, height, width):
    """Construct a separable Hann window.
    
    Parameters
    ----------
    depth : int
        Number of Z slices.
    height : int
        Output height, in pixels.
    width : int
        Output width, in pixels.
    
    Returns
    -------
    numpy.ndarray
        Window with dimensions (Z,Y,X) specified by the arguments.
    """
    han_z = np.hanning(depth)
    han_y = np.hanning(height)
    han_x = np.hanning(width)
    return np.multiply.outer(np.multiply.outer(han_z, han_y), han_x)


if __name__ == '__main__':
    hn = hanning_3d(32, 512, 128)
    print(hn.shape)

