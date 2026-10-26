import math
import numpy as np
from src.utils.window_functions import hanning_3d


def diagonal_resample(image):
    """Generate a diagonal single-frame pair by averaging complementary pixels.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image data to process.
    
    Returns
    -------
    tuple of numpy.ndarray
        Two float arrays at half the even-cropped XY dimensions; leading dimensions are
        preserved.
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape[-2:]
    if h < 2 or w < 2: raise ValueError('Single-frame input must be at least 2 x 2 pixels.')
    image = image[..., :h-h%2, :w-w%2]
    return ((image[...,0::2,0::2]+image[...,1::2,1::2])/2,
            (image[...,0::2,1::2]+image[...,1::2,0::2])/2)


def pad2even(im, pad_value=1):
    """Pad a volume to even dimensions.
    
    Parameters
    ----------
    im : numpy.ndarray
        Input image or volume.
    pad_value : int
        0 selects zero padding; other values select symmetric padding. {default: 1}
    
    Returns
    -------
    numpy.ndarray
        Volume with one trailing sample added along each odd-length axis, using zero or
        symmetric padding.
    """
    d, h, w = im.shape
    complement = [s % 2 for s in (d, h, w)]
    if pad_value == 0:
        im_padded = np.pad(im, ((0, complement[0]), (0, complement[1]), (0, complement[2])), mode='constant')
    else:
        im_padded = np.pad(im, ((0, complement[0]), (0, complement[1]), (0, complement[2])), mode='symmetric')
    return im_padded


def pad2cube(im, pad_value=1):
    """Prepare an even cubic volume with padding, windowing, and normalization.
    
    Parameters
    ----------
    im : numpy.ndarray
        Input image or volume.
    pad_value : int
        0 selects zero padding; other values select symmetric padding. {default: 1}
    
    Returns
    -------
    numpy.ndarray
        Cubic volume scaled to a maximum of 255 when the maximum is positive.
    """
    im_padded = pad2even(im, pad_value)
    h_new, w_new, d_new = im_padded.shape
    max_dim = max(h_new, w_new, d_new)
    pad_shape = [(max_dim - h_new) // 2, (max_dim - w_new) // 2, (max_dim - d_new) // 2]
    if pad_value == 0:
        hn = hanning_3d(h_new, w_new, d_new)
        windowed_im = im_padded * hn
    else:
        windowed_im = im_padded
    padded = np.pad(
        windowed_im,
        (
            (pad_shape[0], max_dim - h_new - pad_shape[0]),
            (pad_shape[1], max_dim - w_new - pad_shape[1]),
            (pad_shape[2], max_dim - d_new - pad_shape[2])
        ),
        mode='constant' if pad_value == 0 else 'symmetric'
    )
    if pad_value != 0:
        hn = hanning_3d(*padded.shape)
        padded = padded * hn

    max_val = np.max(padded)
    if max_val > 0:
        padded = 255 * padded / max_val
    return padded


def sampling_xy(im, sampling_factor):
    """Resample XY in a (Z, Y, X) volume using the original index rule.

    The selected source coordinates are ``int(j * sampling_factor)`` and
    ``int(k * sampling_factor)``, exactly as in the former per-voxel loops.
    NumPy broadcasting performs the selection in compiled code instead.

    For backwards compatibility, a factor within 1e-4 of 1 returns the
    original array, while other factors return a fresh float64 array (as
    ``np.zeros`` did in the original implementation).

    Parameters
    ----------
    im : numpy.ndarray
        Input 3D volume in (Z, Y, X) order.
    sampling_factor : float
        XY sampling stride. Values above 1 reduce XY size.

    Returns
    -------
    numpy.ndarray
        Resampled volume with unchanged Z depth.
    """
    if abs(sampling_factor - 1) < 1e-4:
        return im

    d, h, w = im.shape
    new_h, new_w = int(h / sampling_factor), int(w / sampling_factor)

    # Preserve the truncation-based nearest-index positions of the old loops.
    iy = (np.arange(new_h) * sampling_factor).astype(np.intp)
    ix = (np.arange(new_w) * sampling_factor).astype(np.intp)

    # Both XY index arrays broadcast into one advanced-indexing operation.
    # The conversion to float64 retains the old np.zeros() output dtype.
    return np.asarray(im[:, iy[:, None], ix[None, :]], dtype=np.float64)


def sampling_z(im, sampling_factor):
    """Resample the Z dimension by nearest-index selection.
    
    Parameters
    ----------
    im : numpy.ndarray
        Input image or volume.
    sampling_factor : float
        Sampling stride; values above 1 reduce the corresponding dimensions.
    
    Returns
    -------
    numpy.ndarray
        Volume with unchanged XY dimensions and Z depth reduced by sampling_factor.
    """
    if abs(sampling_factor - 1) < 1e-4:
        return im
    d, h, w = im.shape
    new_d = int(d / sampling_factor)
    resized = np.zeros((new_d, h, w))
    for i in range(new_d):
        resized[i, :, :] = im[int(i * sampling_factor), :, :]
    return resized


def create_padded_image(image, padded_height, padded_width):
    """Symmetrically pad a 2D image to the requested dimensions.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image data to process.
    padded_height : int
        Requested padded image height in pixels.
    padded_width : int
        Requested padded image width in pixels.
    
    Returns
    -------
    numpy.ndarray
        Image with shape (padded_height,padded_width), preserving the original
        intensities.
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    extra_height = padded_height - image_height
    extra_width = padded_width - image_width
    extra_row = np.array((extra_height / 2), dtype='int')
    extra_col = np.array((extra_width / 2), dtype='int')
    padded_array = np.pad(image, ((extra_height//2, extra_height-extra_height//2),
                                  (extra_width//2, extra_width-extra_width//2)), 'symmetric')
    return padded_array


def select_planes(im, num_slices=20):
    """Select the brightest Z slices by mean image intensity.
    
    Parameters
    ----------
    im : numpy.ndarray
        Input image or volume.
    num_slices : int
        Maximum number of brightest Z slices to select. {default: 20}
    
    Returns
    -------
    list of int
        Zero-based slice indices sorted by decreasing mean intensity, limited to
        num_slices.
    """
    assert len(im.shape) == 3, "Input must be a 3D image!"
    depth = im.shape[0]
    stats = []
    for i in range(depth):
        stats.append((i, np.mean(im[i, :, :])))
    stats.sort(key=lambda x: x[1], reverse=True)
    select_indices = [idx for idx, _ in stats[:num_slices]]
    return select_indices



