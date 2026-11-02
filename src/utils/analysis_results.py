"""In-memory batch handling, quantitative map processing and result guidance."""
import warnings
from functools import wraps
import numpy as np
from scipy.interpolate import CubicSpline
from src.utils.sfsc_guidance import ANGLE_GUIDANCE, fit_range_advice

DRI_GUIDANCE = 'E_DRI (relative fit error): <10% reliable; 10-20% use with caution; >20% unacceptable.'

def with_sfsc_dri_guidance(analysis):
    """Print the complete E_DRI grading guide whenever an sFSC call finishes.

    Parameters
    ----------
    analysis : callable
        sFSC entry point to invoke without changing its arguments or result.

    Returns
    -------
    callable
        Wrapped entry point that preserves its signature and documentation.

    Notes
    -----
    The final notice also runs after early returns and raised errors, independently
    of plotting or warning filters. It does not suppress errors or assign a grade
    to an unavailable E_DRI.
    """
    @wraps(analysis)
    def wrapped(*args, **kwargs):
        """Forward the analysis call and append E_DRI guidance on exit.

        Parameters
        ----------
        *args : tuple
            Original positional analysis arguments.
        **kwargs : dict
            Original keyword analysis arguments.

        Returns
        -------
        object
            The unchanged analysis result; original errors propagate to the caller.
        """
        try:
            return analysis(*args, **kwargs)
        finally:
            print('sFSC - E_DRI guidance: ' + DRI_GUIDANCE, flush=True)
    return wrapped

def dri_interpretation(error):
    """Interpret the relative directional-resolution fit error.
    
    Parameters
    ----------
    error : float
        Relative E_DRI as a fraction, so 0.10 means 10 percent.
    
    Returns
    -------
    str
        reliable, use with caution, unacceptable, or unavailable.
    
    Notes
    -----
    E_DRI is a fraction: <0.10 is reliable; 0.10 through 0.20 requires caution; >0.20 is
    unacceptable.
    """
    if not np.isfinite(error): return 'unavailable'
    if error < 0.10: return 'reliable'
    if error > 0.20: return 'unacceptable'
    return 'use with caution'

def frame_pairs(image1, image2=None, single_frame=True, stack_axis=0,
                batch_processing=True):
    """Select 2D frames or paired frames from image arrays.
    
    Parameters
    ----------
    image1 : numpy.ndarray
        First image or stack of images.
    image2 : numpy.ndarray or None
        Second image or stack, when paired input is supplied separately. {default: None}
    single_frame : bool
        Use the correlation correction for samples derived from one acquisition.
        {default: True}
    stack_axis : int
        Axis containing independent 2D frames; 0 means (T,Y,X), and 2 means (Y,X,T).
        {default: 0}
    batch_processing : bool
        Process every available frame or adjacent pair; False selects the first
        only. {default: True}
    
    Returns
    -------
    list of tuple
        (first,second,label) entries in analysis order. second is None in single-frame
        mode. Labels are one-based frame or pair identifiers.
    
    Notes
    -----
    One N-frame stack produces N single-frame entries or N-1 adjacent-pair
    entries when batching is enabled. Use stack_axis=2 for MATLAB-style (Y,X,T).
    Paired input uses matching frames when image2 is supplied; otherwise adjacent
    overlapping pairs are formed. Single-frame mode ignores image2. Non-batch mode always uses the first available index.
    """
    first = np.asarray(image1)
    if first.ndim == 2: first = first[None]
    elif first.ndim == 3: first = np.moveaxis(first, stack_axis, 0)
    else: raise ValueError('Provide a 2D grayscale image (Y,X) or a 3D stack. For (Y,X,T), set stack_axis=2.')
    if first.shape[0] == 0: raise ValueError('The input stack is empty. Provide at least one image.')
    if not isinstance(batch_processing,(bool,np.bool_)):
        raise ValueError('batch_processing must be True or False.')
    def indices(count):
        """Return all valid indices or just the first, depending on batch mode."""
        return range(count) if batch_processing else [0]
    if single_frame:
        return [(first[i],None,f'frame{i+1:04d}') for i in indices(len(first))]
    if image2 is not None:
        second = np.asarray(image2)
        if second.ndim == 2: second = second[None]
        elif second.ndim == 3: second = np.moveaxis(second, stack_axis, 0)
        else: raise ValueError('The paired input must be 2D or 3D.')
        if first.shape != second.shape: raise ValueError('Provide paired images or stacks with matching shapes.')
        return [(first[i],second[i],f'pair{i+1:04d}') for i in indices(len(first))]
    if len(first)<2: raise ValueError('Double-frame mode needs at least two frames in the input stack for an adjacent pair.')
    return [(first[i],first[i+1],f'pair{i+1:04d}-{i+2:04d}') for i in indices(len(first)-1)]

def normalize_threshold(threshold):
    """Resolve the two supported public correlation criteria.

    Parameters
    ----------
    threshold : str
        '1/7 hard threshold' or '3-sigma'. Legacy fixed_threshold and 1/7
        spellings remain aliases of the same fixed criterion.

    Returns
    -------
    str
        Internal curve key: fixed_threshold or 3-sigma.
    """
    if isinstance(threshold, str):
        value = threshold.strip().lower()
        if value in ('1/7 hard threshold', '1/7', 'fixed_threshold'):
            return 'fixed_threshold'
        if value == '3-sigma':
            return '3-sigma'
    raise ValueError("Choose threshold='1/7 hard threshold' or '3-sigma'.")


def resolve_plot(requested, batch_processing):
    """Resolve the plotting policy for batch and individual runs.
    
    Parameters
    ----------
    requested : bool or str
        True previews the first frame/pair with or without batching.
        The legacy 'auto' alias is accepted for older callers.
        True/'true' requests plots; False/'false' hides them.
    batch_processing : bool
        Process every available frame or adjacent pair; False processes only the first frame/pair.
    
    Returns
    -------
    bool
        True when plotting was requested. Batch selection is independent of
        plotting; current_result_index chooses the single result to display.
    """
    if isinstance(requested, str):
        value = requested.strip().lower()
        if value not in ('auto', 'true', 'false'):
            raise ValueError("Choose plot=True or False; legacy 'auto' is also accepted.")
        requested = value != 'false'
    elif not isinstance(requested, (bool, np.bool_)):
        raise ValueError("Choose plot=True or False; legacy 'auto' is also accepted.")
    return bool(requested)


def current_result_index(count):
    """Return index zero for the first computed frame/pair, or None if empty.

    Batch size and frame metadata never control preview selection. This keeps
    the first-result behavior identical in batch and non-batch modes.
    """
    if count <= 0:
        return None
    return 0

def finalize_resolution_map(map_data, original_pixel_size_nm=None):
    """Apply background cleanup and the acquisition-based map lower bound.
    
    Parameters
    ----------
    map_data : numpy.ndarray
        Resolution-map values in nm.
    original_pixel_size_nm : float or None
        Original acquisition pixel size in nm. If supplied, finite positive map values
        are floored at 2.3 times this spacing. {default: None}
    
    Returns
    -------
    numpy.ndarray
        Copied float32 map in nm. Non-positive values become zero; finite positive
        values are floored when original_pixel_size_nm is supplied.
    
    Notes
    -----
    The float32 floor is rounded upward when necessary so it never falls below 2.3 times
    the original pixel size. Non-finite values are left unchanged and excluded by
    map_statistics.
    """
    data = np.array(map_data, dtype=np.float32, copy=True)
    data[data <= 0] = 0
    if original_pixel_size_nm is not None:
        spacing = float(original_pixel_size_nm)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError('Enter a positive, finite original pixel size in nm.')
        minimum = 2.3 * spacing
        floor = np.float32(minimum)
        if not np.isfinite(floor):
            raise ValueError('The original pixel size is too large for a float32 resolution map.')
        if float(floor) < minimum:
            floor = np.nextafter(floor, np.float32(np.inf))
        foreground = np.isfinite(data) & (data > 0)
        data[foreground] = np.maximum(data[foreground], floor)
    return data


def restore_map_size(map_data, target_shape, original_pixel_size_nm=None):
    """Restore a quantitative map with a separable interpolating cubic spline.
    
    Parameters
    ----------
    map_data : numpy.ndarray
        Resolution-map values in nm.
    target_shape : tuple of int
        Positive target image dimensions (Y,X).
    original_pixel_size_nm : float or None
        Original acquisition pixel size in nm. If supplied, finite positive map values
        are floored at 2.3 times this spacing. {default: None}
    
    Returns
    -------
    numpy.ndarray
        Float32 resolution map in nm with shape target_shape.
    
    Notes
    -----
    The spline uses not-a-knot boundary conditions, first along X, then Y.
    Output index j samples source coordinate j/2, clamped to the final source
    sample. Singleton axes are constant; two/three samples give linear/quadratic
    interpolation. Integer-repeated source validity defines foreground support.
    Non-positive candidates remain zero and positive candidates are bounded by
    the source foreground range. The final floor uses the original pixel spacing.
    Interpolated foreground means need not equal the source mean.
    """
    source = np.asarray(map_data, dtype=float)
    target = tuple(target_shape)
    if source.ndim != 2 or len(target) != 2 or min(source.shape) < 1:
        raise ValueError('Provide a nonempty 2D resolution map and a 2D target shape.')
    if any(not isinstance(size, (int, np.integer)) or size < 1 for size in target):
        raise ValueError('Target dimensions must be positive integers.')
    if source.shape != target and source.shape != tuple(size // 2 for size in target):
        raise ValueError('Restoration expects a half-size map or an unchanged paired map.')
    valid = np.isfinite(source) & (source > 0)
    clean = np.where(valid, source, 0)
    if source.shape == target:
        data = clean.copy()
    else:
        y = np.minimum(np.arange(target[0]) / 2.0, source.shape[0] - 1)
        x = np.minimum(np.arange(target[1]) / 2.0, source.shape[1] - 1)
        support = valid[np.floor(y).astype(int)[:, None],
                        np.floor(x).astype(int)[None, :]]
        if np.any(valid):
            if source.shape[1] > 1:
                data = CubicSpline(np.arange(source.shape[1]), clean, axis=1,
                                   bc_type='not-a-knot', extrapolate=False)(x)
            else:
                data = np.repeat(clean, target[1], axis=1)
            if source.shape[0] > 1:
                data = CubicSpline(np.arange(source.shape[0]), data, axis=0,
                                   bc_type='not-a-knot', extrapolate=False)(y)
            else:
                data = np.repeat(data, target[0], axis=0)
            positive = support & np.isfinite(data) & (data > 0)
            data = np.where(positive, np.clip(data, clean[valid].min(),
                                            clean[valid].max()), 0)
        else:
            data = np.zeros(target, dtype=float)
    return finalize_resolution_map(data, original_pixel_size_nm)

def map_statistics(data):
    """Measure the finite positive foreground of a resolution map.
    
    Parameters
    ----------
    data : numpy.ndarray
        Numerical samples to process.
    
    Returns
    -------
    dict
        mean_resolution_nm, min_resolution_nm, max_resolution_nm, and rFRC_value. Empty
        foreground returns NaN for all fields.
    
    Notes
    -----
    rFRC_value = mean / minimum - 1; background zero and non-finite values are excluded.
    """
    valid = np.asarray(data)[np.isfinite(data) & (np.asarray(data)>0)]
    if valid.size == 0: return dict(mean_resolution_nm=np.nan,min_resolution_nm=np.nan,max_resolution_nm=np.nan,rFRC_value=np.nan)
    mean, minimum, maximum = float(np.mean(valid,dtype=np.float64)),float(np.min(valid)),float(np.max(valid))
    return dict(mean_resolution_nm=mean,min_resolution_nm=minimum,max_resolution_nm=maximum,rFRC_value=mean/minimum-1)

def extrapolation_notice(curve, threshold, context):
    """Report successful extrapolation or an unavailable resolution.
    
    Parameters
    ----------
    curve : FourierCorrelationData
        Correlation result whose crossing and extrapolation state are inspected.
    threshold : str
        Correlation threshold: fixed_threshold represents 1/7; 3-sigma is also
        supported.
    context : str
        Analysis-stage label used in diagnostics.
    
    Returns
    -------
    None
        Prints a stage-specific notice or emits a RuntimeWarning.
    """
    if curve.extrapolated():
        print(f'{context}: No threshold crossing was found in the measured range. Automatic 1/7 extrapolation provided the resolution estimate.')
    elif not np.isfinite(curve.get_resolution(threshold)):
        if curve.needs_extrapolation() and threshold != 'fixed_threshold':
            message = f'{context}: No threshold crossing was found in the measured range. Automatic extrapolation is available only for 1/7. Resolution is unavailable (NaN); check the threshold and pixel size.'
        else:
            message = f'{context}: Automatic extrapolation could not provide a usable threshold crossing. Resolution is unavailable (NaN); check the pixel size and threshold.'
        warnings.warn(message, RuntimeWarning, stacklevel=2)
