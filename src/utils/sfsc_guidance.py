"""sFSC parameter guidance and manual-only Z-sampling settings."""
import numpy as np
import warnings

ANGLE_GUIDANCE = 'Recommended DRI fitting angles: lowAngle > 10 degrees and highAngle < 80 degrees.'
FIT_RANGE_ADVICE = 'Consider adjusting the DRI fitting angle range, then repeat the analysis. Recommended DRI fitting angles: lowAngle > 10 degrees and highAngle < 80 degrees.'
Z_SAMPLING_NOTE = 'Z sampling is OFF and can be enabled manually. If the axial resolution appears overly optimistic (too small), enable Z sampling and repeat the analysis.'

def resolve_z_sampling(value):
    """Resolve the manual Z-sampling option.
    
    Parameters
    ----------
    value : bool or None
        Manual Z-sampling request; legacy None means False.
    
    Returns
    -------
    bool
        The requested boolean value; legacy None is treated as False.
    
    Notes
    -----
    Z sampling is disabled by default and is never automatically selected from the voxel
    spacings.
    """
    if value is None:
        return False
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('z_sampling must be True or False (default). Legacy None means False.')
    return bool(value)

def angle_range_notice(low, high):
    """Warn when the DRI fitting range extends beyond the recommended angles.
    
    Parameters
    ----------
    low : float
        Lower DRI fitting angle in degrees.
    high : float
        Upper DRI fitting angle in degrees.
    
    Returns
    -------
    None
        Emits a warning for low <=10 degrees or high >=80 degrees.
    """
    if low <= 10 or high >= 80:
        warnings.warn(f'sFSC - fitting angles: lowAngle={low:g}, highAngle={high:g} degrees. {ANGLE_GUIDANCE}',
                      UserWarning, stacklevel=3)

def fit_range_advice(error):
    """Choose fitting-angle advice from the relative E_DRI.
    
    Parameters
    ----------
    error : float
        Relative E_DRI as a fraction, so 0.10 means 10 percent.
    
    Returns
    -------
    str
        Angle-range advice for finite E_DRI >=0.10, otherwise an empty string.
    """
    return FIT_RANGE_ADVICE if np.isfinite(error) and error >= 0.10 else ''

def z_sampling_note(single_frame, enabled, axial_resolution):
    """Provide an axial-resolution interpretation note when Z sampling is off.
    
    Parameters
    ----------
    single_frame : bool
        Use the correlation correction for samples derived from one acquisition.
    enabled : bool
        Whether Z sampling was enabled by the user.
    axial_resolution : float
        Axial resolution estimate in nm; NaN denotes an unavailable estimate.
    
    Returns
    -------
    str
        Manual Z-sampling advice for a finite single-frame axial estimate, otherwise an
        empty string.
    """
    return Z_SAMPLING_NOTE if single_frame and not enabled and np.isfinite(axial_resolution) else ''

def validate_num_cones(value):
    """Validate the positive directional-cone count.
    
    Parameters
    ----------
    value : int
        Positive integer directional-cone count; 5 or more is recommended.
    
    Returns
    -------
    int
        The positive integer count.
    
    Notes
    -----
    Counts below 5 remain valid; 5 or more is a recommendation rather than an enforced
    minimum.
    """
    if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or value <= 0:
        raise ValueError('num_cones must be a positive integer.')
    return int(value)
