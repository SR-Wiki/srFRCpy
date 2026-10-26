"""Floating-point PANEL display values derived from the final nm map."""
import numpy as np
from skimage import filters


def otsu(rfrc_map, enable_otsu=True):
    """Convert resolution maps to normalized image-quality display panels.
    
    Parameters
    ----------
    rfrc_map : numpy.ndarray
        Resolution map in nm, with shape (Y,X) or (T,Y,X).
    enable_otsu : bool
        Apply the Otsu-based display mask to the normalized quality panel.
        {default: True}
    
    Returns
    -------
    numpy.ndarray
        Scores in [0,1] with the input image/stack shape; background and non-finite
        samples are zero.
    """
    data=np.asarray(rfrc_map,dtype=float)
    if data.ndim not in (2,3):raise ValueError('Expected a 2D map or (T,Y,X) stack.')
    frames=data[None] if data.ndim==2 else data
    results=[]
    for frame in frames:
        valid=np.isfinite(frame) & (frame>0)
        display=np.zeros(frame.shape,dtype=np.float32)
        if np.any(valid):
            display[valid]=np.clip(frame[valid]/np.min(frame[valid])-1,0,1)
            if enable_otsu:
                display=np.maximum(display-.4,0)
                level=filters.threshold_otsu(display) if np.ptp(display)>0 else 0
                display[display<level]=0
        results.append(display)
    results=np.stack(results)
    return results[0] if data.ndim==2 else results
