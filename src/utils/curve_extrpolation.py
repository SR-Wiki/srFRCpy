import numpy as np
from scipy.optimize import curve_fit,OptimizeWarning
import warnings
import matplotlib.pyplot as plt

def fit_func(x, a, k):
    """Evaluate the logarithmic correlation-decay model.
    
    Parameters
    ----------
    x : array_like
        Coordinates at which the fitted model is evaluated.
    a : float
        Positive amplitude parameter of the correlation-decay model.
    k : float
        Nonnegative decay parameter of the correlation model.
    
    Returns
    -------
    numpy.ndarray
        Logarithm of the sigmoid decay at x.
    """
    return np.log(a*np.exp(-k*x*x)/(a*np.exp(-k*x*x)+1))

def frc_extrapolation(single_frc, new_length, frequency, return_info=False):
    """Fit and extend a correlation curve while retaining its measured samples.
    
    Parameters
    ----------
    single_frc : numpy.ndarray
        One-dimensional measured correlation curve.
    new_length : int
        Requested curve length, at least as large as the measured curve.
    frequency : numpy.ndarray
        Measured positive-range spatial-frequency coordinates in 1/um; must match
        single_frc.
    return_info : bool
        Include fit parameters, R-squared, RMSE, and fallback status in the return
        value. {default: False}
    
    Returns
    -------
    tuple
        (new_frequency,extrapolated_frc), plus a fit-info dictionary when
        return_info=True.
    
    Notes
    -----
    Fit coordinates are normalized by the last measured frequency. Nonlinear fitting
    uses a linearized fallback if needed. Metadata contains A, k, r_squared, rmse, and
    used_fallback. Trigger decisions are made by the caller.
    """
    single_frc = np.asarray(single_frc,dtype=float)
    frequency = np.asarray(frequency,dtype=float)
    if len(frequency)<4 or len(frequency)!=len(single_frc) or new_length<len(frequency):
        raise ValueError('Invalid curve/frequency lengths.')
    fmax = float(frequency[-1])
    if fmax<=0: raise ValueError('The frequency range must be positive.')
    x = frequency/fmax
    y = np.clip(single_frc,1e-6,1-1e-6)
    slope,intercept = np.polyfit(x*x,np.log(y/(1-y)),1)
    initial=(np.exp(np.clip(intercept,-30,30)),max(-slope,1e-6))
    used_fallback=False
    def model(x,a,k):
        """Evaluate the positive sigmoid correlation-decay model.
        
        Parameters
        ----------
        x : array_like
            Coordinates at which the fitted model is evaluated.
        a : float
            Positive amplitude parameter of the correlation-decay model.
        k : float
            Nonnegative decay parameter of the correlation model.
        
        Returns
        -------
        numpy.ndarray
            A*exp(-k*x*x)/(1+A*exp(-k*x*x)) at the requested coordinates.
        """
        q=a*np.exp(-k*x*x)
        return q/(1+q)
    try:
        with warnings.catch_warnings():
            # Only fitted curve parameters are used; covariance diagnostics
            # do not determine whether a crossing is available.
            warnings.simplefilter('ignore',OptimizeWarning)
            popt,_=curve_fit(model,x,y,p0=initial,bounds=([1e-12,0],[np.inf,np.inf]),maxfev=5000)
    except (RuntimeError,ValueError,FloatingPointError):
        popt=initial
        used_fallback=True
    a_fit,k_fit=popt
    pre_length=len(frequency)
    step=float(np.median(np.diff(frequency)))
    new_frequency=np.concatenate((frequency,frequency[-1]+step*np.arange(1,new_length-pre_length+1)))
    extrapolated_frc=model(new_frequency/fmax,a_fit,k_fit)
    extrapolated_frc[:pre_length]=single_frc

    # Fit diagnostics are calculated only for reporting. They do not alter the
    # fitted parameters, extrapolated curve, or threshold-crossing result.
    fitted_measured = model(x,a_fit,k_fit)
    finite = np.isfinite(single_frc) & np.isfinite(fitted_measured)
    if np.any(finite):
        residual = single_frc[finite] - fitted_measured[finite]
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        ss_res = float(np.sum(residual ** 2))
        centered = single_frc[finite] - np.mean(single_frc[finite])
        ss_tot = float(np.sum(centered ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        rmse = np.nan
        r_squared = np.nan

    info = {
        'A': float(a_fit),
        'k': float(k_fit),
        'r_squared': r_squared,
        'rmse': rmse,
        'used_fallback': used_fallback,
    }

    # Keep the historical two-return-value API unless diagnostics are explicitly
    # requested by the correlation container.
    if return_info:
        return new_frequency, extrapolated_frc, info
    return new_frequency, extrapolated_frc
