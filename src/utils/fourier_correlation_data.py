import numpy as np
from src.utils.curve_extrpolation import frc_extrapolation


def find_intercept(data, threshold, pixel_size):
    """Locate a usable downward threshold crossing and convert it to resolution.
    
    Parameters
    ----------
    data : numpy.ndarray
        Numerical samples to process.
    threshold : numpy.ndarray
        One-dimensional threshold values matching data.
    pixel_size : float
        Positive calibrated pixel spacing in nm.
    
    Returns
    -------
    tuple
        (resolution_nm,crossing_index); NaN and -1 indicate no usable crossing.
    """
    if data.shape != threshold.shape:
        raise ValueError('data and threshold must have equal length')
    d = data.shape[0]
    intercept = -1
    resolution = np.nan
    # Both FRC and FSC frequency arrays start at zero (np.arange(max_r)).
    # Return the zero-based crossing sample unchanged; do not add 1.
    for i in range(3, d - 1):
        if data[i] < threshold[i] < max(data[i - 1], data[i - 2], data[i - 3]):
            resolution = 2 * d * pixel_size / i
            intercept = i
            break
    return resolution, intercept


class FourierCorrelationData:
    """Container for FRC/FSC curves and resolution estimates.

    ``extrapolation`` is permission to extrapolate, not evidence that
    extrapolation was used.  Actual use is reported by ``extrapolated()``.
    """

    def __init__(self, frequency, smoothed_frc, smoothed_frc_corrected, fixed_threshold,
                 pixel_size, num_pixels, single_frame=False, extrapolation=False):
        """Store measured and corrected correlation curves and threshold state.
        
        Parameters
        ----------
        frequency : numpy.ndarray
            One-dimensional spatial-frequency coordinates in 1/um.
        smoothed_frc : numpy.ndarray
            Measured, smoothed correlation values.
        smoothed_frc_corrected : numpy.ndarray
            Correlation values after the single-frame correction.
        fixed_threshold : numpy.ndarray
            Fixed 1/7 threshold samples matching the correlation curve.
        pixel_size : float
            Positive calibrated pixel spacing in nm.
        num_pixels : numpy.ndarray
            Number of Fourier samples in each ring or shell.
        single_frame : bool
            Use the correlation correction for samples derived from one acquisition.
            {default: False}
        extrapolation : bool
            Internal permission to attempt curve extrapolation for the fixed 1/7
            threshold. {default: False}
        
        Returns
        -------
        None
            Initializes the curve object; call calculate_resolution to evaluate
            crossings.
        """
        self.fixed_resolution = np.nan
        self.fixed_resolution_corrected = np.nan
        self.three_sigma_resolution = np.nan
        self.three_sigma_resolution_corrected = np.nan
        self.fixed_index = -1
        self.fixed_index_corrected = -1
        self.three_sigma_index = -1
        self.three_sigma_index_corrected = -1

        self.allow_extrapolation = bool(extrapolation)
        # Backward-compatible public attribute.
        self.extrapolation = self.allow_extrapolation
        self.extrapolation_required = False
        self.extrapolation_used = False
        self.extrapolation_attempted = False

        self.frequency = np.asarray(frequency)
        self.smoothed_frc = np.asarray(smoothed_frc)
        self.smoothed_frc_corrected = np.asarray(smoothed_frc_corrected)
        self.fixed_threshold = np.asarray(fixed_threshold)
        self.pixel_size = pixel_size
        self.num_pixels = np.asarray(num_pixels)
        self.single_frame = single_frame
        self.three_sigma = 3 / np.sqrt(self.num_pixels / 2)
        self.three_sigma[self.three_sigma > 1] = 1

        self.five_sigma = None
        self.two_sigma = None
        self.five_sigma_index_corrected = None
        self.five_sigma_resolution_corrected = None
        self.five_sigma_index = None
        self.five_sigma_resolution = None
        self.two_sigma_index_corrected = None
        self.two_sigma_resolution_corrected = None
        self.two_sigma_index = None
        self.two_sigma_resolution = None

        self.pre_length = max(self.frequency.shape)
        self.new_frequency = None
        self.extrapolated_frc = None
        self.extrapolated_fixed_threshold = None
        self.extrapolation_fit_info = None

    def _ensure_extrapolated_curve(self):
        """Prepare a twice-length fitted correlation extension when needed.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None
            Stores the extrapolated curve, frequencies, fixed threshold, and fit
            diagnostics.
        """
        if self.extrapolated_frc is not None:
            return
        self.new_frequency, self.extrapolated_frc, self.extrapolation_fit_info = frc_extrapolation(
            self.smoothed_frc_corrected if self.single_frame else self.smoothed_frc, 2 * self.pre_length, self.frequency,
            return_info=True)
        self.extrapolated_fixed_threshold = 1 / 7 * np.ones_like(self.extrapolated_frc)

    def calculate_resolution(self, additional_sigma=False, pad_factor=1, threshold='fixed_threshold'):
        """Calculate measured crossings and automatically extend a missing 1/7 crossing.
        
        Parameters
        ----------
        additional_sigma : bool
            Compute 2-sigma and 5-sigma estimates and apply padding calibration to sigma
            thresholds. {default: False}
        pad_factor : float
            Multiplicative calibration factor for the sigma threshold. {default: 1}
        threshold : str
            Correlation threshold selector: fixed_threshold, 3-sigma, 2-sigma, or
            5-sigma, as supported by the method. {default: 'fixed_threshold'}
        
        Returns
        -------
        None
            Updates resolution values, crossing indices, and
            extrapolation-required/attempted/used flags.
        
        Notes
        -----
        Automatic extrapolation requires allow_extrapolation=True,
        threshold=fixed_threshold, and no usable selected measured crossing. Sigma
        thresholds do not trigger extrapolation. Additional sigma estimates are intended
        for rolling-map fallbacks.
        """
        self.extrapolation_required = False
        self.extrapolation_used = False
        self.extrapolation_attempted = False

        self.fixed_resolution, self.fixed_index = find_intercept(
            self.smoothed_frc, self.fixed_threshold, self.pixel_size)
        self.fixed_resolution_corrected, self.fixed_index_corrected = find_intercept(
            self.smoothed_frc_corrected, self.fixed_threshold, self.pixel_size)

        self.three_sigma_resolution, self.three_sigma_index = find_intercept(
            self.smoothed_frc, self.three_sigma, self.pixel_size)
        self.three_sigma_resolution_corrected, self.three_sigma_index_corrected = find_intercept(
            self.smoothed_frc_corrected, self.three_sigma, self.pixel_size)

        if additional_sigma or pad_factor != 1:
            self.three_sigma = np.minimum(1, 3 / np.sqrt(self.num_pixels/2) * pad_factor)
            self.three_sigma_resolution,self.three_sigma_index=find_intercept(self.smoothed_frc,self.three_sigma,self.pixel_size)
            self.three_sigma_resolution_corrected,self.three_sigma_index_corrected=find_intercept(self.smoothed_frc_corrected,self.three_sigma,self.pixel_size)
        selected_resolution = self.get_resolution(threshold)
        if not np.isfinite(selected_resolution):
            self.extrapolation_required = True
            if self.allow_extrapolation and threshold == 'fixed_threshold':
                self.extrapolation_attempted = True
                try:
                    self._ensure_extrapolated_curve()
                    resolution, index = find_intercept(
                        self.extrapolated_frc, self.extrapolated_fixed_threshold,
                        self.pixel_size * self.pre_length / len(self.extrapolated_frc))
                    if np.isfinite(resolution) and resolution > 0:
                        if self.single_frame:
                            self.fixed_resolution_corrected, self.fixed_index_corrected = resolution,index
                        else:
                            self.fixed_resolution, self.fixed_index = resolution,index
                        self.extrapolation_used = True
                except (RuntimeError,ValueError,FloatingPointError):
                    pass

        if additional_sigma:
            self.two_sigma = 2 / np.sqrt(self.num_pixels / 2) * pad_factor
            self.five_sigma = 5 / np.sqrt(self.num_pixels / 2) * pad_factor
            self.two_sigma[self.two_sigma > 1] = 1
            self.five_sigma[self.five_sigma > 1] = 1
            self.two_sigma_resolution, self.two_sigma_index = find_intercept(
                self.smoothed_frc, self.two_sigma, self.pixel_size)
            self.two_sigma_resolution_corrected, self.two_sigma_index_corrected = find_intercept(
                self.smoothed_frc_corrected, self.two_sigma, self.pixel_size)
            self.five_sigma_resolution, self.five_sigma_index = find_intercept(
                self.smoothed_frc, self.five_sigma, self.pixel_size)
            self.five_sigma_resolution_corrected, self.five_sigma_index_corrected = find_intercept(
                self.smoothed_frc_corrected, self.five_sigma, self.pixel_size)

    def get_resolution(self, threshold='fixed_threshold'):
        """Read the resolution associated with a selected threshold.
        
        Parameters
        ----------
        threshold : str
            Correlation threshold selector: fixed_threshold, 3-sigma, 2-sigma, or
            5-sigma, as supported by the method. {default: 'fixed_threshold'}
        
        Returns
        -------
        float
            Resolution in nm, using single-frame-corrected data when appropriate.
            Missing estimates are NaN.
        
        Notes
        -----
        2-sigma and 5-sigma returns include the existing fallback calibration factors
        and require additional_sigma=True during calculation.
        """
        if threshold == 'fixed_threshold':
            return self.fixed_resolution_corrected if self.single_frame else self.fixed_resolution
        if threshold == '3-sigma':
            return self.three_sigma_resolution_corrected if self.single_frame else self.three_sigma_resolution
        if threshold == '2-sigma':
            if self.two_sigma_resolution is None:
                return np.nan
            return 1.5 * (self.two_sigma_resolution_corrected if self.single_frame
                          else self.two_sigma_resolution)
        if threshold == '5-sigma':
            if self.five_sigma_resolution is None:
                return np.nan
            return 3.0 / 5 * (self.five_sigma_resolution_corrected if self.single_frame
                              else self.five_sigma_resolution)
        raise ValueError(f'Unknown threshold: {threshold}')

    def extrapolated(self):
        """Report whether an extrapolated crossing supplied the selected estimate.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        bool
            True only after successful use of extrapolation; an attempted fit alone is
            insufficient.
        """
        return self.extrapolation_used

    def needs_extrapolation(self):
        """Report whether the selected measured curve lacked a usable crossing.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        bool
            The measured-crossing-unavailable flag from the latest calculation.
        """
        return self.extrapolation_required

    def get_extrapolation_fit_summary(self, context='sFRC'):
        """Format diagnostics for a successfully used extrapolation fit.
        
        Parameters
        ----------
        context : str
            Analysis-stage label used in diagnostics. {default: 'sFRC'}
        
        Returns
        -------
        str or None
            Context-labelled A, k, R-squared, RMSE, and fallback status, or None if no
            extrapolated estimate was used.
        """
        info = self.extrapolation_fit_info
        if not self.extrapolation_used or not info:
            return None
        suffix = ' [linearized fallback]' if info.get('used_fallback', False) else ''
        return (
            f'{context} - curve extrapolation fit: '
            f"A={info['A']:.6g}, k(norm)={info['k']:.6g}, "
            f"R^2={info['r_squared']:.4f}, RMSE={info['rmse']:.4g}{suffix}"
        )

    def get_index(self, threshold='fixed_threshold'):
        """Read the selected fixed or 3-sigma crossing index.
        
        Parameters
        ----------
        threshold : str
            Correlation threshold selector: fixed_threshold, 3-sigma, 2-sigma, or
            5-sigma, as supported by the method. {default: 'fixed_threshold'}
        
        Returns
        -------
        int
            Zero-based curve index; -1 denotes an unavailable or unsupported crossing
            index.
        """
        if threshold == 'fixed_threshold':
            return self.fixed_index_corrected if self.single_frame else self.fixed_index
        if threshold == '3-sigma':
            return self.three_sigma_index_corrected if self.single_frame else self.three_sigma_index
        return -1

    def sigma_curve_calibration(self, pad_factor):
        """Scale the stored 3-sigma threshold by the padding factor.
        
        Parameters
        ----------
        pad_factor : float
            Multiplicative calibration factor for the sigma threshold.
        
        Returns
        -------
        None
            Updates the 3-sigma array in place; retained 2-sigma/5-sigma arrays are
            unchanged.
        """
        self.three_sigma = self.three_sigma * pad_factor
        if self.two_sigma is not None:
            self.two_sigma = self.two_sigma
        if self.five_sigma is not None:
            self.five_sigma = self.five_sigma

    def plot_results(self, threshold='fixed_threshold', save_path=None, title='sFRC - correlation curve', show=False):
        """Backward-compatible entry point for the unified plotter."""
        from src.utils.plotting import ResolutionPlotter
        return ResolutionPlotter.frc_curve(self, threshold=threshold, save_path=save_path, title=title, show=show)

    def to_dict(self):
        """Serialize the basic correlation arrays and extrapolation state.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        dict
            Frequency, measured/corrected curves, fixed threshold, and extrapolation
            flags as Python values.
        """
        return {
            'frequency': self.frequency.tolist(),
            'smoothed_frc': self.smoothed_frc.tolist(),
            'smoothed_frc_corrected': self.smoothed_frc_corrected.tolist(),
            'fixed_threshold': self.fixed_threshold.tolist(),
            'extrapolation_required': self.extrapolation_required,
            'extrapolation_used': self.extrapolation_used,
        }


def plot_frc_comparison(single_curve, paired_curve, threshold='fixed_threshold',
                        title='Single-frame and paired-frame FRC',
                        save_path=None, show=False):
    """Backward-compatible overlay of single-frame and paired-frame FRC."""
    from src.utils.plotting import ResolutionPlotter
    return ResolutionPlotter.frc_comparison(single_curve, paired_curve, threshold=threshold,
                                            title=title, save_path=save_path, show=show)
