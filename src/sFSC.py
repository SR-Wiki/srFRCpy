from src.utils import fft_backend
import warnings
import time
from src.utils.input_image import normalize_image
import numpy as np
from src.utils.image_operations import (
    diagonal_resample, select_planes, pad2even, sampling_z, sampling_xy, pad2cube
)
from src.utils.window_functions import hanning_2d, hanning_3d
from src.utils.correlations import perform_frc
from tqdm import tqdm
from src.utils.polar_plot import polar_plot
from src.utils.drift_correct import drift_correct
from src.utils.directional_resolution_inference import DirectionalResolutionInference
from src.utils.analysis_results import (DRI_GUIDANCE,dri_interpretation,with_sfsc_dri_guidance,
                                    normalize_threshold)
from src.utils.sfsc_guidance import validate_num_cones, ANGLE_GUIDANCE, resolve_z_sampling, angle_range_notice, z_sampling_note


def _validate_settings(pixel_size, xy_mode, mean_filter_width,
                       low_angle, high_angle, num_cones, threshold):
    """Validate sFSC sampling and fitting settings.
    
    Parameters
    ----------
    pixel_size : array_like
        Required positive voxel spacings [z,y,x] in nm. None is a placeholder and must
        be replaced with the acquisition calibration.
    xy_mode : str
        'slice' is the default; consider '3D' for low-SNR data.
        Lateral estimate from selected Z slices (slice) or the central plane of the 3D
        FFT (3D).
    mean_filter_width : int
        Moving-mean neighborhood radius in frequency bins; must be at least 1.
    low_angle : float
        Lower DRI fitting angle in degrees; values above 10 degrees are recommended.
    high_angle : float
        Upper DRI fitting angle in degrees; values below 80 degrees are recommended.
    num_cones : int
        Positive number of directional cones. At least 5 is recommended, but smaller
        positive counts are accepted.
    threshold : str
        Correlation threshold: fixed_threshold represents 1/7; 3-sigma is also
        supported.
    
    Returns
    -------
    numpy.ndarray
        Copied, positive finite voxel spacings in nm, ordered [z,y,x].
    
    Notes
    -----
    Also emits sampling-anisotropy and fitting-angle guidance where appropriate.
    """
    pixel_size = np.asarray(pixel_size, dtype=float).reshape(-1).copy()
    if pixel_size.size != 3 or not np.all(np.isfinite(pixel_size)) or np.any(pixel_size <= 0):
        raise ValueError('Enter three positive, finite voxel spacings in nm, ordered [z, y, x].')
    if xy_mode not in ('slice', '3D'):
        raise ValueError("xy_mode must be 'slice' or '3D'.")
    if mean_filter_width < 1:
        raise ValueError('mean_filter_width must be >= 1.')
    if not (0 <= low_angle < high_angle <= 90):
        raise ValueError('Angles must satisfy 0 <= low_angle < high_angle <= 90 degrees.')
    validate_num_cones(num_cones)
    if threshold not in ('fixed_threshold', '3-sigma'):
        raise ValueError("threshold must be 'fixed_threshold' or '3-sigma'.")
    ratio = np.max(pixel_size) / np.min(pixel_size)
    if ratio >= 10:
        warnings.warn(
            f'Sampling is strongly anisotropic (largest/smallest spacing = {ratio:.1f}x). '
            'Check the voxel spacings and their units before interpreting the result.',
            RuntimeWarning,
            stacklevel=3,
        )
    angle_range_notice(low_angle, high_angle)
    return pixel_size


def _resolution_or_warning(frc_data, threshold, extrapolation, context):
    """Read a usable lateral resolution or mark it unavailable.
    
    Parameters
    ----------
    frc_data : FourierCorrelationData
        Lateral correlation result to inspect.
    threshold : str
        Correlation threshold: fixed_threshold represents 1/7; 3-sigma is also
        supported.
    extrapolation : bool
        Internal permission to attempt curve extrapolation for the fixed 1/7 threshold.
    context : str
        Analysis-stage label used in diagnostics.
    
    Returns
    -------
    float
        Resolution in nm, or NaN when no usable estimate is present.
    
    Notes
    -----
    The context and extrapolation arguments are retained for compatibility; this helper
    does not emit a warning itself.
    """
    resolution = frc_data.get_resolution(threshold=threshold)
    if np.isfinite(resolution):
        return resolution
    if frc_data.needs_extrapolation() and not extrapolation:
        return np.nan
    if extrapolation and threshold != 'fixed_threshold':
        return np.nan
    return np.nan


@with_sfsc_dri_guidance
def sFSC(im1, im2=None, pixel_size=None, mean_filter_width=3, xy_mode='slice',
         single_frame=True, z_sampling=False, low_angle=15, high_angle=60,
         num_cones=5, threshold='1/7 hard threshold', motion_correction=False, plot_fitting_curve=False):
    """Estimate lateral and axial resolution of a 3D volume with directional resolution inference.
    
    Parameters
    ----------
    im1 : numpy.ndarray
        First 3D volume with axes ordered (Z,Y,X).
    im2 : numpy.ndarray or None
        Matching second (Z,Y,X) volume, required when single_frame=False. See Notes for
        supplied im2 in single-frame mode. {default: None}
    pixel_size : array_like
        Required positive voxel spacings [z,y,x] in nm. None is a placeholder and must
        be replaced with the acquisition calibration. {default: None}
    mean_filter_width : int
        Moving-mean neighborhood radius in frequency bins; must be at least 1.
        {default: 3}
    xy_mode : str
        Lateral estimate from selected Z slices (slice) or the central plane of the 3D
        FFT (3D). {default: 'slice'}
    single_frame : bool
        Generate an XY-resampled pair from im1; False analyses the supplied volume pair.
        {default: True}
    z_sampling : bool
        Manually enable odd/even Z sampling; disabled by default, independently of the
        voxel spacings. {default: False}
    low_angle : float
        Lower DRI fitting angle in degrees; values above 10 degrees are recommended.
        {default: 15}
    high_angle : float
        Upper DRI fitting angle in degrees; values below 80 degrees are recommended.
        {default: 60}
    num_cones : int
        Positive number of directional cones. At least 5 is recommended, but smaller
        positive counts are accepted. {default: 5}
    threshold : str
        '1/7 hard threshold' or '3-sigma'. fixed_threshold and 1/7 remain
        aliases of the fixed criterion. {default: '1/7 hard threshold'}
    motion_correction : bool
        Align paired input images before correlation; ignored for single-frame sampling.
        {default: False}
    plot_fitting_curve : bool
        Additionally display the DRI elliptical fit. {default: False}
    
    Returns
    -------
    DirectionalResolutionInference
        Result with lateral/axial resolutions in nm, angular samples in radians, cutoffs
        in 1/um, and goodness_of_fit as a relative fraction. summary includes
        extrapolation_triggered, extrapolation_stage, lateral_resolution_nm,
        axial_resolution_nm, and E_DRI_percent. lateral_curve, selected_plane,
        parameters, and z_sampling_enabled provide the associated context.
        computation_time_seconds stores total numerical time, while
        stage_elapsed_seconds contains only lateral and axial seconds.
    
    Notes
    -----
    Input arrays are copied to float32 and divided by their maximum before
    resampling; each stack/volume shares one maximum. All-zero data remain zero.
    Both volumes use (Z,Y,X). Single-frame mode generates an XY-resampled pair
    internally. A supplied im2 is not used for that pair, but is still checked for a
    matching shape. In paired mode im2 is required. Z sampling is OFF by default and
    must be enabled manually; consider comparing the enabled result if the axial
    estimate appears overly optimistic. DRI extrapolation is attempted only when every
    measured directional curve lacks a usable crossing, and only 1/7 is supported. E_DRI
    <10% is reliable, 10-20% requires caution, and >20% is unacceptable; a high value
    suggests adjusting the fitting angle range. An available result is always plotted;
    there is no plot/plot_results switch. This function does not save files.
    Every call appends the complete E_DRI grading guide on
    exit, including early returns and errors, independently of plotting.
    """
    extrapolation = True  # internal policy, no user-facing permission switch
    threshold = normalize_threshold(threshold)
    lateral_triggered=False
    pixel_size = _validate_settings(
        pixel_size, xy_mode, mean_filter_width,
        low_angle, high_angle, num_cones, threshold)

    z_sampling = resolve_z_sampling(z_sampling)
    parameters=dict(pixel_size_nm=pixel_size.tolist(),mean_filter_width=mean_filter_width,
                    single_frame=bool(single_frame),z_sampling=bool(z_sampling),
                    low_angle=low_angle,high_angle=high_angle,num_cones=num_cones,
                    threshold=threshold,xy_mode=xy_mode,motion_correction=bool(motion_correction))
    computation_started = time.perf_counter()
    lateral_started = computation_started  # Include input preparation in lateral timing.
    im1 = normalize_image(im1)
    if im2 is not None:
        im2 = normalize_image(im2)
        if im1.shape != im2.shape:
            raise ValueError('Provide two volumes with matching shapes.')
    if im1.ndim != 3:
        raise ValueError('Provide a 3D volume with axes ordered (Z,Y,X).')

    analysis_pixel_size = pixel_size.copy()
    if single_frame:
        im1, im2 = diagonal_resample(im1)
        analysis_pixel_size[1:] *= 2.0
    else:
        if im2 is None:
            raise ValueError('Paired-volume mode needs a second volume (im2).')
        for i in range(im1.shape[0]) if motion_correction else []:
            im1[i], im2[i] = drift_correct(im1[i], im2[i], analysis_pixel_size[1])

    min_spacing = np.min(analysis_pixel_size)
    max_spacing = np.max(analysis_pixel_size)

    print('sFSC - lateral: estimating resolution...')
    d, h, w = im1.shape
    selected_frc = None
    selected_plane = None
    xy_resolution = np.nan
    extrapolation_required = False

    if xy_mode == 'slice':
        hn = hanning_2d(h, w)
        planes_xy = np.arange(d) if d <= 20 else select_planes(im1, 20)
        best = np.inf
        progress_bar = tqdm(total=len(planes_xy), desc='slice-based lateral resolution estimation', unit='plane')
        for i in planes_xy:
            ffta = fft_backend.fftshift(fft_backend.fft2(im1[i] * hn))
            fftb = fft_backend.fftshift(fft_backend.fft2(im2[i] * hn))
            frc_data = perform_frc(
                ffta, fftb, analysis_pixel_size[1],
                single_frame=single_frame,
                mean_filter_width=mean_filter_width,
                extrapolation=extrapolation,
            )
            frc_data.calculate_resolution(threshold=threshold)
            temp_resolution = _resolution_or_warning(
                frc_data, threshold, extrapolation, 'lateral FRC')
            extrapolation_required |= frc_data.needs_extrapolation()
            lateral_triggered |= frc_data.extrapolation_attempted
            progress_bar.update(1)

            lower_bound = (1.15 if single_frame else 2.3) * analysis_pixel_size[1]
            if np.isfinite(temp_resolution) and temp_resolution >= lower_bound and temp_resolution < best:
                best = temp_resolution
                selected_frc = frc_data
                selected_plane = int(i)
        progress_bar.close()
        if selected_frc is not None:
            xy_resolution = best
    else:
        hn = hanning_3d(d, h, w)
        ffta = fft_backend.fftshift(fft_backend.fftn(im1 * hn))
        fftb = fft_backend.fftshift(fft_backend.fftn(im2 * hn))
        zc = d // 2
        selected_frc = perform_frc(
            ffta[zc, :, :], fftb[zc, :, :], min_spacing,
            single_frame=single_frame,
            mean_filter_width=mean_filter_width,
            extrapolation=extrapolation,
        )
        selected_frc.calculate_resolution(threshold=threshold)
        extrapolation_required = selected_frc.needs_extrapolation()
        lateral_triggered |= selected_frc.extrapolation_attempted
        xy_resolution = selected_frc.get_resolution(threshold=threshold)

    if not np.isfinite(xy_resolution):
        detail=('automatic extrapolation supports 1/7 only' if threshold!='fixed_threshold'
                else 'no usable lateral crossing after automatic extrapolation')
        warnings.warn(f'sFSC - lateral: {detail}. Axial resolution and E_DRI are unavailable; check the pixel size and threshold. {DRI_GUIDANCE}',
                      RuntimeWarning,stacklevel=2)
        # Preserve the usual return type, marking all unavailable quantities.
        erf=DirectionalResolutionInference(max_spacing,1,single_frame=single_frame,z_sampling=z_sampling,
            low_angle=low_angle,high_angle=high_angle,num_cones=num_cones,threshold=threshold)
        erf.xy_resolution,erf.xy_cutoff,erf.z_resolution,erf.z_cutoff=np.nan,np.nan,np.nan,np.nan
        erf.angles,erf.cutoffs=np.empty(0),np.empty(0)
        erf.summary=dict(lateral_resolution_nm=np.nan,axial_resolution_nm=np.nan,E_DRI_percent=np.nan,
            singleframe=bool(single_frame),threshold=threshold,
            extrapolation_triggered=bool(lateral_triggered),extrapolation_stage='lateral' if lateral_triggered else 'none')
        erf.parameters=parameters
        erf.z_sampling_enabled=bool(z_sampling)
        erf.lateral_curve=selected_frc
        erf.selected_plane=selected_plane
        lateral_seconds = time.perf_counter() - lateral_started
        erf.computation_time_seconds = time.perf_counter() - computation_started
        erf.stage_elapsed_seconds = dict(lateral=lateral_seconds, axial=0.0)
        print(f'sFSC - numerical computation time: {erf.computation_time_seconds:.3f} s '
              '(axial computation unavailable; excludes plotting and file export).')
        return erf

    if xy_mode == 'slice' and selected_plane is not None:
        print(f'sFSC - lateral: selected Z plane {selected_plane} (zero-based).')

    if selected_frc is not None and selected_frc.extrapolated():
        # Candidate slices are evaluated silently. Report extrapolation-fit
        # diagnostics only for the finally selected lateral slice.
        fit_summary = selected_frc.get_extrapolation_fit_summary(context='sFSC - lateral')
        if fit_summary is not None:
            print(fit_summary)
        print('sFSC - lateral: No lateral threshold crossing was found in the measured range. Automatic 1/7 extrapolation provided the lateral estimate.')
    print(f'\nsFSC - lateral resolution: {xy_resolution:.2f} nm')

    lateral_seconds = time.perf_counter() - lateral_started
    axial_started = time.perf_counter()
    print('sFSC - axial: estimating resolution with DRI...')
    if z_sampling:
        im1, im2 = pad2even(im1), pad2even(im2)
        im1, im2 = im1[0::2], im2[1::2]
        analysis_pixel_size[0] *= 2.0

    downsampling_factor = analysis_pixel_size[0] / analysis_pixel_size[1]
    if downsampling_factor <= 1:
        tmp_im1 = sampling_z(im1, 1.0 / downsampling_factor)
        tmp_im2 = sampling_z(im2, 1.0 / downsampling_factor)
    else:
        tmp_im1 = sampling_xy(im1, downsampling_factor)
        tmp_im2 = sampling_xy(im2, downsampling_factor)

    max_spacing = np.max(analysis_pixel_size)
    tmp_im1, tmp_im2 = pad2cube(tmp_im1), pad2cube(tmp_im2)
    # d, h, w = tmp_im1.shape
    # hn = hanning_3d(d, h, w)
    ffta = fft_backend.fftshift(fft_backend.fftn(tmp_im1))
    fftb = fft_backend.fftshift(fft_backend.fftn(tmp_im2))

    print('sFSC - DRI: fitting directional cutoffs...')
    erf = DirectionalResolutionInference(
        max_spacing, xy_resolution,
        single_frame=single_frame,
        z_sampling=z_sampling,
        low_angle=low_angle,
        high_angle=high_angle,
        num_cones=num_cones,
        threshold=threshold,
        extrapolation=extrapolation,
    )
    try:
        erf.fit(ffta,fftb)
    except RuntimeError as error:
        warnings.warn(f'sFSC - axial: the DRI fit could not provide an estimate ({error}). Check the fitting angles and voxel calibration.',RuntimeWarning,stacklevel=2)
        erf.z_resolution,erf.z_cutoff=np.nan,np.nan
    if not np.isfinite(erf.goodness_of_fit):print(f'E_DRI is unavailable because no usable DRI fit was obtained. {DRI_GUIDANCE}')
    z_resolution,_=erf.get_z_resolution()
    erf.z_sampling_enabled=bool(z_sampling)
    erf.interpretation_note=z_sampling_note(single_frame,z_sampling,z_resolution)
    if erf.interpretation_note: print('sFSC - axial: '+erf.interpretation_note)
    stage=('lateral and axial' if erf.extrapolation_attempted else 'lateral') if lateral_triggered else ('axial' if erf.extrapolation_attempted else 'none')
    row=dict(lateral_resolution_nm=xy_resolution,axial_resolution_nm=z_resolution,E_DRI_percent=100*erf.goodness_of_fit,
             singleframe=bool(single_frame),threshold=threshold,
             extrapolation_triggered=bool(lateral_triggered or erf.extrapolation_attempted),extrapolation_stage=stage)
    erf.summary=row
    erf.parameters=parameters
    erf.lateral_curve=selected_frc
    erf.selected_plane=selected_plane
    axial_seconds = time.perf_counter() - axial_started
    erf.computation_time_seconds = time.perf_counter() - computation_started
    erf.stage_elapsed_seconds = dict(lateral=lateral_seconds, axial=axial_seconds)
    print(f'sFSC - numerical computation time: {erf.computation_time_seconds:.3f} s '
          '(excludes plotting and file export).')
    for name, seconds in erf.stage_elapsed_seconds.items():
        print(f'  {name}: {seconds:.3f} s')
    if np.isfinite(z_resolution):
        polar_plot(erf,show=True)
    if plot_fitting_curve and np.isfinite(z_resolution):
        from src.utils.plotting import ResolutionPlotter
        ResolutionPlotter.dri_fit(erf, show=True)
    return erf
