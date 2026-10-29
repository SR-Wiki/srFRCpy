from src.utils import fft_backend
import numpy as np
from src.utils.image_operations import diagonal_resample
from src.utils.window_functions import hanning_2d
from src.utils.correlations import perform_frc
from src.utils.drift_correct import drift_correct
from src.utils.analysis_results import (frame_pairs,extrapolation_notice,resolve_plot,
                                    normalize_threshold,current_result_index)
from src.utils.input_image import normalize_image


def _validate_pixel_size(pixel_size):
    """Validate calibrated XY pixel spacings.
    
    Parameters
    ----------
    pixel_size : float or array_like
        Required positive calibrated XY spacing in nm: a scalar or [y,x]. None is a
        placeholder and must be replaced with the acquisition calibration.
    
    Returns
    -------
    numpy.ndarray
        Copied spacing array in nm; a scalar is repeated for Y and X.
    """
    values=np.asarray(pixel_size,dtype=float).reshape(-1).copy()
    if values.size==1: values=np.repeat(values,2)
    if values.size<2 or np.any(~np.isfinite(values[:2])) or np.any(values[:2]<=0):
        raise ValueError('Enter positive, finite XY pixel spacings in nm.')
    return values


def sFRC(im1,im2=None,pixel_size=None,mean_filter_width=3,single_frame=True,
         threshold='1/7 hard threshold',plot='auto',stack_axis=0,motion_correction=False,batch_processing=True,current_frame=0):
    """Estimate Fourier ring correlation resolution from 2D images or image stacks.
    
    Parameters
    ----------
    im1 : numpy.ndarray
        One 2D image (Y,X) or image stack, normally (T,Y,X), used by both modes.
    im2 : numpy.ndarray or None
        Second image/stack with matching shape for paired mode. Omit it to use adjacent
        pairs from im1; ignored in single-frame mode. {default: None}
    pixel_size : float or array_like
        Required positive calibrated XY spacing in nm: a scalar or [y,x]. None is a
        placeholder and must be replaced with the acquisition calibration.
        {default: None}
    mean_filter_width : int
        Moving-mean neighborhood radius in frequency bins; must be at least 1.
        {default: 3}
    single_frame : bool
        Generate a diagonal resampled pair independently from each input frame.
        False compares adjacent frames of the one input stack when im2 is omitted.
        {default: True}
    threshold : str
        '1/7 hard threshold' or '3-sigma'. fixed_threshold and 1/7 are legacy
        aliases of the fixed criterion. {default: '1/7 hard threshold'}
    plot : bool or str
        'auto' and True show the first frame/pair in batch mode, or the selected result otherwise. Paired
        mode shows one calculation for that frame and the next frame. False
        hides plots without changing the computation batch. {default: 'auto'}
    stack_axis : int
        Axis containing independent 2D frames; 0 means (T,Y,X), and 2 means (Y,X,T).
        {default: 0}
    motion_correction : bool
        Align paired input images before correlation; ignored for single-frame sampling.
        {default: False}
    batch_processing : bool
        Process every available frame or adjacent pair; False selects current_frame
        only. {default: True}
    current_frame : int
        Zero-based preview frame; also selects computation when batching is False. A pair
        needs the following frame. {default: 0}
    
    Returns
    -------
    FourierCorrelationData or list of FourierCorrelationData
        One curve object for one analysed frame or pair; otherwise an ordered list. Each
        object contains summary and parameters dictionaries. summary records the frame,
        singleframe, threshold, extrapolation_triggered, and resolution_nm.
    
    Notes
    -----
    Input arrays are copied to float32 and divided by their maximum before
    resampling; each stack/volume shares one maximum. All-zero data remain zero.
    Images use (Y,X); stacks default to (T,Y,X). Single-frame sampling crops odd XY
    edges and generates a diagonal pair internally, doubling the analysis spacing.
    With one N-frame stack and batching enabled, single-frame mode produces N
    results; paired mode produces N-1 overlapping pairs: (1,2), (2,3), and so on.
    For MATLAB-style (Y,X,T) input, set stack_axis=2.
    Paired mode uses matching frames from im2, or consecutive overlapping pairs from im1
    when im2 is omitted. Auto/True plotting shows only current_frame, independently
    of batching; all requested numerical results are retained. Automatic extrapolation is attempted only
    when the selected 1/7 curve has no measured crossing. No files are written here;
    demos perform optional exports to results/.
    """
    spacing=_validate_pixel_size(pixel_size)
    if mean_filter_width<1: raise ValueError('Choose a mean_filter_width of at least 1.')
    threshold=normalize_threshold(threshold)
    im1=normalize_image(im1)
    if not single_frame and im2 is not None:im2=normalize_image(im2)
    # current_frame is used only for individual (non-batch) selection.
    if batch_processing:
        current_frame = 0
    pairs=frame_pairs(im1,im2,single_frame,stack_axis,batch_processing,current_frame)
    plot=resolve_plot(plot,batch_processing)
    plot_index=current_result_index(len(pairs),batch_processing,current_frame,single_frame,notify=plot)
    parameters=dict(pixel_size_nm=spacing.tolist(),mean_filter_width=mean_filter_width,
                    single_frame=bool(single_frame),threshold=threshold,
                    stack_axis=stack_axis,motion_correction=bool(motion_correction),
                    batch_processing=bool(batch_processing),
                    plot=plot,plot_index=plot_index)
    if not batch_processing:
        parameters['current_frame'] = current_frame
    curves=[]
    for result_index,(first,second,label) in enumerate(pairs):
        analysis_spacing=spacing[1]
        if single_frame:
            first,second=diagonal_resample(first)
            analysis_spacing*=2
        elif motion_correction:
            first,second=drift_correct(first,second,analysis_spacing)
        hn=hanning_2d(*first.shape)
        curve=perform_frc(fft_backend.fftshift(fft_backend.fft2(first*hn)),
                          fft_backend.fftshift(fft_backend.fft2(second*hn)),analysis_spacing,
                          single_frame=single_frame,mean_filter_width=mean_filter_width,extrapolation=True)
        curve.calculate_resolution(threshold=threshold)
        extrapolation_notice(curve,threshold,f'sFRC - {label}')
        curve.summary=dict(frame=label,singleframe=bool(single_frame),threshold=threshold,
                           extrapolation_triggered=curve.extrapolation_attempted,
                           resolution_nm=curve.get_resolution(threshold))
        curve.parameters=parameters.copy()
        curves.append(curve)
        if plot and result_index==plot_index:
            curve.plot_results(threshold,title=f'sFRC - correlation curve - {label}',show=True)
    return curves[0] if len(curves)==1 else curves
