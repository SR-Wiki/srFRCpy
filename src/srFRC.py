from src.utils import fft_backend
import os
from src.utils.input_image import normalize_image
from src.utils.analysis_results import (frame_pairs,restore_map_size,map_statistics,
                                   finalize_resolution_map,resolve_plot,normalize_threshold,
                                   current_result_index)

from src.utils.drift_correct import drift_correct
from src.utils.image_operations import *
from skimage import filters
from src.utils.window_functions import hanning_2d
from src.utils import fft_backend as ft
from src.utils.correlations import perform_frc
import matplotlib.pyplot as plt
from src.utils.shifted_jet import sjet_colorbar
from src.utils.srfrc_plot import plot_srfrc_results
import matplotlib.colors as mcolors
from src.utils.otsu import otsu
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import multiprocessing as mp



# Initialized once in each worker process; task messages then contain only patch
# bounds, not two full padded images or the analysis object.
_ROLLING_WORKER_STATE = None


def _init_rolling_worker(analyzer, raw_a, raw_b, shape, dtype, threshold, hm):
    global _ROLLING_WORKER_STATE
    a = np.frombuffer(raw_a, dtype=np.dtype(dtype)).reshape(shape)
    b = np.frombuffer(raw_b, dtype=np.dtype(dtype)).reshape(shape)
    _ROLLING_WORKER_STATE = (analyzer, a, b, threshold, hm)


def _run_rolling_patch(bounds):
    analyzer, pad_a, pad_b, threshold, hm = _ROLLING_WORKER_STATE
    return analyzer._process_patch(
        bounds, pad_a, pad_b, threshold, analyzer.block_size, analyzer.skip,
        analyzer.single_frame, analyzer._analysis_pixel_size, hm)


def _shared_copy(array):
    # RawArray is backed by OS shared memory, works with spawn on Windows.
    array = np.ascontiguousarray(array)
    dtype = np.dtype('float32') if array.dtype == np.float32 else np.dtype('float64')
    if array.dtype != dtype:
        array = array.astype(dtype)
    raw = mp.RawArray('f' if dtype == np.float32 else 'd', array.size)
    np.frombuffer(raw, dtype=dtype)[:] = array.ravel()
    return raw, str(dtype)


class RollingFourierRingCorrelation:
    def __init__(self, pixel_size, adaptive_background=True, single_frame=True,
                 block_size=64, monitor_change=False, background_threshold=30,
                 skip=2, parallel=True, patch_size=64, plot=True, stack_axis=0, motion_correction=False, batch_processing=True,
                 threshold='3-sigma'):
        """Configure rolling sFRC resolution-map estimation for images or stacks.
        
        Parameters
        ----------
        pixel_size : float
            Positive calibrated pixel spacing in nm.
        adaptive_background : bool
            Use 0.8 times the first image Otsu threshold. Forced to False when
            monitor_change is True. {default: True}
        single_frame : bool
            Generate a diagonal pair from each input image or frame; False uses paired
            acquisitions. {default: True}
        block_size : int
            Side length of each rolling correlation window in pixels. {default: 64}
        monitor_change : bool
            Compare Otsu multipliers 0.6, 0.8, and 1.0. Takes priority and forces
            adaptive_background=False; the manual threshold is ignored. {default: False}
        background_threshold : float
            Manual background intensity threshold on the normalized 0-255 scale.
            Used only when both background switches are False. {default: 30}
        skip : int
            Positive spatial step between rolling windows, in pixels. {default: 2}
        parallel : bool
            Use multiprocessing for rolling-map patches. {default: True}
        patch_size : int
            Side length of a spatial patch assigned to a worker, in pixels.
            {default: 64}
        plot : bool
            True previews only the first frame/pair in batch or non-batch mode.
            False suppresses automatic plots without changing the calculation.
            The old 'auto' spelling is still accepted. {default: True}
        stack_axis : int
            Axis containing independent 2D frames; 0 means (T,Y,X), and 2 means (Y,X,T).
            {default: 0}
        motion_correction : bool
            Align paired input images before correlation; ignored for single-frame
            sampling. {default: False}
        batch_processing : bool
            Process every available frame or adjacent pair; False processes only the first
            frame/pair. {default: True}
        threshold : str
            Primary criterion: '1/7 hard threshold' or '3-sigma'. Existing calibrated
            5-sigma then 2-sigma fallbacks remain internal. {default: '3-sigma'}
        
        Returns
        -------
        None
            Initializes the analyser; call execute with image arrays to compute maps.
        
        Notes
        -----
        pixel_size always refers to the original acquisition spacing. block_size is
        rounded up to an even integer; skip is capped at half the block size. Batch
        processing is enabled by default; only the first frame/pair is previewed in either mode. Automatic background uses
        0.8 times Otsu. monitor_change=True disables adaptive_background and
        independently compares 0.6, 0.8, and 1.0 times Otsu. Manual background is
        used only when both switches are False, on a 0-255 intensity scale. The maintained
        API keeps numerical results in memory and leaves exports to demos.
        """
        self.map_interpolation_backend = 'cubic_spline' if single_frame else 'none'
        self.pixel_size = float(pixel_size)
        self.threshold = normalize_threshold(threshold)
        self._analysis_pixel_size = self.pixel_size
        # Monitoring three backgrounds always uses the first frame/pair.
        effective_batch = bool(batch_processing) and not bool(monitor_change)
        self.plot=resolve_plot(plot,effective_batch)
        self.batch_processing=effective_batch
        self.stack_axis,self.motion_correction=stack_axis,motion_correction
        if self.pixel_size<=0 or not np.isfinite(self.pixel_size):raise ValueError('Enter a positive, finite pixel_size in nm.')
        if block_size<2 or skip<1 or patch_size<1:raise ValueError('Choose block_size >= 2, skip >= 1, and patch_size >= 1.')
        block_size=int(block_size)+(int(block_size)%2)
        skip=min(int(skip),block_size//2)
        self.monitor_change = bool(monitor_change)
        self.adaptive_background = bool(adaptive_background) and not self.monitor_change
        if self.monitor_change:
            print('srFRC - background monitoring enabled: batch processing is OFF; only the first frame/pair is analysed. Absolute thresholds are reported; manual threshold is ignored.')
        self.skip = skip
        self.single_frame = single_frame
        self.drift_correct = not self.single_frame
        self.block_size = block_size
        self.background_threshold = background_threshold
        self.parallel = parallel
        self.patch_size = patch_size
        self.rfrc_map = None
        self.mean_res = None
        self.min_res = None
        self.max_res = None
        self.rfrc_value = None
        self._global_statistics = None
        self.global_statistics_by_background = {}
        self.rfrc_maps = {}
        self.statistics_stage = 'pre_upsampling'
        self.statistics_map_shape = None
        self.panels = None
        # Monotonic wall-clock time in seconds; excludes optional plotting/export.
        self.computation_time_seconds = None
        self.elapsed_seconds = None  # convenient alias
        self.background_elapsed_seconds = {}
        self.frame_elapsed_seconds_by_background = {}
        self.parameters=dict(pixel_size_nm=self.pixel_size,single_frame=bool(single_frame),
            threshold=self.threshold,adaptive_background=self.adaptive_background,block_size=block_size,
            monitor_change=self.monitor_change,background_threshold=background_threshold,
            skip=skip,parallel=bool(parallel),patch_size=patch_size,stack_axis=stack_axis,
            motion_correction=bool(motion_correction),
            batch_processing=self.batch_processing,
            plot=self.plot,minimum_resolution_nm=2.3*self.pixel_size)

    def get_rfrc_map(self,background=None):
        """Return the latest completed resolution map.
        
        Parameters
        ----------
        background : str or None
            Optional group such as bg0.8, bg1.0, or bg1.2. None returns every
            group in monitoring mode. {default: None}
        
        Returns
        -------
        numpy.ndarray or dict
            Final map in nm, shaped (Y,X) for one result or (T,Y,X) for multiple
            results. Monitoring returns a dictionary containing all three groups;
            selecting one background returns that group's array.
        
        Notes
        -----
        Maps have already undergone original-size restoration and the final
        positive-value lower bound. The compatibility attribute rfrc_map still
        refers to the last group; this accessor no longer silently drops groups.
        """
        if not self.rfrc_maps:
            raise RuntimeError('Run execute(image) before requesting resolution maps.')
        if background is not None and background not in self.rfrc_maps:
            raise ValueError('Choose a background from: '+', '.join(self.rfrc_maps))
        labels=[background] if background is not None else list(self.rfrc_maps)
        maps={label:(self.rfrc_maps[label][0] if len(self.rfrc_maps[label])==1
                     else self.rfrc_maps[label]) for label in labels}
        return maps[labels[0]] if len(labels)==1 else maps

    def get_panels(self,background=None):
        """Build dimensionless quality panels from completed resolution maps.
        
        Parameters
        ----------
        background : str or None
            Select one background group, or retain all monitored groups.
            {default: None}
        
        Returns
        -------
        numpy.ndarray or dict
            Quality scores in [0,1] for one group, or one array per monitored group.
        """
        if self.rfrc_map is None:
            raise RuntimeError('Run execute(image) before requesting resolution panels.')
        maps=self.get_rfrc_map(background)
        self.panels=({label:otsu(data) for label,data in maps.items()}
                     if isinstance(maps,dict) else otsu(maps))
        return self.panels

    def calculate_global(self,background=None):
        """Return cached foreground statistics calculated before map upsampling.
        
        Parameters
        ----------
        background : str or None
            Select one group's native-grid statistics. None returns all groups
            in monitoring mode. {default: None}
        
        Returns
        -------
        numpy.ndarray or dict
            Columns are [min_resolution_nm, max_resolution_nm, mean_resolution_nm,
            rFRC_value]. One frame returns shape (4,); several return (T,4). The scalar
            analyser attributes retain the last frame/group for compatibility.
            Monitoring returns one statistics array per background label.
        
        Notes
        -----
        The cache is calculated on the native rolling grid after background cleanup
        and the 2.3-times-original-pixel-size lower bound. Only finite positive values
        contribute. rFRC_value is mean_resolution_nm / min_resolution_nm - 1.
        An empty foreground produces NaN statistics. No monitored group is dropped.
        This method does not recalculate values from display maps.
        """
        if self._global_statistics is None:
            raise RuntimeError('Run execute(image) before requesting global statistics.')
        rows=self._global_statistics
        last=rows[-1]
        self.mean_res,self.min_res,self.max_res,self.rfrc_value=(last[k] for k in ('mean_resolution_nm','min_resolution_nm','max_resolution_nm','rFRC_value'))
        groups=self.global_statistics_by_background
        if background is not None and background not in groups:
            raise ValueError('Choose a background from: '+', '.join(groups))
        labels=[background] if background is not None else list(groups)
        output={}
        for label in labels:
            values=np.asarray([[r['min_resolution_nm'],r['max_resolution_nm'],
                r['mean_resolution_nm'],r['rFRC_value']] for r in groups[label]])
            output[label]=values[0] if len(values)==1 else values
        return output[labels[0]] if len(labels)==1 else output

    def print_summary(self):
        """Print per-frame srFRC statistics with absolute background thresholds.

        Internal background-group keys are intentionally unchanged for saved
        maps, timings, and backward-compatible programmatic access.
        """
        if not getattr(self, 'metrics', None):
            raise RuntimeError('Run execute(image) before printing an srFRC summary.')
        print('srFRC - native-grid results (background intensity scale 0-255):')
        for row in self.metrics:
            print(
                f"  {row['frame']}: Background threshold = {row['background_threshold']:.2f}; "
                f"mean = {row['mean_resolution_nm']:.2f} nm; "
                f"min = {row['min_resolution_nm']:.2f} nm; "
                f"max = {row['max_resolution_nm']:.2f} nm; "
                f"rFRC = {row['rFRC_value']:.3f}"
            )

    def execute(self,image1,image2=None):
        """Compute rolling resolution maps and per-frame summaries from image arrays.
        
        Parameters
        ----------
        image1 : numpy.ndarray
            One 2D image (Y,X) or image stack, normally (T,Y,X), used by both modes.
        image2 : numpy.ndarray or None
            Matching second image/stack for paired mode. If omitted, adjacent pairs are
            selected from image1; ignored in single-frame mode. {default: None}
        
        Returns
        -------
        RollingFourierRingCorrelation
            The same analyser with rfrc_maps, rfrc_map, metrics,
            final_background_thresholds, global statistics, and numerical timing populated.
        
        Notes
        -----
        Inputs are copied to float32 and divided by the array maximum before
        resampling. Each stack shares one maximum; an all-zero input stays zero.
        Images use (Y,X); stacks default to (T,Y,X). In single-frame mode image2 is
        ignored and each image generates its own diagonal pair. Paired mode uses
        matching frames from image2 or consecutive overlapping pairs from image1.
        With one N-frame stack and batching enabled, single-frame mode produces N
        maps per background setting; paired mode produces N-1 maps for pairs (1,2),
        (2,3), and so on. Set stack_axis=2 in the constructor for (Y,X,T) input.
        Single-frame maps use separable not-a-knot cubic-spline restoration,
        retaining the measured foreground support and positive source range.
        Non-positive map values are set to background zero; finite
        positive values are at least 2.3 times the original pixel size. All summary
        statistics are computed on the cleaned native rolling map BEFORE spline
        restoration. They are cached for calculate_global and exports; reducing the
        restored image can give a different mean. Timing fields are
        computation_time_seconds (total) and background_elapsed_seconds (one per
        background group), measured with a monotonic wall clock in seconds.
        Optional plotting and exports are excluded. No files are written here.
        """
        computation_started = time.perf_counter()
        self.computation_time_seconds = None
        self.elapsed_seconds = None
        self.background_elapsed_seconds = {}
        self.frame_elapsed_seconds_by_background = {}
        image1=normalize_image(image1)
        if not self.single_frame and image2 is not None:image2=normalize_image(image2)
        pairs=frame_pairs(image1,image2,self.single_frame,self.stack_axis,self.batch_processing)
        plot_index=current_result_index(len(pairs))
        self.parameters['plot_index']=plot_index
        self._analysis_pixel_size=self.pixel_size*(2 if self.single_frame else 1)
        self.metrics=[];self.rfrc_maps={};self.final_background_thresholds={}
        self._global_statistics=None
        self.global_statistics_by_background={}
        # Pixel-threshold display reference (first member of the FRC pair),
        # stored once per analysed frame; does not affect numerical results.
        self.background_reference_frames=[] if self.monitor_change else None
        for ratio in self.get_ratios():
            background_started = time.perf_counter()
            ratio_label=f'bg{ratio:.1f}' if self.adaptive_background or self.monitor_change else 'manual-bg'
            maps,thresholds,statistics_rows=[],[],[]
            frame_elapsed_seconds=[]
            for result_index,(first,second,label) in enumerate(pairs):
                frame_started = time.perf_counter()
                target=first.shape
                if self.single_frame:first,second=diagonal_resample(first)
                elif self.motion_correction:first,second=drift_correct(first,second,self.pixel_size)
                first=normalize_image(first);second=normalize_image(second)
                if self.monitor_change and len(self.background_reference_frames)==result_index:
                    self.background_reference_frames.append(np.asarray(first, dtype=np.float32).copy())
                data,background=(self.rfrc_parallel(first,second,ratio) if self.parallel else self.rfrc(first,second,ratio))
                # Freeze scientific statistics on the native grid before restoration.
                data=finalize_resolution_map(data,self.pixel_size)
                statistics=map_statistics(data)
                statistics_rows.append(statistics.copy())
                self.statistics_map_shape=data.shape
                data=restore_map_size(data,target)
                maps.append(data);thresholds.append(float(background))
                row=dict(**statistics,background_threshold=float(background),frame=label,
                         background_setting=ratio_label,singleframe=bool(self.single_frame),threshold=self.threshold)
                frame_elapsed_seconds.append(time.perf_counter()-frame_started)
                self.metrics.append(row)
            stack=np.stack(maps)
            self.rfrc_maps[ratio_label]=stack
            self._global_statistics=statistics_rows
            self.global_statistics_by_background[ratio_label]=statistics_rows
            self.final_background_thresholds[ratio_label]=thresholds
            self.rfrc_map=stack[0] if len(stack)==1 else stack
            self.background_elapsed_seconds[ratio_label] = time.perf_counter() - background_started
            self.frame_elapsed_seconds_by_background[ratio_label] = frame_elapsed_seconds
        self.calculate_global()
        self.computation_time_seconds = time.perf_counter() - computation_started
        self.elapsed_seconds = self.computation_time_seconds
        print(f'srFRC - numerical computation time: {self.computation_time_seconds:.3f} s '
              '(excludes plotting and file export).')
        for group_number, (setting, seconds) in enumerate(self.background_elapsed_seconds.items(), start=1):
            group_rows = [row for row in self.metrics if row['background_setting'] == setting]
            thresholds = ', '.join(
                f"{row['frame']}: Background threshold = {row['background_threshold']:.2f}"
                for row in group_rows
            )
            print(f'  Background group {group_number}: {thresholds}; computation time = {seconds:.3f} s')
        print(f'srFRC - completed {len(self.rfrc_maps)} background group(s).')
        if self.plot:
            figure=plot_srfrc_results(self,show=True)
            if figure is not None:
                plt.close(figure)
        return self

    def iterate(self,im1,im2):
        """Compute a map on an already prepared paired grid.
        
        Parameters
        ----------
        im1 : numpy.ndarray
            First normalized image (float32 or higher, maximum 1); matches im2.
        im2 : numpy.ndarray or None
            Second image or volume for paired-acquisition analysis.
        
        Returns
        -------
        numpy.ndarray or dict
            One finalized map, or all three named maps in monitoring mode.
        
        Notes
        -----
        This low-level method does not restore the original acquisition dimensions; use
        execute for the complete single-frame restoration and summary workflow.
        """
        computation_started = time.perf_counter()
        self.computation_time_seconds = None
        self.elapsed_seconds = None
        self.background_elapsed_seconds = {}
        self.frame_elapsed_seconds_by_background = {}
        self.rfrc_maps={}; self.metrics=[]; self.final_background_thresholds={}
        self.background_reference_frames=None
        self.global_statistics_by_background={}
        for ratio in self.get_ratios():
            background_started = time.perf_counter()
            data,level=(self.rfrc_parallel(im1,im2,ratio) if self.parallel else self.rfrc(im1,im2,ratio))
            self.rfrc_map=finalize_resolution_map(data,self.pixel_size)
            label=f'bg{ratio:.1f}' if self.adaptive_background or self.monitor_change else 'manual-bg'
            statistics=map_statistics(self.rfrc_map)
            self.rfrc_maps[label]=self.rfrc_map[None]
            self.global_statistics_by_background[label]=[statistics]
            self.final_background_thresholds[label]=[float(level)]
            self._global_statistics=[statistics]
            self.statistics_map_shape=self.rfrc_map.shape
            self.background_elapsed_seconds[label] = time.perf_counter() - background_started
            self.frame_elapsed_seconds_by_background[label] = [self.background_elapsed_seconds[label]]
            self.metrics.append(dict(frame='prepared-pair',background_setting=label,
                singleframe=bool(self.single_frame),threshold=self.threshold,
                **statistics,background_threshold=float(level)))
        self.computation_time_seconds = time.perf_counter() - computation_started
        self.elapsed_seconds = self.computation_time_seconds
        print(f'srFRC - numerical computation time: {self.computation_time_seconds:.3f} s '
              '(prepared pair; excludes plotting/export).')
        return self.get_rfrc_map()

    def get_ratios(self):
        """Choose the adaptive-background multiplier settings.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        list of float
            [0.8] for default adaptive background, [0.6,0.8,1.0] for monitoring, or
            [0.0] for manual background.
        """
        if self.monitor_change:return [0.6,0.8,1.0]
        if self.adaptive_background:return [0.8]
        return [0.0]

    def rfrc(self, im1, im2, ratio=0.0):
        """Calculate a rolling map serially on the prepared image pair.
        
        Parameters
        ----------
        im1 : numpy.ndarray
            First normalized image (float32 or higher, maximum 1); matches im2.
        im2 : numpy.ndarray or None
            Second image or volume for paired-acquisition analysis.
        ratio : float
            Multiplier of the Otsu background intensity threshold. {default: 0.0}
        
        Returns
        -------
        tuple
            (map, background_threshold): a map in nm on the paired grid and the final
            intensity threshold.
        
        Notes
        -----
        Use the selected primary criterion, retaining the calibrated 5-sigma then
        2-sigma fallbacks. Zero is used only if all criteria fail. Local curves
        are not extrapolated.
        This method does not perform original-size restoration.
        """
        if self.adaptive_background or self.monitor_change:
            # Keep reported/manual thresholds on the established 0-255 scale.
            threshold = 255 * ratio * filters.threshold_otsu(im1)
            print(f'srFRC - background threshold: {threshold:.2f} (intensity scale 0-255)')
        else:
            threshold = self.background_threshold
        h, w = im1.shape
        pad_width = w + self.block_size
        pad_height = h + self.block_size
        pad_im1 = create_padded_image(im1, pad_height, pad_width)
        pad_im2 = create_padded_image(im2, pad_height, pad_width)
        start_center = np.array((self.block_size / 2 - 1), dtype='int')
        rfrc_map = np.zeros((h, w), dtype='float32')
        size_padded = max(128, self.block_size)
        pad_factor = 1
        # Window geometry is constant for every rolling position.
        hm = (hanning_2d(size_padded, size_padded) if self.single_frame
              else hanning_2d(self.block_size, self.block_size))

        total_blocks = h * w / self.skip / self.skip
        progress_bar = tqdm(total=total_blocks, desc="rolling resolution calculation", unit="pixel")

        for y in range(0, h, self.skip):
            for x in range(0, w, self.skip):
                im1_roi = np.zeros((self.block_size, self.block_size), dtype='float32')
                im1_roi[:, :] = pad_im1[y:y + self.block_size, x:x + self.block_size]
                im2_roi = np.zeros((self.block_size, self.block_size), dtype='float32')
                im2_roi[:, :] = pad_im2[y:y + self.block_size, x:x + self.block_size]
                summing = np.mean(im1_roi[start_center:start_center + self.skip, start_center:start_center + self.skip])
                summing += np.mean(
                    im2_roi[start_center:start_center + self.skip, start_center:start_center + self.skip])
                summing /= 2
                if 255 * summing > threshold:
                    if np.max(im1_roi)>0:im1_roi = im1_roi / np.max(im1_roi)
                    if np.max(im2_roi)>0:im2_roi = im2_roi / np.max(im2_roi)
                    if self.single_frame:
                        im1_roi = create_padded_image(im1_roi, size_padded, size_padded)
                        im2_roi = create_padded_image(im2_roi, size_padded, size_padded)
                        pad_factor = math.sqrt(size_padded / self.block_size)
                    ffta = ft.fftshift(ft.fft2(im1_roi * hm))
                    fftb = ft.fftshift(ft.fft2(im2_roi * hm))
                    frc_data = perform_frc(ffta, fftb, self._analysis_pixel_size, single_frame=self.single_frame)
                    resolution = self._local_resolution(frc_data,pad_factor)
                    rfrc_map[y:y + self.skip, x:x + self.skip] = resolution
                progress_bar.update(1)

        progress_bar.close()
        return rfrc_map, threshold

    def rfrc_parallel(self, im1, im2, ratio=0.0):
        """Calculate a rolling map using multiprocessing patches.
        
        Parameters
        ----------
        im1 : numpy.ndarray
            First normalized image (float32 or higher, maximum 1); matches im2.
        im2 : numpy.ndarray or None
            Second image or volume for paired-acquisition analysis.
        ratio : float
            Multiplier of the Otsu background intensity threshold. {default: 0.0}
        
        Returns
        -------
        tuple
            (map, background_threshold), matching the serial method on the prepared
            grid.
        
        Notes
        -----
        Patch merging includes image boundaries. Original-size restoration and final
        statistics are performed later by execute.
        """
        if self.adaptive_background or self.monitor_change:
            # Keep reported/manual thresholds on the established 0-255 scale.
            threshold = 255 * ratio * filters.threshold_otsu(im1)
            print(f'srFRC - background threshold: {threshold:.2f} (intensity scale 0-255)')
        else:
            threshold = self.background_threshold

        h, w = im1.shape
        patch_size = self.patch_size

        pad_height = h + self.block_size
        pad_width = w + self.block_size
        pad_im1 = create_padded_image(im1, pad_height, pad_width)
        pad_im2 = create_padded_image(im2, pad_height, pad_width)

        size_padded = max(128, self.block_size)
        hm = hanning_2d(size_padded, size_padded) if self.single_frame else hanning_2d(self.block_size, self.block_size)

        # Create a patch list that covers every pixel.
        patches = self.create_patches(h, w, patch_size)
        total_patches = len(patches)
        print(f"Processing {total_patches} patches...")

        rfrc_map = np.zeros((h, w), dtype=np.float32)
        # Copy each complete input into shared memory ONCE per map. With a spawn
        # start method, each worker attaches to the same buffers; only small
        # four-integer bounds tuples are sent to each patch task.
        shared_a, dtype_a = _shared_copy(pad_im1)
        shared_b, dtype_b = _shared_copy(pad_im2)
        if dtype_a != dtype_b:
            raise ValueError('Paired padded images must have matching dtypes')
        process_count = min(mp.cpu_count(), max(1, total_patches))
        with mp.Pool(processes=process_count, initializer=_init_rolling_worker,
                     initargs=(self, shared_a, shared_b, pad_im1.shape,
                               dtype_a, threshold, hm)) as pool:
            with tqdm(total=total_patches, desc="patch processing", unit="patch") as pbar:
                for patch_result in pool.imap_unordered(_run_rolling_patch, patches):
                    if patch_result is not None:
                        y_start, x_start, patch_map = patch_result
                        height, width = patch_map.shape
                        rfrc_map[y_start:y_start + height,
                                 x_start:x_start + width] = patch_map
                    pbar.update(1)
        print("srFRC - parallel map calculation complete.")
        return rfrc_map, threshold

    def create_patches(self, height, width, patch_size):
        """Partition an image into patches with complete boundary coverage.
        
        Parameters
        ----------
        height : int
            Output height, in pixels.
        width : int
            Output width, in pixels.
        patch_size : int
            Side length of a spatial patch assigned to a worker, in pixels.
        
        Returns
        -------
        list of tuple
            Patch bounds (y_start,y_end,x_start,x_end), including partial boundary
            patches.
        """
        # Keep every worker on the global rolling-window lattice.
        patch_size = max(self.skip, int(math.ceil(patch_size / self.skip)) * self.skip)
        patches = []
        y_patches = int((height + patch_size - 1) / patch_size)
        x_patches = int((width + patch_size - 1) / patch_size)
        for y in range(y_patches):
            for x in range(x_patches):
                y_start = y * patch_size
                x_start = x * patch_size
                y_end = min(height, y_start + patch_size)
                x_end = min(width, x_start + patch_size)
                patches.append((y_start, y_end, x_start, x_end))
        return patches

    def _process_patch(self, patch_info, pad_im1, pad_im2, threshold,
                       block_size, skip, single_frame, pixel_size, hm):
        """Calculate the rolling resolution values within one worker patch.
        
        Parameters
        ----------
        patch_info : tuple of int
            Patch bounds (y_start, y_end, x_start, x_end), with exclusive end indices.
        pad_im1 : numpy.ndarray
            Symmetrically padded first image.
        pad_im2 : numpy.ndarray
            Symmetrically padded second image.
        threshold : float
            Final background intensity threshold on the normalized 0-255 scale.
        block_size : int
            Side length of each rolling correlation window in pixels.
        skip : int
            Positive spatial step between rolling windows, in pixels.
        single_frame : bool
            Generate a diagonal pair from each input image or frame; False uses paired
            acquisitions.
        pixel_size : float
            Positive calibrated pixel spacing in nm.
        hm : numpy.ndarray
            Hanning window matching the Fourier-transform block shape.
        
        Returns
        -------
        tuple
            (y_start,x_start,patch_map), where patch_map contains local resolution
            values in nm.
        """
        y_start, y_end, x_start, x_end = patch_info
        patch_height = y_end - y_start
        patch_width = x_end - x_start
        patch_im1 = pad_im1[y_start:y_end + block_size, x_start:x_end + block_size]
        patch_im2 = pad_im2[y_start:y_end + block_size, x_start:x_end + block_size]
        patch_map = np.zeros((patch_height, patch_width))
        size_padded = max(128, block_size)

        for local_y in range(0, patch_height, skip):
            for local_x in range(0, patch_width, skip):
                im1_block = patch_im1[local_y:local_y + block_size, local_x:local_x + block_size]
                im2_block = patch_im2[local_y:local_y + block_size, local_x:local_x + block_size]

                # Compute the mean intensity in the central region, as in the serial version.
                start_center = np.array((block_size // 2 - 1), dtype='int')
                center_block1 = im1_block[start_center:start_center + skip, start_center:start_center + skip]
                center_mean1 = np.mean(center_block1)
                center_block2 = im2_block[start_center:start_center + skip, start_center:start_center + skip]
                center_mean2 = np.mean(center_block2)
                summing = (center_mean1 + center_mean2) / 2

                if 255 * summing <= threshold:
                    continue

                im1_block_norm = im1_block.copy()
                im2_block_norm = im2_block.copy()

                im1_max = np.max(im1_block_norm)
                im2_max = np.max(im2_block_norm)
                if im1_max > 0:
                    im1_block_norm /= im1_max
                if im2_max > 0:
                    im2_block_norm /= im2_max

                pad_factor = 1
                if single_frame:
                    im1_block_norm = create_padded_image(im1_block_norm, size_padded, size_padded)
                    im2_block_norm = create_padded_image(im2_block_norm, size_padded, size_padded)
                    pad_factor = math.sqrt(size_padded / block_size)

                ffta = fft_backend.fftshift(fft_backend.fft2(im1_block_norm * hm))
                fftb = fft_backend.fftshift(fft_backend.fft2(im2_block_norm * hm))
                frc_data = perform_frc(ffta, fftb, pixel_size, single_frame=single_frame)
                resolution = self._local_resolution(frc_data,pad_factor)
                patch_map[local_y:local_y + skip, local_x: local_x + skip] = resolution
        return y_start, x_start, patch_map

    def _local_resolution(self, curve, pad_factor):
        """Evaluate the chosen primary criterion and preserve sigma fallbacks.

        Parameters
        ----------
        curve : FourierCorrelationData
            Measured rolling-window correlation; local extrapolation is disabled.
        pad_factor : float
            Existing sigma-threshold calibration for padded single-frame windows.

        Returns
        -------
        float
            Primary estimate in nm, otherwise calibrated 5-sigma then 2-sigma.
            get_resolution applies the existing 0.6 and 1.5 factors. Zero is
            returned only if no criterion provides a finite positive estimate.
        """
        curve.calculate_resolution(additional_sigma=True,pad_factor=pad_factor,
                                   threshold=self.threshold)
        for criterion in (self.threshold,'5-sigma','2-sigma'):
            resolution=curve.get_resolution(threshold=criterion)
            if np.isfinite(resolution) and resolution>0:
                return resolution
        return 0.0

