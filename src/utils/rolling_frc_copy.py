import os

import tifffile

from src.utils.drift_correct import drift_correct
from src.utils.image_operations import *
from skimage import filters
from src.utils.window_functions import hanning_2d
import numpy.fft as ft
from src.utils.correlations import perform_frc
import matplotlib.pyplot as plt
from src.utils.shifted_jet import sjet_colorbar
import matplotlib.colors as mcolors
from src.utils.otsu import otsu
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import multiprocessing as mp


class RollingFourierRingCorrelation:
    def __init__(self, pixel_size, adaptative_background=True, single_frame=True,
                 block_size=64, monitor_change=True, background_threshold=30,
                 skip=2, parallel=True, patch_size=128):
        """Configure the retained historical rolling-FRC implementation.
        
        Parameters
        ----------
        pixel_size : float
            Positive calibrated pixel spacing in nm.
        adaptative_background : bool
            Legacy spelling of the adaptive-background flag. {default: True}
        single_frame : bool
            Use the correlation correction for samples derived from one acquisition.
            {default: True}
        block_size : int
            Side length of each rolling correlation window in pixels. {default: 64}
        monitor_change : bool
            Evaluate multiple background multipliers when supported by this
            implementation. {default: True}
        background_threshold : float
            Manual background intensity threshold on the normalized 0-255 scale.
            {default: 30}
        skip : int
            Positive spatial step between rolling windows, in pixels. {default: 2}
        parallel : bool
            Use multiprocessing for rolling-map patches. {default: True}
        patch_size : int
            Side length of a spatial patch assigned to a worker, in pixels.
            {default: 128}
        
        Returns
        -------
        None
            Initializes the analyser; call execute with image arrays to compute maps.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        self.pixel_size = pixel_size
        self.adaptative_background = adaptative_background
        self.skip = skip
        self.single_frame = single_frame
        self.drift_correct = not self.single_frame
        self.block_size = block_size
        self.monitor_change = monitor_change
        self.background_threshold = background_threshold
        self.parallel = parallel
        self.patch_size = patch_size
        self.rfrc_map = None
        self.mean_res = None
        self.min_res = None
        self.max_res = None
        self.rfrc_value = None
        self.panels = None

    def get_rfrc_map(self):
        """Return the latest completed resolution map.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        numpy.ndarray
            Latest legacy paired-grid resolution map in nm.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        assert self.rfrc_map is not None, 'You have not done the rolling calculation!'
        return self.rfrc_map

    def get_panels(self):
        """Build dimensionless quality panels from completed resolution maps.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        numpy.ndarray
            Quality scores in [0,1], with the same image/stack shape as rfrc_map.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        self.panels = otsu(self.rfrc_map)
        return self.panels

    def calculate_global(self):
        """Calculate positive-foreground statistics from the final resolution maps.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        numpy.ndarray
            Legacy [minimum_nm,maximum_nm,mean_nm,rFRC_value] array when the historical
            routine completes.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        indices = self.rfrc_map > 0
        valid_components = self.rfrc_map[indices]
        self.max_res = valid_components.max()
        self.min_res = valid_components.min()
        self.mean_res = valid_components.mean()
        num_non_zeros = indices.sum()
        self.rfrc_value = valid_components.sum() / (num_non_zeros * self.min_res) - 1
        print('Global metrics:')
        print(f'rFRC value: {self.rfrc:.3f}')
        print(f'mean resolution: {self.mean_res:.2f} nm')
        print(f'minimum resolution: {self.min_res:.2f} nm')
        print(f'maximum resolution: {self.max_res:.2f} nm')
        absolute_value = np.array((self.min_res, self.max_res, self.mean_res, self.rfrc_value), dtype='float32')
        return absolute_value

    def execute(self, image1, image2=None):
        """Run the historical rolling calculation for one prepared image pair.
        
        Parameters
        ----------
        image1 : numpy.ndarray
            First image or stack of images.
        image2 : numpy.ndarray or None
            Second image or stack, when paired input is supplied separately.
            {default: None}
        
        Returns
        -------
        None
            Updates the legacy analyser and performs the historical plot/image exports.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API. This historical
        implementation writes PNG/TIFF files; its behavior is not changed by
        documentation updates.
        """
        print('PANEL estimation start...')
        start_time = datetime.datetime.now()
        print('High-resolution rFRC estimation...')
        if self.single_frame:
            im1, im2 = diagonal_resample(image1)
            im1 = 255 * im1 / np.max(im1)
            im2 = 255 * im2 / np.max(im2)
            self.pixel_size *= 2.0
        else:
            assert image2 is not None, 'There should be 2 input images if single frame mode is off!'
            im1, im2 = drift_correct(image1, image2, self.pixel_size)
        self.iterate(im1, im2)
        print('PANEL estimation done, thank you for your waiting')
        elapsed_time = datetime.datetime.now() - start_time
        print('Total time cost: %s' % elapsed_time)

    def iterate(self, im1, im2):
        """Run the historical rolling calculation for one prepared image pair.
        
        Parameters
        ----------
        im1 : numpy.ndarray
            First image or volume; its shape must match the paired input.
        im2 : numpy.ndarray or None
            Second image or volume for paired-acquisition analysis.
        
        Returns
        -------
        None
            Updates the legacy analyser and performs the historical plot/image exports.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API. This historical
        implementation writes PNG/TIFF files; its behavior is not changed by
        documentation updates.
        """
        sjet_map = sjet_colorbar()
        ratios = self.get_ratios()
        num_ratios = len(ratios)
        if num_ratios > 1:
            fig, axes = plt.subplots(1, num_ratios, figsize=(16, 4), constrained_layout=True)
            j = 0
            for ratio in ratios:
                if self.parallel:
                    rfrc_map, threshold = self.rfrc_parallel(im1, im2, ratio)
                else:
                    rfrc_map, threshold = self.rfrc(im1, im2, ratio)
                self.rfrc_map = rfrc_map
                norm = mcolors.Normalize(vmin=0., vmax=rfrc_map.max())
                pc_kwargs = {'rasterized': True, 'cmap': sjet_map, 'norm': norm}
                ax = axes[j]
                im = ax.pcolormesh(rfrc_map[::-1], **pc_kwargs)
                title = f'Background Threshold = {threshold}'
                ax.set_title(title)
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                j += 1
            fig.colorbar(im, ax=axes)
            plt.show()
            plt.savefig('./monitor_change.png')
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            if self.parallel:
                rfrc_map, threshold = self.rfrc_parallel(im1, im2, ratios[0])
            else:
                rfrc_map, threshold = self.rfrc(im1, im2, ratios[0])
            self.rfrc_map = rfrc_map
            norm = mcolors.Normalize(vmin=0., vmax=rfrc_map.max())
            pc_kwargs = {'rasterized': True, 'cmap': sjet_map, 'norm': norm}
            im = ax.pcolormesh(rfrc_map[::-1], **pc_kwargs)
            title = f'Background Threshold = {threshold}'
            ax.set_title(title)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            fig.colorbar(im, ax=ax)
            plt.savefig('./rFRCMap.png')
            plt.close()
            tifffile.imwrite('./rFRCMap.tif', rfrc_map)

    def get_ratios(self):
        """Choose the adaptive-background multiplier settings.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        list of float
            [0.8] when adaptative_background is True; otherwise [0.6,0.8,1.0] for
            monitor_change, or [0] for manual background.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        if self.adaptative_background:
            return [0.8]
        elif self.monitor_change:
            return [0.6, 0.8, 1.0]
        else:
            return [0]

    def rfrc(self, im1, im2, ratio=0.0):
        """Calculate a rolling map serially on the prepared image pair.
        
        Parameters
        ----------
        im1 : numpy.ndarray
            First image or volume; its shape must match the paired input.
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
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        if self.adaptative_background or self.monitor_change:
            threshold = ratio * filters.threshold_otsu(im1)
            print(f'Adaptative Background = {threshold:.2f}')
        else:
            threshold = self.background_threshold
        h, w = im1.shape
        pad_width = w + self.block_size
        pad_height = h + self.block_size
        pad_im1 = create_padded_image(im1, pad_width, pad_height)
        pad_im2 = create_padded_image(im2, pad_width, pad_height)
        start_center = np.array((self.block_size / 2 - 1), dtype='int')
        rfrc_map = np.zeros((h, w), dtype='float32')
        size_padded = 128
        pad_factor = 1

        total_blocks = h * w / self.skip / self.skip
        progress_bar = tqdm(total=total_blocks, desc="rolling resolution calculation", unit="pixel")

        for y in range(0, w, self.skip):
            for x in range(0, h, self.skip):
                im1_roi = np.zeros((self.block_size, self.block_size), dtype='float32')
                im1_roi[:, :] = pad_im1[y:y + self.block_size, x:x + self.block_size]
                im2_roi = np.zeros((self.block_size, self.block_size), dtype='float32')
                im2_roi[:, :] = pad_im2[y:y + self.block_size, x:x + self.block_size]
                summing = np.mean(im1_roi[start_center:start_center + self.skip, start_center:start_center + self.skip])
                summing += np.mean(
                    im2_roi[start_center:start_center + self.skip, start_center:start_center + self.skip])
                summing /= 2
                if summing > threshold:
                    im1_roi = im1_roi / np.max(im1_roi)
                    im2_roi = im2_roi / np.max(im2_roi)
                    if self.single_frame:
                        im1_roi = create_padded_image(im1_roi, size_padded, size_padded)
                        im2_roi = create_padded_image(im2_roi, size_padded, size_padded)
                        pad_factor = math.sqrt(size_padded / self.block_size)
                        hm = hanning_2d(size_padded, size_padded)
                    else:
                        hm = hanning_2d(self.block_size, self.block_size)
                    ffta = ft.fftshift(ft.fft2(im1_roi * hm))
                    fftb = ft.fftshift(ft.fft2(im2_roi * hm))
                    frc_data = perform_frc(ffta, fftb, self.pixel_size, single_frame=self.single_frame)
                    frc_data.calculate_resolution(additional_sigma=True, pad_factor=pad_factor)
                    resolution = frc_data.get_resolution(threshold='3-sigma')
                    if np.isnan(resolution):
                        resolution = 0.6 * frc_data.get_resolution(threshold='5-sigma')
                        if np.isnan(resolution):
                            resolution = 1.5 * frc_data.get_resolution(threshold='2-sigma')
                            if np.isnan(resolution):
                                resolution = 0
                    if 0.0 < resolution < self.pixel_size * 2.3:
                        resolution = self.pixel_size * 2.3
                    rfrc_map[y:y + self.skip, x:x + self.skip] = resolution
                progress_bar.update(1)

        return rfrc_map, threshold

    def rfrc_parallel(self, im1, im2, ratio=0.0):
        """Calculate a rolling map using multiprocessing patches.
        
        Parameters
        ----------
        im1 : numpy.ndarray
            First image or volume; its shape must match the paired input.
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
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        if self.adaptative_background:
            threshold = ratio * filters.threshold_otsu(im1)
            print(f'Adaptative Background = {threshold:.2f}')
        else:
            threshold = self.background_threshold

        h, w = im1.shape
        patch_size = self.patch_size

        pad_height = h + self.block_size
        pad_width = w + self.block_size
        pad_im1 = create_padded_image(im1, pad_height, pad_width)
        pad_im2 = create_padded_image(im2, pad_height, pad_width)

        size_padded = 128
        hm = hanning_2d(size_padded, size_padded) if self.single_frame else hanning_2d(self.block_size, self.block_size)

        # Create a patch list that covers every pixel.
        patches = self.create_patches(h, w, patch_size)
        total_patches = len(patches)
        print(f"Processing {total_patches} patches...")

        rfrc_map = np.zeros((h, w), dtype=np.float32)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Bind the shared arguments to the processing function.
            process_patch = partial(
                self._process_patch,
                pad_im1=pad_im1,
                pad_im2=pad_im2,
                threshold=threshold,
                block_size=self.block_size,
                skip=self.skip,
                single_frame=self.single_frame,
                pixel_size=self.pixel_size,
                hm=hm,
            )

            # Submit all patch tasks.
            results = []
            with tqdm(total=total_patches, desc="patch processing", unit="patch") as pbar:
                for result in pool.imap_unordered(process_patch, patches):
                    results.append(result)
                    pbar.update(1)

        # Merge patch results, including pixels at the boundaries.
        for patch_result in results:
            if patch_result is not None:
                y_start, x_start, patch_map = patch_result
                height, width = patch_map.shape
                rfrc_map[y_start:y_start + width, x_start:x_start + height] = patch_map
        print("Parallel Processing Completed.!")
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
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
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
            Use the correlation correction for samples derived from one acquisition.
        pixel_size : float
            Positive calibrated pixel spacing in nm.
        hm : numpy.ndarray
            Hanning window matching the Fourier-transform block shape.
        
        Returns
        -------
        tuple
            (y_start,x_start,patch_map), where patch_map contains local resolution
            values in nm.
        
        Notes
        -----
        Historical copy retained for compatibility. Use
        src.srFRC.RollingFourierRingCorrelation for the maintained API.
        """
        y_start, y_end, x_start, x_end = patch_info
        patch_height = y_end - y_start
        patch_width = x_end - x_start
        patch_im1 = pad_im1[y_start:y_end + block_size, x_start:x_end + block_size]
        patch_im2 = pad_im2[y_start:y_end + block_size, x_start:x_end + block_size]
        patch_map = np.zeros((patch_height, patch_width))
        size_padded = 128

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

                if summing <= threshold:
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

                ffta = np.fft.fftshift(np.fft.fft2(im1_block_norm * hm))
                fftb = np.fft.fftshift(np.fft.fft2(im2_block_norm * hm))
                frc_data = perform_frc(ffta, fftb, pixel_size, single_frame=single_frame)
                frc_data.calculate_resolution(additional_sigma=True, pad_factor=pad_factor)
                resolution = frc_data.get_resolution(threshold='3-sigma')
                if np.isnan(resolution):
                    resolution = 0.6 * frc_data.get_resolution(threshold='5-sigma')
                    if np.isnan(resolution):
                        resolution = 1.5 * frc_data.get_resolution(threshold='2-sigma')
                    if np.isnan(resolution):
                        resolution = 0
                if 0.0 < resolution < pixel_size * 2.3:
                    resolution = pixel_size * 2.3
                patch_map[local_y:local_y + skip, local_x: local_x + skip] = resolution
        return y_start, x_start, patch_map

