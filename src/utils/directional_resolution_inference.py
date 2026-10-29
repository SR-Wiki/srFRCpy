import math
import warnings
import numpy as np
from src.utils.angle_mask import AngleMask
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.correlations import perform_fsc, FSCWorkspace
from src.utils.sfsc_guidance import validate_num_cones, fit_range_advice
from src.utils.analysis_results import DRI_GUIDANCE,dri_interpretation


class DirectionalResolutionInference:
    MIN_VALID_DIRECTIONS = 1
    GOF_WARNING_THRESHOLD = 0.10

    def __init__(self, pixel_size, xy_resolution, single_frame=False, z_sampling=False,
                 num_cones=5, low_angle=15, high_angle=60,
                 threshold='fixed_threshold', plot_fit=False, extrapolation=True):
        """Configure a directional-resolution fit with a fixed lateral cutoff.
        
        Parameters
        ----------
        pixel_size : float
            Positive calibrated pixel spacing in nm.
        xy_resolution : float
            Positive lateral resolution estimate in nm.
        single_frame : bool
            Use the correlation correction for samples derived from one acquisition.
            {default: False}
        z_sampling : bool
            Whether odd/even Z sampling and its associated correction are enabled.
            {default: False}
        num_cones : int
            Positive number of directional cones. At least 5 is recommended, but smaller
            positive counts are accepted. {default: 5}
        low_angle : float
            Lower DRI fitting angle in degrees; values above 10 degrees are recommended.
            {default: 15}
        high_angle : float
            Upper DRI fitting angle in degrees; values below 80 degrees are recommended.
            {default: 60}
        threshold : str
            Correlation threshold: fixed_threshold represents 1/7; 3-sigma is also
            supported. {default: 'fixed_threshold'}
        plot_fit : bool
            Stored compatibility flag for fit visualization. {default: False}
        extrapolation : bool
            Retained compatibility flag; fitting follows the automatic
            all-directions-missing policy. {default: True}
        
        Returns
        -------
        None
            Initializes the lateral/axial model and directional sampling settings.
        
        Notes
        -----
        Input fitting angles are in degrees and are stored internally in radians. E_DRI
        is stored as a relative fraction. This is an internal fitting object; public
        sFSC uses automatic extrapolation policy.
        """
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError('pixel_size must be a positive finite value.')
        if not np.isfinite(xy_resolution) or xy_resolution <= 0:
            raise ValueError('xy_resolution must be a positive finite value.')
        if not (0 <= low_angle < high_angle <= 90):
            raise ValueError('Angles must satisfy 0 <= low_angle < high_angle <= 90 degrees.')
        num_cones = validate_num_cones(num_cones)
        if threshold not in ('fixed_threshold', '3-sigma'):
            raise ValueError("threshold must be 'fixed_threshold' or '3-sigma'.")

        self.pixel_size = pixel_size
        self.xy_resolution = xy_resolution
        self.xy_cutoff = 1000 / xy_resolution
        self.single_frame = single_frame
        self.z_sampling = z_sampling
        self.low_angle = math.radians(low_angle)
        self.high_angle = math.radians(high_angle)
        self.num_cones = num_cones
        self.threshold = threshold
        self.plot_fit = plot_fit
        self.extrapolation = bool(extrapolation)
        self.z_resolution = None
        self.z_cutoff = None
        self.cutoffs = None
        self.angles = None
        self.resolutions = None
        self.goodness_of_fit = np.nan
        self.extrapolation_attempted = False
        self.extrapolation_used_count = 0
        self.measured_crossing_count = 0
        self.sectors = []

    def get_xy_resolution(self):
        """Read the lateral resolution and cutoff.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        tuple of float
            (lateral_resolution_nm,lateral_cutoff_per_um).
        """
        return self.xy_resolution, self.xy_cutoff

    def get_z_resolution(self):
        """Read the axial resolution and cutoff.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        tuple
            (axial_resolution_nm,axial_cutoff_per_um); values are unavailable before a
            usable fit.
        """
        return self.z_resolution, self.z_cutoff

    def get_resolutions(self):
        """Convert valid directional cutoffs to resolution values.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        numpy.ndarray or None
            Directional resolutions in nm, or None before cutoffs are available.
        """
        if self.cutoffs is None:
            return None
        return 1000 / self.cutoffs

    def get_cutoffs(self):
        """Read the valid directional frequency cutoffs.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        numpy.ndarray or None
            Cutoffs in 1/um, or None before fitting.
        """
        return self.cutoffs

    def get_angles(self):
        """Read the valid directional sampling angles.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        numpy.ndarray or None
            Angles in radians, or None before fitting.
        """
        return self.angles

    def get_angle_range(self):
        """Read the stored DRI angle range and cone count.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        tuple
            (low_angle_rad,high_angle_rad,num_cones).
        """
        return self.low_angle, self.high_angle, self.num_cones

    def fit(self, ffta, fftb):
        """Fit axial resolution to measured directional shell-correlation crossings.
        
        Parameters
        ----------
        ffta : numpy.ndarray
            Centered complex Fourier transform of the first input.
        fftb : numpy.ndarray
            Centered complex Fourier transform of the second input.
        
        Returns
        -------
        tuple of float
            (axial_resolution_nm,axial_cutoff_per_um), with NaN values when no usable
            crossings remain.
        
        Notes
        -----
        All directions are first evaluated without extrapolation. Only if every
        direction lacks a measured crossing are the directional curves retried with
        automatic 1/7 extrapolation. One or more measured crossings use the ordinary fit
        path. The fit stores goodness_of_fit (E_DRI fraction), valid angles, cutoffs,
        and extrapolation state. E_DRI <10% is reliable, 10-20% requires caution, and
        >20% is unacceptable; high errors prompt fitting-angle advice.
        """
        self.sectors=[]
        self.extrapolation_attempted=False
        self.extrapolation_used_count=0
        self.goodness_of_fit=np.nan
        self.z_resolution,self.z_cutoff=np.nan,np.nan
        a = self.xy_cutoff
        angle_mask = AngleMask(ffta)
        workspace = FSCWorkspace(ffta, fftb)
        if self.num_cones == 1:
            angles = np.array([(self.low_angle + self.high_angle) / 2])
            angle_step = self.high_angle - self.low_angle
        else:
            angles = np.linspace(self.low_angle, self.high_angle, self.num_cones)
            angle_step = angles[1] - angles[0]
        cutoffs = []
        used_extrapolation = 0
        required_extrapolation = 0

        progress_bar = tqdm(total=self.num_cones, desc='directional resolution inference', unit='cone')
        for angle in angles:
            lower, upper = angle - angle_step / 2, angle + angle_step / 2
            mask = angle_mask.get_mask(lower, upper)
            temp_fsc = perform_fsc(
                ffta, fftb, self.pixel_size,
                single_frame=self.single_frame,
                z_sampling=self.z_sampling,
                extrapolation=False, mask=mask, workspace=workspace,
            )
            temp_fsc.calculate_resolution(threshold=self.threshold)
            self.sectors.append(temp_fsc)
            resolution = temp_fsc.get_resolution(threshold=self.threshold)
            if temp_fsc.needs_extrapolation():
                required_extrapolation += 1
            if temp_fsc.extrapolated():
                used_extrapolation += 1
            cutoffs.append(1000 / resolution if np.isfinite(resolution) and resolution > 0 else np.nan)
            progress_bar.update(1)
        progress_bar.close()

        self.measured_crossing_count = sum(np.isfinite(c) and c>0 for c in cutoffs)
        if self.measured_crossing_count == 0:
            for i,sector in enumerate(self.sectors):
                sector.allow_extrapolation = True
                sector.calculate_resolution(threshold=self.threshold)
                self.extrapolation_attempted |= sector.extrapolation_attempted
                self.extrapolation_used_count += int(sector.extrapolated())
                resolution=sector.get_resolution(self.threshold)
                cutoffs[i]=1000/resolution if np.isfinite(resolution) and resolution>0 else np.nan
        used_extrapolation=self.extrapolation_used_count

        cutoffs = np.asarray(cutoffs, dtype=float)
        valid = np.isfinite(cutoffs) & (cutoffs > 0)
        self.cutoffs = cutoffs[valid]
        self.angles = angles[valid]

        if len(self.cutoffs)==0:
            detail=('automatic 1/7 extrapolation failed' if self.threshold=='fixed_threshold'
                    else 'automatic extrapolation supports 1/7 only')
            warnings.warn(f'sFSC - DRI: No usable threshold crossing was found in any direction; {detail}. Axial resolution is unavailable (NaN). Check the voxel calibration, threshold, and fitting angles.',RuntimeWarning,stacklevel=2)
            self.z_resolution,self.z_cutoff=np.nan,np.nan
            return self.get_z_resolution()
        if used_extrapolation:
            print(f'sFSC - axial: Automatic 1/7 extrapolation provided {used_extrapolation} directional crossing(s), after all measured directions had no crossing.')

        x = self.cutoffs * np.cos(self.angles)
        y = self.cutoffs * np.sin(self.angles)
        weight = x / (np.sqrt(x * x + y * y) + 1e-12)
        top = weight * y * y * (1 - x * x / a / a)
        bot = weight * y ** 4
        denominator = bot.sum()
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError('The directional samples could not produce a stable elliptical fit.')

        k = top.sum() / denominator
        if not np.isfinite(k) or k <= 0:
            raise RuntimeError('The fitted axial cutoff is not physically valid.')

        self.z_resolution = 1000 * np.sqrt(k)
        self.z_cutoff = 1000 / self.z_resolution
        self.resolutions = 1000 / self.cutoffs

        self.goodness_of_fit = self.dri_goodness_of_fit(
            self.cutoffs, self.angles, a, self.z_cutoff)
        print(f'E_DRI = {100*self.goodness_of_fit:.2f}% ({dri_interpretation(self.goodness_of_fit)}); {DRI_GUIDANCE}')
        if not np.isfinite(self.goodness_of_fit) or self.goodness_of_fit>=0.10:
            warnings.warn(f'sFSC - DRI: E_DRI={100*self.goodness_of_fit:.2f}% '
                          f'({dri_interpretation(self.goodness_of_fit)}). {DRI_GUIDANCE} '
                          'An E_DRI above 20% means the axial estimate should not be accepted. '+fit_range_advice(self.goodness_of_fit),RuntimeWarning,stacklevel=2)

        return self.get_z_resolution()

    @staticmethod
    def dri_goodness_of_fit(k_measured, theta, k_xy, k_z):
        """Compute the cosine-weighted relative RMS error of the DRI ellipse.
        
        Parameters
        ----------
        k_measured : numpy.ndarray
            Measured directional cutoffs in 1/um.
        theta : array_like
            Angular coordinates in radians.
        k_xy : float
            Positive lateral frequency cutoff in 1/um.
        k_z : float
            Positive axial frequency cutoff in 1/um.
        
        Returns
        -------
        float
            Relative E_DRI as a fraction; multiply by 100 for percent.
        """
        k_measured = np.asarray(k_measured, dtype=float)
        theta = np.asarray(theta, dtype=float)
        if k_measured.shape != theta.shape:
            raise ValueError('k_measured and theta must have the same shape.')
        if k_xy <= 0 or k_z <= 0:
            raise ValueError('k_xy and k_z must be positive.')
        k_fit = 1.0 / np.sqrt(
            np.cos(theta) ** 2 / k_xy ** 2 +
            np.sin(theta) ** 2 / k_z ** 2
        )
        weights = np.cos(theta)
        relative_residual = (k_measured - k_fit) / k_fit
        return np.sqrt(np.sum(weights * relative_residual ** 2) / np.sum(weights))
