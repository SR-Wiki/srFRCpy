"""Unified, visualization-only plotting API for sFRC, sFSC/DRI, and srFRC.

The numerical analysers keep their old plotting entry points for backwards
compatibility; all maintained main-results plotting implementations live here.
"""
import matplotlib.pyplot as plt
import numpy as np
from src.utils.shifted_jet import sjet_colorbar


class ResolutionPlotter:
    """Stateless renderer for FRC curves, DRI profiles, and srFRC resolution maps."""

    @staticmethod
    def frc_curve(curve, threshold='fixed_threshold', save_path=None, title='sFRC - correlation curve', show=False):
        """Plot the measured curve, selected threshold and any genuine extrapolation.

        When an extrapolated crossing was used, distinguish measured samples,
        the model fit on that measured interval, and the predicted continuation.
        The numerical curve, cutoff, and fitting parameters are not modified.
        """
        resolution = curve.get_resolution(threshold)
        measured = curve.smoothed_frc_corrected if curve.single_frame else curve.smoothed_frc
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(curve.frequency, measured, color='#E67373', linewidth=1.7,
                label='Measured FRC')
        ax.plot(curve.frequency,
                curve.fixed_threshold if threshold == 'fixed_threshold' else curve.three_sigma,
                'k--', linewidth=1, label='1/7 threshold' if threshold == 'fixed_threshold' else threshold)

        if threshold == 'fixed_threshold' and curve.extrapolation_used \
                and curve.extrapolated_frc is not None and curve.new_frequency is not None:
            n = len(curve.frequency)
            fit_info = curve.extrapolation_fit_info or {}
            if curve.frequency[-1] > 0 and all(key in fit_info for key in ('A', 'k')):
                x = curve.frequency / curve.frequency[-1]
                q = fit_info['A'] * np.exp(-fit_info['k'] * x * x)
                fitted = q / (1 + q)
                ax.plot(curve.frequency, fitted, color='#4D80B3', linestyle='-.',
                        linewidth=1.5, label='Fitted model (measured range)')
            # Include the last measured point so the dashed prediction connects naturally.
            ax.plot(curve.new_frequency[n-1:], curve.extrapolated_frc[n-1:],
                    color='#4D80B3', linestyle='--', linewidth=1.9,
                    label='Extrapolated prediction (not measured)')
            ax.axvline(curve.frequency[-1], color='gray', linewidth=1,
                       linestyle=':', label='End of measured range')
            index = curve.get_index(threshold)
            cutoff_frequency = curve.new_frequency
            cutoff_curve = curve.extrapolated_frc
        else:
            index = curve.get_index(threshold)
            cutoff_frequency = curve.frequency
            cutoff_curve = measured

        if 0 <= index < len(cutoff_frequency) and np.isfinite(resolution):
            ax.scatter([cutoff_frequency[index]], [cutoff_curve[index]],
                       color='#345F92', s=35, zorder=5, label='Resolution cutoff')
            ax.annotate(f'{resolution:.1f} nm',
                        (cutoff_frequency[index], cutoff_curve[index]),
                        xytext=(7, 10), textcoords='offset points')

        ax.set(xlabel='Spatial frequency (1/µm)', ylabel='Correlation', title=title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=160, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)


    @staticmethod
    def frc_comparison(single_curve, paired_curve, threshold='fixed_threshold',
                            title='Single-frame and paired-frame FRC',
                            save_path=None, show=False):
        """Overlay single-frame and independently paired FRC estimates in one figure.

        Both series use their *own* sampled spatial-frequency coordinates. Single-frame
        samples use the corrected FRC; the independent paired-frame estimate uses the
        ordinary FRC. The selected threshold is shared when it is 1/7; 3-sigma curves
        are shown separately if their sampling differs. Model-based continuations are
        distinctly dashed and drawn only if extrapolation actually supplied a cutoff.

        This method is visualization-only: no numerical curve, crossing, or resolution
        property is modified. Returns (figure, axes) for optional inspection; when
        show=True the figure is displayed and closed by pyplot.
        """
        if not single_curve.single_frame or paired_curve.single_frame:
            raise ValueError('Expected one single-frame result and one paired-frame result.')
        if threshold not in ('fixed_threshold', '3-sigma'):
            raise ValueError("Use 'fixed_threshold' or '3-sigma'.")

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        colors = {'Single-frame': '#E67373', 'Paired-frame': '#3C78A8'}
        for curve, mode in ((single_curve, 'Single-frame'),
                            (paired_curve, 'Paired-frame')):
            color = colors[mode]
            measured = (curve.smoothed_frc_corrected if curve.single_frame
                        else curve.smoothed_frc)
            ax.plot(curve.frequency, measured, color=color, linewidth=1.9,
                    label=f'{mode} measured FRC' + (' (corrected)' if curve.single_frame else ''))
            cutoff_frequency, cutoff_curve = curve.frequency, measured
            if (threshold == 'fixed_threshold' and curve.extrapolation_used
                    and curve.new_frequency is not None
                    and curve.extrapolated_frc is not None):
                n = len(curve.frequency)
                info = curve.extrapolation_fit_info or {}
                if (n >= 2 and curve.frequency[-1] > 0
                        and all(k in info for k in ('A', 'k'))):
                    norm_freq = curve.frequency / curve.frequency[-1]
                    ssnr_fit = info['A'] * np.exp(-info['k'] * norm_freq**2)
                    ax.plot(curve.frequency, ssnr_fit / (1 + ssnr_fit),
                            color=color, alpha=0.7, linestyle='-.', linewidth=1.3,
                            label=f'{mode} fitted model (measured range)')
                ax.plot(curve.new_frequency[n-1:], curve.extrapolated_frc[n-1:],
                        color=color, linestyle='--', linewidth=2.0,
                        label=f'{mode} extrapolated prediction (not measured)')
                if mode == 'Single-frame':
                    ax.axvline(curve.frequency[-1], linestyle=':', color='gray',
                               linewidth=1.1, label='Single-frame measured frequency limit')
                cutoff_frequency, cutoff_curve = curve.new_frequency, curve.extrapolated_frc
            index = curve.get_index(threshold)
            resolution = curve.get_resolution(threshold)
            if (0 <= index < len(cutoff_frequency) and np.isfinite(resolution)):
                ax.scatter([cutoff_frequency[index]], [cutoff_curve[index]],
                           color=color, s=38, zorder=5,
                           label=f'{mode} cutoff: {resolution:.1f} nm')

        if threshold == 'fixed_threshold':
            ax.axhline(1 / 7, color='black', linestyle='--', linewidth=1.1,
                       label='1/7 threshold')
        else:
            threshold_a = single_curve.three_sigma
            threshold_b = paired_curve.three_sigma
            equal = (single_curve.frequency.shape == paired_curve.frequency.shape
                     and np.array_equal(single_curve.frequency, paired_curve.frequency)
                     and np.allclose(threshold_a, threshold_b, equal_nan=True))
            ax.plot(single_curve.frequency, threshold_a, color='black',
                    linestyle='--', linewidth=1, label='3-sigma threshold' if equal
                    else 'Single-frame 3-sigma threshold')
            if not equal:
                ax.plot(paired_curve.frequency, threshold_b, color='gray',
                        linestyle=':', linewidth=1.1,
                        label='Paired-frame 3-sigma threshold')

        ax.set(xlabel='Spatial frequency (1/µm)', ylabel='Correlation', title=title)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=160, bbox_inches='tight')
        if show:
            plt.show()
            plt.close(fig)
        return fig, ax


    @staticmethod
    def sfsc_polar(erf, use_valid=False, save_path=None, show=False):
        """Plot the fitted 3D anisotropic resolution profile without diagnostic text.

        Parameters
        ----------
        erf : DirectionalResolutionInference
            Fitted lateral/axial resolution model.
        use_valid : bool
            Compatibility flag; measured points are currently omitted from the polar
            outline. {default: False}
        save_path : str, pathlib.Path or None
            Optional figure filename for this plotting utility. {default: None}
        show : bool
            Display the figure before closing it. {default: False}

        Returns
        -------
        matplotlib.axes.Axes
            Polar axes belonging to the generated figure.

        Notes
        -----
        Resolution radii are in nm. Optional save_path controls this visualization utility;
        core numerical functions do not write files.
        """
        xy_resolution, a = erf.get_xy_resolution()
        z_resolution, b = erf.get_z_resolution()
        valid_angles = []
        valid_resolutions = []
        if use_valid:
            for angle, resolution in zip(erf.get_angles(), erf.get_resolutions()):
                valid_angles.append(angle)
                valid_angles.append(np.pi - angle)
                valid_angles.append(np.pi + angle)
                valid_angles.append(2 * np.pi - angle)
                for i in range(4):
                    valid_resolutions.append(resolution)
        valid_angles = np.array(valid_angles)
        valid_resolutions = np.array(valid_resolutions)

        angles = np.linspace(0, 2 * np.pi, 24, endpoint=True)
        cosin, sin = np.cos(angles), np.sin(angles)
        frequency_radii = a * b / np.sqrt(b * b * cosin * cosin + a * a * sin * sin)
        radii = 1000 / frequency_radii
        angles = np.hstack((angles,valid_angles))
        radii = np.hstack((radii,valid_resolutions))
        order=np.argsort(angles)
        angles,radii=angles[order],radii[order]
        angles = np.concatenate((angles, [0]))
        radii = np.concatenate((radii, [xy_resolution]))

        fig = plt.figure(figsize=(12, 12), dpi=300)
        ax = plt.subplot(111, projection="polar")
        line = ax.plot(angles, radii, color='#E67373', linewidth=3, alpha=0.8,
                       label=f'XY: {xy_resolution:.1f} nm, Z: {z_resolution:.1f} nm')
        max_radius = radii.max()

        ax.text(-0.01, 1.15 * max_radius, 'XY',
                fontsize=14, fontweight='bold',
                horizontalalignment='right', verticalalignment='center')

        ax.text(np.pi / 2, 1.15 * max_radius, 'Z',
                fontsize=14, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')

        ax.text(1.5 * np.pi, 1.35 * max_radius, 'Resolution (nm)',
                fontsize=16, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')

        ax.text(-0.01, 1.2 * xy_resolution, f'{xy_resolution:.1f}')
        ax.text(np.pi / 1.92, 1.02 * z_resolution, f'{z_resolution:.1f}')

        ax.set_rmax(1.2 * max_radius)

        ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_rgrids(
            np.linspace(0.1 * max_radius, max_radius, 5),
            labels=[f'{x:.1f}' for x in np.linspace(0.1 * max_radius, max_radius, 5)],
            angle=225,
            fontsize=10
        )

        ax.set_thetagrids(
            np.arange(0, 360, 45),
            labels=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
            fontsize=10
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15), fontsize=12)

        ax.set_title('sFSC - lateral/axial resolution',
                     fontsize=16, fontweight='bold', pad=30)

        ax.set_facecolor('#F8F9F9')
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('#CCCCCC')
        ax.spines['polar'].set_linewidth(1)


        if save_path is not None:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
        return ax


    @staticmethod
    def sfsc_polar_legacy(xy_resolution, z_resolution, save_path=None, show=False):
        """Plot a lateral/axial resolution outline from two scalar estimates.

        Parameters
        ----------
        xy_resolution : float
            Positive lateral resolution estimate in nm.
        z_resolution : float
            Axial resolution estimate in nm.
        save_path : str, pathlib.Path or None
            Optional figure filename for this plotting utility. {default: None}
        show : bool
            Display the figure before closing it. {default: False}

        Returns
        -------
        matplotlib.axes.Axes
            Polar axes with resolution radii in nm.
        """
        a = 1000 / xy_resolution
        b = 1000 / z_resolution
        # valid_angles = erf.get_angles()
        # valid_resolutions = erf.get_resolutions()
        # low_angle, high_angle, num_cones = erf.get_angle_range()
        # angles = np.linspace(low_angle, high_angle, num_cones)
        # angle_step = angles[1] - angles[0]
        # num = math.ceil(2 * np.pi / angle_step)

        angles = np.linspace(0, 2 * np.pi, 24, endpoint=True)
        angles = np.concatenate((angles, [0]))
        cosin, sin = np.cos(angles), np.sin(angles)
        frequency_radii = a * b / np.sqrt(b * b * cosin * cosin + a * a * sin * sin)
        radii = 1000 / frequency_radii

        fig = plt.figure(figsize=(12, 12), dpi=300)
        ax = plt.subplot(111, projection="polar")
        line = ax.plot(angles, radii, color='#E67373', linewidth=3, alpha=0.8,
                       label=f'XY: {xy_resolution:.1f} nm, Z: {z_resolution:.1f} nm')
        max_radius = radii.max()

        ax.text(-0.01, 1.15 * max_radius, 'XY',
                fontsize=14, fontweight='bold',
                horizontalalignment='right', verticalalignment='center')

        ax.text(np.pi / 2, 1.15 * max_radius, 'Z',
                fontsize=14, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')

        ax.text(1.5 * np.pi, 1.35 * max_radius, 'Resolution (nm)',
                fontsize=16, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')

        ax.text(-0.01, 1.2 * xy_resolution, f'{xy_resolution}')
        ax.text(np.pi / 2, 1.02 * z_resolution, f'{z_resolution}')

        ax.set_rmax(1.2 * max_radius)

        ax.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)
        ax.set_rgrids(
            np.linspace(0.1 * max_radius, max_radius, 5),
            labels=[f'{x:.1f}' for x in np.linspace(0.1 * max_radius, max_radius, 5)],
            angle=225,
            fontsize=10
        )

        ax.set_thetagrids(
            np.arange(0, 360, 45),
            labels=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
            fontsize=10
        )
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15), fontsize=12)

        ax.set_title('sFSC - lateral/axial resolution',
                     fontsize=16, fontweight='bold', pad=30)

        ax.set_facecolor('#F8F9F9')
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_color('#CCCCCC')
        ax.spines['polar'].set_linewidth(1)

        if save_path is not None:
            plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()
        return ax


    @staticmethod
    def srfrc_display_max(analysis, index):
        """Return one common positive display maximum for the background comparisons.

        Parameters
        ----------
        analysis : RollingFourierRingCorrelation
            Completed analyser retaining all background groups.
        index : int
            Frame/pair index within each computed group's stack.

        Returns
        -------
        float
            Maximum finite positive map value, or 1 for empty foreground.
        """
        maxima=[]
        for maps in analysis.rfrc_maps.values():
            valid=maps[index][np.isfinite(maps[index]) & (maps[index]>0)]
            if valid.size:
                maxima.append(float(np.max(valid)))
        return max(maxima) if maxima else 1.0


    @staticmethod
    def srfrc_maps(analysis, show=True):
        """Plot one row of srFRC maps with absolute background-threshold labels.

        Monitor mode shows three maps side by side, with a common sJet color
        scale [0, max_resolution]. No reference images or pixel masks are drawn.
        Non-monitor mode displays one map with its own right-hand colorbar.
        """
        index=analysis.parameters.get('plot_index')
        if index is None:
            return None
        groups=list(analysis.rfrc_maps.items())
        upper=ResolutionPlotter.srfrc_display_max(analysis,index)
        is_monitor=bool(getattr(analysis,'monitor_change',False)) and len(groups)==3
        fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),4),
                              squeeze=False,constrained_layout=True)
        for column,(setting,maps) in enumerate(groups):
            matches=[row for row in analysis.metrics if row['background_setting']==setting]
            metric=matches[index]
            label=metric['frame']
            # Avoid eagerly evaluating a missing fallback list when metric already
            # contains the absolute threshold (dict.get defaults are eager).
            value=metric.get('background_threshold')
            if value is None:
                series=getattr(analysis,'final_background_thresholds',{}).get(setting,[])
                value=series[index] if index < len(series) else float('nan')
            value=float(value)
            ax_map=axes[0,column]
            artist=ax_map.imshow(maps[index],cmap=sjet_colorbar(),vmin=0,vmax=upper,origin='upper')
            ax_map.set_title(f'srFRC - resolution map - {label}\nBackground threshold = {value:.1f}')
            ax_map.set_axis_off()
        fig.colorbar(artist,ax=list(axes[0]),fraction=0.025,pad=0.025,label='Resolution (nm)')
        if is_monitor:
            fig.suptitle('srFRC background monitoring')
        if show:
            plt.show()
        return fig


    @staticmethod
    def dri_fit(result, save_path=None, show=False):
        """Render the optional DRI fit (visualization only)."""
        fig, ax = plt.subplots()
        xx = np.linspace(0, result.xy_cutoff, 200)
        ax.plot(xx, result.z_cutoff*np.sqrt(np.maximum(0, 1-(xx/result.xy_cutoff)**2)),
                color='#E67373')
        ax.scatter(result.cutoffs*np.cos(result.angles),
                   result.cutoffs*np.sin(result.angles), s=15)
        ax.set(xlabel='Spatial frequency XY (1/um)',
               ylabel='Spatial frequency Z (1/um)',
               title='sFSC - DRI elliptical fit')
        if save_path is not None:
            fig.savefig(save_path, dpi=160, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return fig, ax

    @staticmethod
    def srfrc_export_map(data, upper, label, background_threshold, save_path):
        """Save a single current-frame srFRC preview, leaving data unchanged."""
        fig, ax = plt.subplots(figsize=(8, 5))
        artist = ax.imshow(data, cmap=sjet_colorbar(), vmin=0, vmax=upper)
        ax.set_title(f'srFRC - resolution map - {label}\n'
                     f'Background threshold = {background_threshold:.1f}')
        ax.axis('off')
        fig.colorbar(artist, ax=ax, label='Resolution (nm)')
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
        plt.close(fig)
