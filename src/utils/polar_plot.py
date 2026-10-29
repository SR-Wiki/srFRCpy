import matplotlib.pyplot as plt
import numpy as np
import math


def radians_to_degrees(angle):
    """Convert an angle from radians to degrees.
    
    Parameters
    ----------
    angle : float or None
        Angle in radians; None is represented as zero degrees.
    
    Returns
    -------
    float
        Converted angle, or zero for None.
    """
    if angle == 0:
        return 0
    else:
        return angle * 180.0 / np.pi


def polar_plot(erf, use_valid=False, save_path=None, show=False):
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


def polar_plot_V0(xy_resolution, z_resolution, save_path=None, show=False):
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


if __name__ == '__main__':
    ax = polar_plot_V0(138, 639, '../trial_polar.png')