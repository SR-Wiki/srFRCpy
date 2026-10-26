import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
import os
from PIL import Image


def add_polar_axes_with_scale(image_path, scale_factor=1.0, unit="nm",
                              max_radius_pixels=None, output_path=None, dpi=300):
    """Overlay physical polar axes on an existing raster plot.
    
    Parameters
    ----------
    image_path : str or pathlib.Path
        Filename of the existing polar-plot raster image.
    scale_factor : float
        Physical units represented by one radial image pixel. {default: 1.0}
    unit : str
        Label for the physical radius unit. {default: 'nm'}
    max_radius_pixels : float or None
        Maximum polar radius in image pixels; None selects an automatic radius.
        {default: None}
    output_path : str or pathlib.Path
        Filename of the annotated output figure. {default: None}
    dpi : int
        Raster output resolution in dots per inch. {default: 300}
    
    Returns
    -------
    tuple
        (figure,axes,metadata) for the annotated plot; the figure is also saved to
        output_path.
    """
    # Read the image.
    if os.path.exists(image_path):
        img_pil = Image.open(image_path)
        img_array = np.array(img_pil)
    else:
        print(f'Invalid image path!')
    # Get the image dimensions.
    height, width = img_array.shape[:2]

    # Create the figure.
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100, facecolor='black')
    ax.set_facecolor('black')

    # Display the image.
    ax.imshow(img_array, extent=[0, width, 0, height], origin='upper')

    # Coordinates of the center.
    center_x, center_y = width / 2, height / 2

    # Calculate the maximum radius.
    if max_radius_pixels is None:
        max_radius_pixels = min(center_x, center_y) * 0.85

    # Calculate the radial ticks.
    num_radial_ticks = 6
    radial_fractions = np.linspace(0.2, 1.0, num_radial_ticks)

    # Physical length = pixel distance × scale factor.
    actual_lengths = [r_frac * max_radius_pixels * scale_factor for r_frac in radial_fractions]

    # Radial grid of concentric circles.
    for i, (r_frac, actual_length) in enumerate(zip(radial_fractions, actual_lengths)):
        if i == 0:  # Skip the innermost circle.
            continue

        radius = r_frac * max_radius_pixels

        # Draw a concentric circle.
        circle = Circle((center_x, center_y), radius,
                        fill=False, linewidth=1.5,
                        color='cyan', alpha=0.7, linestyle='--')
        ax.add_patch(circle)

        # Add radial tick labels in the 0-degree direction.
        label_x = center_x + radius + 15
        label_y = center_y

        # Format the physical length for display.
        if actual_length >= 1000 and unit == "nm":
            display_length = actual_length / 1000
            display_unit = "μm"
        elif actual_length < 1 and unit == "μm":
            display_length = actual_length * 1000
            display_unit = "nm"
        else:
            display_length = actual_length
            display_unit = unit

        # Choose a display format based on the magnitude.
        if display_length < 0.01:
            label_text = f"{display_length:.1e} {display_unit}"
        elif display_length < 1:
            label_text = f"{display_length:.2f} {display_unit}"
        elif display_length < 10:
            label_text = f"{display_length:.1f} {display_unit}"
        else:
            label_text = f"{display_length:.0f} {display_unit}"

        ax.text(label_x, label_y, label_text,
                color='white', fontsize=9, ha='left', va='center',
                fontweight='bold',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    # Angular grid of radial lines.
    num_angular_lines = 12
    angles_deg = np.linspace(0, 360, num_angular_lines, endpoint=False)

    for angle_deg in angles_deg:
        angle_rad = np.deg2rad(angle_deg)

        # Calculate the line endpoint.
        end_x = center_x + max_radius_pixels * np.cos(angle_rad)
        end_y = center_y + max_radius_pixels * np.sin(angle_rad)

        # Draw the angular grid line.
        ax.plot([center_x, end_x], [center_y, end_y], color='cyan', linestyle = '--')

        # Angular tick labels.
        label_radius = max_radius_pixels * 1.08
        label_x = center_x + label_radius * np.cos(angle_rad)
        label_y = center_y + label_radius * np.sin(angle_rad)

        # Adjust text alignment.
        ha = 'center'
        if 45 < angle_deg < 135:
            ha = 'left'
        elif 225 < angle_deg < 315:
            ha = 'right'

        va = 'center'
        if 0 < angle_deg < 180:
            va = 'bottom'
        else:
            va = 'top'

        ax.text(label_x, label_y, f'{angle_deg:.0f}°',
                color='white', fontsize=9, ha=ha, va=va,
                fontweight='bold')

    # Add a marker at the center.
    ax.plot(center_x, center_y, 'o', color='red', markersize=10,
            markeredgewidth=2, markeredgecolor='white',
            label=f'Origin (0,0)\nScale factor: {scale_factor:.3f} px/{unit}')

    # Add axis labels.
    ax.text(center_x, center_y - max_radius_pixels * 1.15, 'Y axis (0°)',
            color='white', fontsize=10, ha='center', va='center',
            fontweight='bold')

    ax.text(center_x + max_radius_pixels * 1.15, center_y, 'X axis (90°)',
            color='white', fontsize=10, ha='center', va='center',
            fontweight='bold')

    # Add the title.
    title_text = f'Polar coordinate overlay\nRadial scale factor: {scale_factor:.4f} px/{unit}'
    ax.text(center_x, height * 0.95, title_text,
            color='white', fontsize=12, ha='center', va='center',
            fontweight='bold')


    # Configure the axes.
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Adjust the layout.
    plt.tight_layout(pad=0)

    # Save the result.
    if output_path is None:
        output_path = f'cross_polar_scale_{scale_factor:.3f}.png'

    plt.savefig(output_path, dpi=dpi, facecolor='black',
                edgecolor='none', bbox_inches='tight', pad_inches=0)
    print(f"Polar image with physical-unit scaling saved to: {output_path}")
    print(f"Radial scale factor: {scale_factor} px/{unit}")

    plt.show()

    return fig, ax, {
        'scale_factor': scale_factor,
        'unit': unit,
        'max_radius_pixels': max_radius_pixels,
        'actual_lengths': actual_lengths
    }




# Run the example.
if __name__ == "__main__":
    print("Overlay polar axes and ticks on the image.")
    print("=" * 60)

    # Image path.
    folder = 'E:/LYZ/'
    image_path = f'{folder}YZ 127.png'


    # 1. Basic example: overlay a polar grid on the image.
    print("\n1. Create the basic example: image with polar grid.")
    add_polar_axes_with_scale(image_path, scale_factor=1.0, unit="nm")

