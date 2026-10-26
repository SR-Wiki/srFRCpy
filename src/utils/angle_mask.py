import math
import numpy as np


class AngleMask:
    def __init__(self, im):
        """Construct the angular coordinate grids used for directional FSC masking.
        
        Parameters
        ----------
        im : numpy.ndarray
            Input image or volume.
        
        Returns
        -------
        None
            Stores coordinate, radius, and angular grids derived from the input volume
            shape.
        """
        d, h, w = im.shape
        z, y, x = np.meshgrid(np.arange(-d/2+1, d/2+1), np.arange(-h/2+1, h/2+1), np.arange(-w/2+1, w/2+1))
        self.z, self.y, self.x = z, y, x
        self.radius = np.sqrt(z * z + y * y + x * x)
        self.zenith = np.asin(np.abs(y / (self.radius + 1e-4)))

    def get_zenith(self):
        """Read the angular grid used by directional masks.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        numpy.ndarray
            Stored angular values in radians.
        """
        return self.zenith
    def get_mask(self, low_angle, high_angle):
        """Select Fourier samples within a directional angular interval.
        
        Parameters
        ----------
        low_angle : float
            Lower bound of the directional angular mask in radians.
        high_angle : float
            Upper bound of the directional angular mask in radians.
        
        Returns
        -------
        numpy.ndarray
            Boolean mask with angular values between low_angle and high_angle,
            inclusive.
        """
        assert low_angle < high_angle, 'low angle must be less than high angle!'
        mask1 = self.zenith >= low_angle
        mask2 = self.zenith <= high_angle
        mask = mask1 * mask2
        return mask
