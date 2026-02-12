import numpy as np

def _get_chromis_diversity(d_chromis=3.35):
    """
    Get the CHROMIS rms diversity coefficient peak
    to be multiplied by the defocusing Zernike
    to get the rms wavefront error in radians.

    Parameters
    ----------
    d_chromis : float
        Defocus in mm    
    """
    
    image_scale = 0.0379  # "/pixel
    pixel_size = 5.86     # micron
    diameter = 97         # cm
    wavelength = 3934     # A

    # Transform to cm
    d_chromis *= 0.1
    pixel_size *= 1e-4    
    wavelength *= 1e-8

    # Focal ratio estimated from the image scale in the focal plane
    # scale = pixel_size / (D * F/D) * 180/pi * 3600
    F_D = pixel_size / (diameter * image_scale) * (180.0 / np.pi) * 3600

    # Defocus RMS in radians
    # a_4 = pi * d / (8 * sqrt(3) * lambda * (F/D)**2)
    a_4 = np.pi * d_chromis / (8 * np.sqrt(3) * wavelength * F_D**2)
    
    return a_4

def get_diversity(defocus = 1.0,
                            FD = 45.0,
                            wavelength = 5500.0):
    """
    Get the rms diversity coefficient 
    to be multiplied by the defocusing Zernike
    to get the rms wavefront error in radians.

    Parameters
    ----------
    defocus : float
        Defocus in wavelength units (1.0lambda)
    FD : float
        Focal ratio
    wavelength : float
        Wavelength in Angstrom
    """
    
    Delta = defocus * wavelength # 1 lambda
    
    d_cm = 8 * FD**2 * Delta

    div = np.pi * d_cm / (8.0 * np.sqrt(3) * wavelength * FD**2)    
    
    return div