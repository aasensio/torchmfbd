import numpy as np
from tqdm import tqdm
import torchmfbd.util as util
import torchmfbd.kl_modes as kl_modes
import torch
import sklearn.decomposition
import logging
import scipy.linalg as la
import os

class Basis(object):
    """
    Class that generates a set of Point Spread Functions (PSFs) using Kolmogorov turbulence and computes the Non-negative Matrix Factorization (NMF) of the PSFs.

    Parameters:
    -----------
    n_pixel : int, optional
        Number of pixels for the telescope aperture (default is 128).
    wavelength : float, optional
        Wavelength in nanometers (default is 8542.0).
    diameter : float, optional
        Diameter of the telescope in centimeters (default is 100.0).
    pix_size : float, optional
        Pixel size in arcseconds (default is 0.059).
    central_obs : float, optional
        Central obscuration of the telescope in centimeters (default is 0.0).
    n_modes : int, optional
        Number of modes for the KL basis (default is 250).
    r0_min : float, optional
        Minimum Fried parameter in centimeters (default is 15.0).
    r0_max : float, optional
        Maximum Fried parameter in centimeters (default is 50.0).
    Raises:
    -------
    Exception
        If the pixel size is not small enough to model the telescope with the given diameter.
    
    """
    
    
    def __init__(self,                 
                 n_pixel=128,
                 wavelength=8542.0,
                 diameter=100.0,
                 pix_size=0.059,
                 central_obs=0.0,
                 n_modes=250,
                 r0_min=15.0,
                 r0_max=50.0):
                
        super().__init__()

        self.logger = logging.getLogger("nmf ")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Compute the overfill to properly generate the PSFs from the wavefronts
        
        self.n_pixel = n_pixel
        self.n_modes = n_modes
        self.wavelength = wavelength
        self.diameter = diameter
        self.pix_size = pix_size
        self.central_obs = central_obs

        overfill = util.psf_scale(self.wavelength,
                                  self.diameter, 
                                  self.pix_size)
        
        if (overfill < 1.0):
            raise Exception(f"The pixel size is not small enough to model a telescope with D={self.diameter} cm")

        # Compute telescope aperture
        pupil = util.aperture(npix=self.n_pixel, 
                        cent_obs = self.central_obs / self.diameter,
                        spider=0, 
                        overfill=overfill)
        
        self.pupil = torch.tensor(pupil, dtype=torch.float32).to(self.device)
                
        self.kl = kl_modes.KL()
        basis = self.kl.precalculate(npix_image = self.n_pixel, 
                            n_modes_max = self.n_modes,                            
                            overfill=overfill)

        self.basis = torch.tensor(basis, dtype=torch.float32).to(self.device)
                
        self.r0_min = r0_min
        self.r0_max = r0_max


    def _compute_psf(self, modes, pupil, basis):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """

        # --------------
        # Focused PSF
        # --------------
        # Compute wavefronts from estimated modes                
        wavefront = torch.einsum('i,ilm->lm', modes, basis)

        # Compute the complex phase
        phase = pupil * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft.fft2(phase)
        psf = (torch.conj(ft) * ft).real
        
        # Normalize PSF to unit amplitude        
        psf_norm = psf / torch.sum(psf)
        
        return wavefront, psf_norm
        
    def _compute_psfs(self, n_training, factor=1.0, n_low=44):
        
        self.n_training = n_training

        psf_all = np.zeros((self.n_pixel, self.n_pixel, self.n_training), dtype='float32')
        r0_all = np.zeros((self.n_training), dtype='float32')
        modes_all = np.zeros((self.n_modes, self.n_training), dtype='float32')
        
        for i in tqdm(range(self.n_training)):
                                                                        
            # Use random value of Fried parameter and generate modes
            r0 = np.random.uniform(low=self.r0_min, high=self.r0_max)
            
            coef = (self.diameter / r0)**(5.0/6.0)

            sigma_KL = coef * np.sqrt(self.kl.varKL)

            modes = np.random.normal(loc=0.0, scale=sigma_KL, size=sigma_KL.shape)

            # Set tip-tilt to zero
            modes[0:2] = 0.0

            modes = torch.tensor(modes, dtype=torch.float32).to(self.device)

            modes[0:n_low] *= factor
                                            
            # Compute PSF and convolve with original image
            wavefront, psf = self._compute_psf(modes, self.pupil, self.basis)

            psf_all[:, :, i] = np.fft.fftshift(psf.cpu().numpy())
            r0_all[i] = r0
            modes_all[:, i] = modes.cpu().numpy()            

        return psf_all, modes_all, r0_all
    
    def _compute_psf_diffraction(self):
        
        psf_all = np.zeros((self.n_pixel, self.n_pixel), dtype='float32')                    
                                                                    
        modes = np.zeros(self.n_modes)
        
        modes = torch.tensor(modes, dtype=torch.float32).to(self.device)
                                        
        # Compute PSF and convolve with original image
        wavefront, psf = self._compute_psf(modes, self.pupil, self.basis)

        psf_all = np.fft.fftshift(psf.cpu().numpy())            

        return psf_all
    
    def compute(self, type='nmf', n=100, n_iter=400, verbose=0):
        """
        Compute Non-negative Matrix Factorization (NMF) for a set of generated Point Spread Functions (PSFs).
        Parameters:
        n (int): Number of random PSFs to generate.
        n_iter (int, optional): Maximum number of iterations for the NMF algorithm. Default is 400.
        verbose (int, optional): Verbosity level of the NMF algorithm. Default is 0.
        Returns:
        None
        This function generates `n` random PSFs using Kolmogorov turbulence with a specified range of r0 values.
        It then computes the NMF of the reshaped PSFs and saves the resulting basis, diffraction PSF, modes, and coefficients
        to a file in the 'basis' directory. The filename includes the wavelength, number of modes, and r0 range.
        """

        if type not in ['nmf', 'pca']:
            raise ValueError(f"Invalid type {type} for computing the basis functions")
        
        self.logger.info(f"Generating {n} random PSFs with Kolmogorov turbulence with r0=[{self.r0_min}, {self.r0_max}]...")
        
        psf, modes, r0 = self._compute_psfs(n)
        psf_diff = self._compute_psf_diffraction()

        tmp = psf.reshape((self.n_pixel * self.n_pixel, n)).T

        if type == 'nmf':
            self.logger.info(f"Solving NMF (be patient)...")
            nmf = sklearn.decomposition.NMF(n_components=self.n_modes, init='nndsvda', random_state=0, max_iter=n_iter, verbose=verbose)
            W = nmf.fit_transform(tmp)
            H = nmf.components_
            Vh = H
            coeffs = W

            filename = f'basis/nmf_{int(self.wavelength)}_n_{self.n_modes}_r0_{int(self.r0_min)}_{int(self.r0_max)}.npz'
            
            # Create the directory if it does not exist
            if not os.path.exists('basis'):
                os.makedirs('basis')
            
            self.logger.info(f"Saving NMF file on {filename}...")
            np.savez(filename, basis=Vh, psf_diffraction=psf_diff, modes=modes, coeffs=coeffs, info=[self.wavelength, self.diameter, self.pix_size, self.central_obs])

        if type == 'pca':
            self.logger.info(f"Solving PCA (be patient)...")

            cov = tmp @ tmp.T
            U, S2, Vh = la.svd(cov, full_matrices=False)

            Vh = (Vh @ tmp) / np.sqrt(S2[:, None])

            coeffs = tmp @ Vh.T
        
            filename = f'basis/pca_{int(self.wavelength)}_n_{self.n_modes}_r0_{int(self.r0_min)}_{int(self.r0_max)}.npz'
            
            # Create the directory if it does not exist
            if not os.path.exists('basis'):
                os.makedirs('basis')
            
            self.logger.info(f"Saving PCA file on {filename}...")
            np.savez(filename, basis=Vh, psf_diffraction=psf_diff, modes=modes, coeffs=coeffs, info=[self.wavelength, self.diameter, self.pix_size, self.central_obs])
    

if (__name__ == '__main__'):

    pass