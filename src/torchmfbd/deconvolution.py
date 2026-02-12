import numpy as np
import torch
import torch.nn.functional as F
import torchmfbd.zern as zern
import torchmfbd.util as util
from collections import OrderedDict
from tqdm import tqdm
from skimage.morphology import flood
import scipy.ndimage as nd
from nvitop import Device
import logging
import torchmfbd.kl_modes as kl_modes
import torchmfbd.noise as noise
from torchmfbd.reg_smooth import RegularizationSmooth
from torchmfbd.reg_iuwt import RegularizationIUWT
import glob
import pathlib
import yaml
import torchmfbd.configuration as configuration
import time
import scipy.optimize as optim
from astropy.io import fits
try:
    import ncg_optimizer
    NGC_OPTIMIZER = True
except:
    NGC_OPTIMIZER = False
    pass

class Deconvolution(object):
    def __init__(self, config, add_piston=False):
        """

        Parameters
        ----------
        npix_apodization : int
            Total number of pixel for apodization (divisible by 2)
        device : str
            Device where to carry out the computations
        batch_size : int
            Batch size
        """
        super().__init__()

        torch.set_default_dtype(torch.float32)
        
        self.logger = logging.getLogger("deconvolution ")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        ch = logging.StreamHandler()        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.fft_norm = 'ortho'

        if isinstance(config, dict):
            self.logger.info(f"Using configuration dictionary")
            self.config = config
        else:
            self.logger.info(f"Using configuration file {config}")
            self.config = self.read_config_file(config)

        # Check configuration file for errors
        self.config = configuration._check_config(self.config)
                
        # Check the presence of a GPU
        self.cuda = torch.cuda.is_available()      

        # Check that the GPU compatible
        if len(Device.all()) == 0:
            self.cuda = False  

        # Ger handlers to later check memory and usage of GPUs
        if self.cuda:
            if self.config['optimization']['gpu'] < 0:
                self.logger.info(f"GPU is available but not used. Computing in cpu")
                self.cuda = False
                self.handle = None
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{self.config['optimization']['gpu']}")
                self.handle = Device.all()[self.config['optimization']['gpu']]
                self.logger.info(f"Computing in {self.handle.name()} (free {self.handle.memory_free() / 1024**3:4.2f} GB) - cuda:{self.config['optimization']['gpu']}")
                self.initial_memory_used = self.handle.memory_used()
        else:
            self.logger.info(f"No GPU is available. Computing in cpu")
            self.device = torch.device("cpu")
            self.handle = None

        # self.n_modes = np.sum(np.arange(self.config['psf']['nmax_modes'])+2)
        self.n_modes = self.config['psf']['nmax_modes']
        self.use_jitter = self.config['psf']['jitter']
        self.npix = self.config['images']['n_pixel']
        self.npix_apod = self.config['images']['apodization_border']
        self.remove_gradient_apodization = self.config['images']['remove_gradient_apodization']

        self.psf_model = self.config['psf']['model']

        self.loss_type = self.config['optimization']['loss_type']
        
        # Whether to take into account the piston mode
        self.add_piston = add_piston                     
                
        # Generate Hamming window function for WFS correlation
        if (self.npix_apod > 0):
            self.logger.info(f"Using apodization mask with a border of {self.npix_apod} pixels")
            if self.remove_gradient_apodization:
                self.logger.info(f"Removing gradient in apodization")
            else:
                self.logger.info(f"Not removing gradient in apodization")
            win = np.hanning(2*self.npix_apod)
            winOut = np.ones(self.npix)
            winOut[0:self.npix_apod] = win[0:self.npix_apod]
            winOut[-self.npix_apod:] = win[-self.npix_apod:]
            self.window = np.outer(winOut, winOut)
        else:
            self.logger.info(f"No apodization")
            self.window = np.ones((self.npix, self.npix))

        self.window = torch.tensor(self.window.astype('float32')).to(self.device)        
                
        # Learning rates
        self.lr_obj = self.config['optimization']['lr_obj']
        self.lr_modes = self.config['optimization']['lr_modes']
        self.lr_prior = self.config['optimization']['lr_prior']

        # Type of Fourier filter for the loss
        if 'filter_loss' in self.config['optimization']:
            self.loss_filter = self.config['optimization']['filter_loss']
        else:
            self.loss_filter = 'wiener'
        self.logger.info(f"Using {self.loss_filter} filter for loss computation")

        # Do some output
        self.logger.info(f"Telescope")        
        self.logger.info(f"  * D: {self.config['telescope']['diameter']} m")
        self.logger.info(f"  * pix: {self.config['images']['pix_size']} arcsec")
        
        # Bookkeeping for objects and diversity
        self.ind_object = []
        self.ind_diversity = []
        self.frames = []
        self.sigma = []
        self.diversity = []

        self.external_regularizations = []

        if 'show_object_info' in self.config['optimization']:
            self.show_object_info = self.config['optimization']['show_object_info']
        else:
            self.show_object_info = False

        self.simultaneous_sequences = None
        self.infer_object = None

        if 'orthogonalize' in self.config['psf']:
            self.orthogonalize_basis = self.config['psf']['orthogonalize']
        else:
            self.orthogonalize_basis = False

       
    def define_basis(self, n_modes=None):

        if n_modes is not None:
            self.n_modes = n_modes

        if self.psf_model.lower() in ['zernike', 'kl']:
            
            # Get Noll's n order from the number of modes
            # The summation of the modes needs to fulfill n^2+n-2*(k+1)=0 when no piston is added
            # The summation of the modes needs to fulfill n^2+n-2*k=0 when a piston is added            
            if (self.add_piston):
                self.n_modes += 1
                a = 1.0
                b = 1.0
                c = -2.0 * self.n_modes
            else:            
                a = 1.0
                b = 1.0
                c = -2.0 * (self.n_modes + 1)            
            
            sol1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
            sol2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
            
            n = 0        
            if sol1 > 0.0 and sol1.is_integer():
                n = int(sol1)
            if sol2 > 0.0 and sol2.is_integer():
                n = int(sol2)
            
            if n == 0:
                raise Exception(f"Number of modes {self.n_modes} do not cover a full radial degree")

            self.noll_max = n   

        self.pupil = [None] * self.n_o
        self.basis = [None] * self.n_o
        self.defocus_basis = [None] * self.n_o
        self.rho = [None] * self.n_o
        self.f_x = [None] * self.n_o
        self.f_y = [None] * self.n_o
        self.x = [None] * self.n_o
        self.y = [None] * self.n_o
        self.s_u = [None] * self.n_o
        self.diffraction_limit = [None] * self.n_o
        self.cutoff = [None] * self.n_o
        self.image_filter = [None] * self.n_o

        # First locate unique wavelengths. We will use the same basis for the same wavelength
        ind_wavelengths = []
        unique_wavelengths = []

        for i in range(self.n_o):
            self.cutoff[i] = self.config[f'object{i+1}']['cutoff']
            self.image_filter[i] = self.config[f'object{i+1}']['image_filter']
            self.s_u[i] = self.config[f'object{i+1}']['s_u_joint']
            w = self.config[f'object{i+1}']['wavelength']
            if w not in unique_wavelengths:
                unique_wavelengths.append(w)
            
            ind_wavelengths.append(unique_wavelengths.index(w))

        # Normalize wavelengths to scale basis
        unique_wavelengths = np.array(unique_wavelengths).astype('float32')
        normalized_wavelengths = unique_wavelengths / np.max(unique_wavelengths)

        # Now iterate over all unique wavelengths and associate the basis
        # to the corresponding object
        for i in range(len(unique_wavelengths)):

            wavelength = unique_wavelengths[i]

            # Compute the overfill to properly generate the PSFs from the wavefronts
            overfill = util.psf_scale(wavelength, 
                                    self.config['telescope']['diameter'], 
                                    self.config['images']['pix_size'])

            if (overfill < 1.0):
                raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

            # Compute telescope aperture
            pixel_size_pupil = self.config['telescope']['diameter'] / self.npix
            pupil = util.aperture(npix=self.npix, 
                            cent_obs = self.config['telescope']['central_obscuration'] / self.config['telescope']['diameter'], 
                            spider=self.config['telescope']['spider'] / pixel_size_pupil, 
                            overfill=overfill)
            
            # Obtain defocus basis for phase diversity
            defocus_basis = self.get_defocus_basis(overfill=overfill)
                        
            # PSF model parameterized with the wavefront
            if (self.psf_model.lower() in ['zernike', 'kl', 'nmf']):
                            
                if (self.psf_model.lower() not in ['zernike', 'kl', 'nmf']):
                    raise Exception(f"Unknown basis {self.basis}. Use 'zernike' or 'kl' for wavefront expansion")
            
                if (self.psf_model.lower() == 'zernike'):

                    self.logger.info(f"PSF model: wavefront expansion in Zernike modes")
                    
                    found, filename = self.find_basis_wavefront('zernike', self.n_modes, int(wavelength))

                    # Define Zernike modes        
                    if found:
                        self.logger.info(f"Loading precomputed Zernike {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']           
                    else:                
                        self.logger.info(f"Computing Zernike modes {filename}")
                        basis = self.precalculate_zernike(overfill=overfill)

                        # Add piston mode if needed
                        if (self.add_piston):
                            basis = np.concatenate([pupil * np.ones((1, self.npix, self.npix)), basis[0:self.n_modes, :, :]], axis=0)
                            self.n_modes += 1

                        # Orthogonalize the Zernike modes if needed
                        if self.orthogonalize_basis:
                            self.logger.info(f"Orthogonalizing Zernike modes")
                            basis = util.orthogonalize(basis, pupil)
                            self.logger.info(f"  * Orthogonalization done")

                        np.savez(f"{filename}", basis=basis)

                if (self.psf_model.lower() == 'kl'):

                    self.logger.info(f"PSF model: wavefront expansion in KL modes")

                    found, filename = self.find_basis_wavefront('kl', self.n_modes, int(wavelength))

                    if found:
                        self.logger.info(f"Loading precomputed KL basis: {filename}")
                        tmp = np.load(f"{filename}")
                        basis = tmp['basis']                
                    else:
                        self.logger.info(f"Computing KL modes {filename}")
                        self.kl = kl_modes.KL()              
                        
                        basis = self.kl.precalculate(npix_image = self.npix, 
                                        n_modes_max = self.n_modes,                                 
                                        overfill=overfill)
                        
                        # Add piston mode if needed                        
                        if (self.add_piston):
                            basis = np.concatenate([pupil * np.ones((1, self.npix, self.npix)), basis[0:self.n_modes, :, :]], axis=0)
                            self.n_modes += 1

                        # Orthogonalize the KL modes if needed
                        if self.orthogonalize_basis:
                            self.logger.info(f"Orthogonalizing KL modes")
                            basis = util.orthogonalize(basis, pupil)
                            self.logger.info(f"  * Orthogonalization done")

                        np.savez(f"{filename}", basis=basis, variance=self.kl.varKL)
                
                if (self.psf_model.lower() == 'nmf'):
                    
                    self.logger.info(f"PSF model: PSF expansion in NMF modes")

                    self.logger.info(f"Loading precomputed NMF basis: {self.config['psf']['filename']}")
                    f = np.load(self.config['psf']['filename'])

                    n_psf = int(np.sqrt(f['basis'].shape[1]))

                    basis = f['basis'][0:self.n_modes, :].reshape((self.n_modes, n_psf, n_psf))

                    basis = basis[:, n_psf//2 - self.npix//2:n_psf//2 + self.npix//2, n_psf//2 - self.npix//2:n_psf//2 + self.npix//2]
                    
                pupil = torch.tensor(pupil.astype('float32')).to(self.device)
                basis = torch.tensor(basis[0:self.n_modes, :, :].astype('float32')).to(self.device)
                defocus_basis = torch.tensor(defocus_basis.astype('float32')).to(self.device)

                # Following van Noort et al. (2005) we normalize the basis to the maximum wavelength
                # so that the wavefront is given in radians
                if (self.psf_model.lower() in ['zernike', 'kl']):
                    basis /= normalized_wavelengths[i]
                    defocus_basis /= normalized_wavelengths[i]

                if (self.add_piston):
                    self.logger.info(f"Adding piston mode...")
                
                self.logger.info(f"  * Using {self.n_modes} modes...")
                
            
            # Output jitter
            if self.use_jitter:
                self.logger.info(f"PSF model: including jitter")
                
            # Compute the diffraction limit and the frequency grid
            # Frequency is given in units of the cutoff frequency
            cutoff = self.config['telescope']['diameter'] / (wavelength * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.npix, d=self.config['images']['pix_size']) / cutoff
            
            f_x, f_y = np.meshgrid(freq, freq)
            rho = np.sqrt(f_x ** 2 + f_y ** 2).astype('float32')
            rho = torch.tensor(rho.astype('float32')).to(self.device)

            # Fourier coordinates for jitter OTF given in pixel
            f = np.fft.fftfreq(self.npix)
            f_x, f_y = np.meshgrid(freq, freq)
            f_x = torch.tensor(f_x.astype('float32')).to(self.device)
            f_y = torch.tensor(f_y.astype('float32')).to(self.device)

            x = np.linspace(-self.npix//2, self.npix//2-1, self.npix)
            y = np.linspace(-self.npix//2, self.npix//2-1, self.npix)
            X, Y = np.meshgrid(x, y)
            X = torch.tensor(X.astype('float32')).to(self.device)
            Y = torch.tensor(Y.astype('float32')).to(self.device)
            
            diffraction_limit = wavelength * 1e-8 / self.config['telescope']['diameter'] * 206265.0

            self.logger.info(f"Wavelength {i} ({wavelength} A)")
            self.logger.info(f"  * Diffraction: {diffraction_limit} arcsec")
            self.logger.info(f"  * Diffraction (x1.22): {1.22 * diffraction_limit} arcsec")

            for j in range(self.n_o):
                if ind_wavelengths[j] == i:
                    self.pupil[j] = pupil
                    self.basis[j] = basis
                    self.defocus_basis[j] = defocus_basis
                    self.rho[j] = rho
                    self.f_x[j] = f_x * cutoff
                    self.f_y[j] = f_y * cutoff
                    self.x[j] = X
                    self.y[j] = Y
                    self.diffraction_limit[j] = diffraction_limit

        return
    
    def set_regularizations(self):

        # Regularization parameters
        self.logger.info(f"Regularizations")
        self.regularization = []
        self.index_regularization = {
            'tiptilt': [],
            'modes': [],
            'object': []
        }
        
        loop = 0

        for k, v in self.config['regularization'].items():            
            if v['lambda'] != 0.0:
                
                if 'smooth' in k:                
                    tmp = RegularizationSmooth(lambda_reg=v['lambda'], variable=v['variable'])
                if 'iuwt' in k:
                    tmp = RegularizationIUWT(lambda_reg=v['lambda'], variable=v['variable'], nbands=v['nbands'], n_pixel=self.npix)

                self.regularization.append(tmp.to(self.device))
                self.index_regularization[v['variable']].append(loop)
                tmp.print()
                loop += 1

        self.logger.info(f"External regularizations")
        for reg in self.external_regularizations:
            self.regularization.append(reg.to(self.device))
            self.index_regularization[reg.variable].append(loop)
            reg.print()
            loop += 1

    def add_external_regularizations(self, external_regularization):        
        """
        Adds external regularizations to the model.

        Parameters:
        -----------

        external_regularization : callable
            A function or callable object that applies the external regularization.
        lambda_reg : float
            The regularization strength parameter.
        variable : str
            The name of the variable to which the regularization is applied.
        **kwargs : dict
            Additional keyword arguments to pass to the external_regularization function.

        """

        # Regularization parameters
        self.logger.info(f"External regularization")
        
        self.external_regularizations.append(external_regularization)
        
    def read_config_file(self, filename):
        """
        Read a configuration file in YAML format.

        Parameters:
        -----------
        filename : str
            The name of the configuration file.

        Returns:
        --------
        dict
            A dictionary containing the configuration parameters.
        """

        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        return config

    def find_basis_wavefront(self, basis, nmax, wavelength):

        p = pathlib.Path('basis/')
        p.mkdir(parents=True, exist_ok=True)

        files = glob.glob(f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.config['images']['n_pixel']}px_{wavelength}A_*.npz")

        if len(files) == 0:
            return False, f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.npix}px_{wavelength}A_{nmax}.npz"
        
        nmodes = []

        for f in files:
            n = int(f.split('_')[-1].split('.')[0])
            if n >= nmax:
                nmodes.append(n)
                self.logger.info(f"Found basis file with {n} modes that can be used for {nmax} modes")

        if len(nmodes) == 0:
            return False, f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.npix}px_{wavelength}A_{nmax}.npz"
        
        filename = f"basis/{basis}_{int(self.config['telescope']['diameter'])}cm_{self.npix}px_{wavelength}A_{min(nmodes)}.npz"
        
        return True, filename
        
    def precalculate_zernike(self, overfill):
        """
        Precalculate Zernike polynomials for a given overfill factor.
        This function computes the Zernike polynomials up to `self.n_modes` and 
        returns them in a 3D numpy array. The Zernike polynomials are calculated 
        over a grid defined by `self.npix` and scaled by the `overfill` factor.
        Parameters:
        -----------
        overfill : float
            The overfill factor used to scale the radial coordinate `rho`.
        Returns:
        --------
        Z : numpy.ndarray
            A 3D array of shape (self.n_modes, self.npix, self.npix) containing 
            the precalculated Zernike polynomials. Each slice `Z[mode, :, :]` 
            corresponds to a Zernike polynomial mode.
        """
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0

        Z = np.zeros((self.n_modes, self.npix, self.npix))

        noll_Z = 2 + np.arange(self.n_modes)
        
        for mode in tqdm(range(self.n_modes)):
                                                
            jz = noll_Z[mode]
            n, m = zern.zernIndex(jz)
            Zmode = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
            Z[mode, :, :] = Zmode * aperture_mask
                
        return Z
    
    def get_defocus_basis(self, overfill):
        """
        Precalculate Zernike polynomials for a given overfill factor.
        This function computes the Zernike polynomials up to `self.n_modes` and 
        returns them in a 3D numpy array. The Zernike polynomials are calculated 
        over a grid defined by `self.npix` and scaled by the `overfill` factor.
        Parameters:
        -----------
        overfill : float
            The overfill factor used to scale the radial coordinate `rho`.
        Returns:
        --------
        Z : numpy.ndarray
            A 3D array of shape (self.n_modes, self.npix, self.npix) containing 
            the precalculated Zernike polynomials. Each slice `Z[mode, :, :]` 
            corresponds to a Zernike polynomial mode.
        """
        
        Z_machine = zern.ZernikeNaive(mask=[])
        x = np.linspace(-1, 1, self.npix)
        xx, yy = np.meshgrid(x, x)
        rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        aperture_mask = rho <= 1.0
        
        defocus = Z_machine.Z_nm(2, 0, rho, theta, True, 'Jacobi')
                
        return defocus * aperture_mask
    
    def compute_annealing(self, modes, n_iterations):
        """
        Annealing function
        We start with 2 modes and end with all modes but we give steps of the number of
        Zernike modes for each n

        Args:
            annealing (_type_): _description_
            n_iterations (_type_): _description_

        Returns:
            _type_: _description_
        """        
        
        anneal = np.zeros(n_iterations, dtype=int)
        
        # Annealing schedules          
        if self.config['annealing']['type'] == 'linear':
            self.logger.info(f"Adding modes using linear schedule")
            for i in range(n_iterations):
                if (i < self.config['annealing']['start_pct'] * n_iterations):
                    anneal[i] = modes[0]
                elif (i > self.config['annealing']['end_pct'] * n_iterations):
                    anneal[i] = modes[-1]
                else:
                    x0 = self.config['annealing']['start_pct'] * n_iterations
                    x1 = self.config['annealing']['end_pct'] * n_iterations
                    y0 = 0
                    y1 = len(modes)-1
                    index = np.clip((y1 - y0) / (x1 - x0) * (i - x0) + y0, y0, y1)
                    anneal[i] = modes[int(index)]

        if self.config['annealing']['type'] == 'sigmoid':
            self.logger.info(f"Adding modes using sigmoid schedule")
            a = 7
            b = -5
            x = np.linspace(0, 1, n_iterations)            
            anneal = (self.noll_max - 2) * (util.sigmoid(a) - util.sigmoid(a + x * (b-a)) ) / ( util.sigmoid(a) - util.sigmoid(b) )
            
            anneal = (anneal + 0.1).astype(int)            
            anneal = modes[anneal]

        if self.config['annealing']['type'] == 'none':
            self.logger.info(f"All modes always active")
            anneal = np.ones(n_iterations, dtype=int) * modes[-1]

        return anneal
    
    def compute_diffraction_masks(self):
        """
        Compute the diffraction masks for the given dimensions and store them as class attributes.
        Args:
            n_x (int): The number of pixels in the x-dimension.
            n_y (int): The number of pixels in the y-dimension.
        Attributes:
            mask_diffraction (numpy.ndarray): A 3D array of shape (n_o, n_x, n_y) containing the diffraction masks.
            mask_diffraction_th (torch.Tensor): A tensor containing the diffraction masks, converted to float32 and moved to the specified device.
            mask_diffraction_shift (numpy.ndarray): A 3D array of shape (n_o, n_x, n_y) containing the FFT-shifted diffraction masks.
        """
        
        # Compute the diffraction masks and convert to tensor
        self.mask_diffraction = [None] * self.n_o
        self.mask_diffraction_th = [None] * self.n_o
        self.mask_diffraction_shift = [None] * self.n_o
        
        for i in range(self.n_o):
            
            self.mask_diffraction[i] = torch.zeros_like(self.rho[i])
            ind = torch.where(self.rho[i] <= self.cutoff[i][0])
            self.mask_diffraction[i][ind[0], ind[1]] = 1.0

            ind = torch.where(self.rho[i] > self.cutoff[i][1])
            self.mask_diffraction[i][ind[0], ind[1]] = 0.0

            ind = torch.where((self.rho[i] > self.cutoff[i][0]) & (self.rho[i] <= self.cutoff[i][1]))

            # self.mask_diffraction[i][ind[0], ind[1]] = 1.0 - (self.rho[i][ind[0], ind[1]] - self.cutoff[i][0]) / (self.cutoff[i][1] - self.cutoff[i][0])
            self.mask_diffraction[i][ind[0], ind[1]] = torch.cos(np.pi / 2.0 * (self.rho[i][ind[0], ind[1]] - self.cutoff[i][0]) / (self.cutoff[i][1] - self.cutoff[i][0]))

            self.mask_diffraction_th[i] = self.mask_diffraction[i].to(self.device).float()
            
            # Shifted mask used for the Lofdahl & Scharmer filter
            self.mask_diffraction_shift[i] = np.fft.fftshift(self.mask_diffraction[i].cpu().numpy())
                     
    def compute_psfs(self, modes, diversity, jitter=None):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
        modes (torch.Tensor): A tensor of shape (batch_size, num_modes, height, width) representing the modes.
        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - psf_ft (torch.Tensor): The FFT of the normalized PSFs.
        """
        

        _, n_f, n_active = modes.shape
                                        
        psf_norm = [None] * self.n_o
        psf_ft = [None] * self.n_o
                    
        for i in range(self.n_o):            
            # Compute wavefronts from estimated modes                
            wavefront = torch.einsum('ijk,klm->ijlm', modes, self.basis[i][0:n_active, :, :])
            
            # Reuse the same wavefront per object but add the diversity
            wavef = []
            for j in self.init_frame_diversity[i]:                
                div = diversity[i][:, j:j+n_f, None, None] * self.defocus_basis[i][None, None, :, :]
                wavef.append(wavefront + div)
            
            wavef = torch.cat(wavef, dim=1)
            
            # Compute the complex phase
            phase = self.pupil[i][None, None, :, :] * torch.exp(1j * wavef)

            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(phase, norm=self.fft_norm)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, [-1, -2], keepdim=True)

            # FFT of the PSF
            psf_ft[i] = torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

            if self.use_jitter and n_active > 20:

                sigma_x = torch.exp(jitter[:, :, 0])
                sigma_y = torch.exp(jitter[:, :, 1])
                rho_xy = torch.tanh(jitter[:, :, 2]) * sigma_x * sigma_y
                
                tx = self.f_x[i][None, None, :, :]**2 * sigma_x[:, :, None, None]**2 
                ty = self.f_y[i][None, None, :, :]**2 * sigma_y[:, :, None, None]**2
                txy = 2 * self.f_x[i][None, None, :, :] * self.f_y[i][None, None, :, :] * rho_xy[:, :, None, None]

                jitter_otf = torch.exp(-2 * np.pi**2 * (tx + ty + txy))
                
                psf_ft[i] = psf_ft[i] * jitter_otf
        
        return psf_norm, psf_ft
    
    def compute_psfs_nmf(self, modes):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
        modes (torch.Tensor): A tensor of shape (batch_size, num_modes, height, width) representing the modes.
        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - psf_ft (torch.Tensor): The FFT of the normalized PSFs.
        """
                
        n_active = modes.shape[2]
                                        
        psf_norm = [None] * self.n_o
        psf_ft = [None] * self.n_o
        
        for i in range(self.n_o):

            # Compute PSF from estimated modes                
            psf = torch.einsum('ijk,klm->ijlm', modes, self.basis[i][0:n_active, :, :])

            psf = torch.fft.fftshift(psf, dim=[-2, -1])
                        
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, [-1, -2], keepdim=True)

            # FFT of the PSF
            psf_ft[i] = torch.fft.fft2(psf_norm[i], norm=self.fft_norm)
        
        return psf_norm, psf_ft
    
    def compute_psf_diffraction(self):
        """
        Compute the Point Spread Functions (PSFs) from diffraction
        
        Returns:
        tuple: A tuple containing:
            - psf_norm (torch.Tensor): The normalized PSFs.
            - psf_ft (torch.Tensor): The FFT of the normalized PSFs.
        """
        
        psf_ft = [None] * self.n_o
        psf_norm = [None] * self.n_o

        for i in range(self.n_o):
            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(self.pupil[i], norm=self.fft_norm)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf)

            # FFT of the PSF
            psf_ft[i] = torch.fft.fft2(psf_norm[i], norm=self.fft_norm)

        return psf_norm, psf_ft
    
    def lofdahl_scharmer_filter(self, Sconj_S, Sconj_I, sigma):
        """
        Applies the Löfdahl-Scharmer filter to the given input tensors.
        Parameters:
        -----------
        Sconj_S : torch.Tensor
            The conjugate of the Fourier transform of the observed image.
        Sconj_I : torch.Tensor
            The conjugate of the Fourier transform of the ideal image.
        Returns:
        --------
        torch.Tensor
            A tensor representing the mask after applying the Löfdahl-Scharmer filter.
        """
        den = torch.conj(Sconj_I) * Sconj_I
        H = (Sconj_S / den).real        

        H = torch.fft.fftshift(H).detach().cpu().numpy()
        
        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,3,3], mode='wrap')    

        sigma_1 = torch.mean(sigma, axis=-1)[:, None, None].cpu().numpy()
        
        filt = 1.0 - H * sigma_1
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0
                
        nb, nx, ny = filt.shape

        mask = np.zeros_like(filt).astype('float32')

        for ib in range(nb):                
            mask[ib, :, :] = flood(1.0 - filt[ib, :, :], (nx//2, ny//2), tolerance=0.9)
            mask[ib, :, :] = np.fft.fftshift(mask[ib, :, :])
        
        return torch.tensor(mask.astype('float32')).to(Sconj_S.device)

    def get_su_s2(self, obj, sigma, pars_s0):
        """
        Transforms the parameters of the s0 function to ensure they are in the correct range.
        Parameters:
        -----------
        pars_s0 : torch.Tensor
            A tensor of shape (batch_size, n_o, 4) containing the parameters for the s0 function.
        i : int
            The index of the object for which to compute s_u.
        Returns:
        --------
        tuple: A tuple containing:
            - K (torch.Tensor): The transformed K parameter.
            - v0 (torch.Tensor): The transformed v0 parameter.
            - p (torch.Tensor): The transformed p parameter.
        """
        # K = torch.exp(2*F.sigmoid(pars_s0[:,  0]))[:, None, None]
        # v0 = torch.exp(5.0*F.logsigmoid(pars_s0[:, 1]/2))[:, None, None]
        # p = torch.exp(1.6*F.sigmoid(pars_s0[:, 2]))[:, None, None]

        if self.loss_type == 'marginal':
            K = torch.exp(pars_s0[:, obj, 0])[:, None, None] * self.npix
            v0 = torch.exp(pars_s0[:, obj, 1])[:, None, None]            
            p = torch.exp(torch.sigmoid(pars_s0[:, obj, 2]) * np.log(10.0))[:, None, None]

            v = self.rho[obj][None, :, :]

            s_u = K / (1.0 + (v/v0)**2)**(p/2.0)
            # s_u = K / (1.0 + (v/v0)**p)

            # s2 = torch.mean(sigma[obj]**2, dim=1)
            s2 = torch.exp(pars_s0[:, obj, 3])
        else:
            K, v0, p = None, None, None
            s_u = self.s_u[obj] * torch.ones_like(self.rho[obj]).to(self.device)
            s2 = torch.mean(sigma[obj]**2, dim=1) #* 10
        
        return s_u, s2, K, v0, p

    def compute_object(self, images_ft, psf_ft, sigma, plane, type_filter='tophat', pars_s0=None):
        """
        Compute the object in Fourier space using the specified filter.
        Parameters:
        --------
        images_ft (torch.Tensor): 
            The Fourier transform of the observed images.
        psf_ft (torch.Tensor): 
            The Fourier transform of the point spread function (PSF).
        type_filter (str, optional): 
            The type of filter to use ('tophat'/'scharmer'). Default is 'tophat'.
        Returns:
        --------
        torch.Tensor: The computed object in Fourier space.
        """

        out_ft = [None] * self.n_o
        out_filter_ft = [None] * self.n_o
        out_filter = [None] * self.n_o
        
        for i in range(self.n_o):
                        
            # Sconj_S = torch.sum(sigma[i][:, :, None, None] * torch.conj(psf_ft[i]) * psf_ft[i], dim=1)
            # Sconj_I = torch.sum(sigma[i][:, :, None, None] * torch.conj(psf_ft[i]) * images_ft[i], dim=1)

            # We assume, for the moment, that the noise is the same for all frames
            Sconj_S = torch.sum(torch.conj(psf_ft[i]) * psf_ft[i], dim=1)
            Sconj_I = torch.sum(torch.conj(psf_ft[i]) * images_ft[i], dim=1)

            s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0)

            # s2 = torch.mean(sigma[i]**2, dim=1) * 10            
            # Use Lofdahl & Scharmer (1994) filter
            if (self.image_filter[i] == 'scharmer'):

                mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I, sigma[i]**2) * self.mask_diffraction_th[i][None, :, :]
                
                out_ft[i] = Sconj_I / (Sconj_S + s2[:, None, None] / s_u)
                            
                out_filter_ft[i] = out_ft[i] * mask
            
            # Use simple Wiener filter with tophat prior            
            if (self.image_filter[i] == 'tophat'):
                                                
                out_ft[i] = Sconj_I / (Sconj_S + s2[:, None, None] / s_u)
                
                out_filter_ft[i] = out_ft[i] * self.mask_diffraction_th[i][None, :, :]

            out_filter[i] = torch.fft.ifft2(out_filter_ft[i]).real

            # Add the gradient that we removed
            if self.remove_gradient_apodization:
                 out_filter[i] += plane[i][:, 0, :, :]
        
        return out_ft, out_filter_ft, out_filter
    
    def compute_loss(self, frames_ft, psf_ft, sigma, type_filter='tophat', pars_s0=None):        
        """
        Compute the object in Fourier space using the specified filter.
        Parameters:
        --------
        images_ft (torch.Tensor): 
            The Fourier transform of the observed images.
        psf_ft (torch.Tensor): 
            The Fourier transform of the point spread function (PSF).
        type_filter (str, optional): 
            The type of filter to use ('tophat'/'scharmer'). Default is 'tophat'.
        Returns:
        --------
        torch.Tensor: The computed object in Fourier space.
        """

        out_ft = [None] * self.n_o
        out_filter_ft = [None] * self.n_o
        out_filter = [None] * self.n_o

        loss_data_total = torch.tensor(0.0).to(self.device)
        loss_prior_total = torch.tensor(0.0).to(self.device)
        loss_total = torch.tensor(0.0).to(self.device)

        self.pars_s0_avg = [None] * 4
        
        for i in range(self.n_o):

            if self.loss_filter == 'wiener':

                s_u, s2, K, v0, p = self.get_su_s2(obj=i, sigma=sigma, pars_s0=pars_s0)

                # s_u, K, v0, p = self.get_su(self.rho[i], pars_s0[:, i, :])
                                
                # The value of s_u in the case of the joint estimation should be
                # selected by hand to give good results
                # sigma**2/s_u should be the ration between noise and estimated object power spectrum
                # s2 = torch.mean(sigma[i]**2, dim=1) * 10
                hu2 = s2[:, None, None] + s_u * torch.sum(psf_ft[i] * torch.conj(psf_ft[i]), dim=1)                
                du2 = torch.sum(frames_ft[i] * torch.conj(frames_ft[i]), dim=1)
                hu_du = torch.sum(torch.conj(frames_ft[i]) * psf_ft[i], dim=1)
                hu_du2 = s_u * hu_du * torch.conj(hu_du)
                
                loss_data = 0.5 * (du2 - hu_du2 / hu2) / s2[:, None, None]

                loss_data *= self.mask_diffraction_th[i][None, :, :]
                
                # If we are doing a marginal estimation of the object, we 
                # # need to add the effect of the marginalized object and also add 
                # a prior on the parameters of s_u and on s2 to keep them in a reasonable range
                if self.loss_type == 'marginal':

                    self.pars_s0_avg = [torch.mean(K).detach(), torch.mean(v0).detach(), torch.mean(p).detach(), torch.mean(s2).detach()]
                    self.pars_s0_out[:, i, 0] = K[:, 0, 0].detach()
                    self.pars_s0_out[:, i, 1] = v0[:, 0, 0].detach()
                    self.pars_s0_out[:, i, 2] = p[:, 0, 0].detach()
                    self.pars_s0_out[:, i, 3] = s2.detach()
                    
                    # Prior loss consequence of the marginalization of the object in the joing loss
                    loss_prior_marginal = 0.5 * torch.log(hu2)

                    # Prior loss on sigma**2 to avoid zero division and to keep it in a reasonable range
                    # We use a Gaussian prior on log(sigma**2) with mean given by the average of sigma**2 
                    # and a sufficiently large variance to avoid constraining it too much
                    sig2 = torch.mean(sigma[i]**2, dim=1)[:, None, None]
                    loss_prior_s2 = (torch.log(s2[:, None, None]) - torch.log(sig2))**2

                    # Prior loss on K, v0 and p to keep them in a reasonable range

                    # Gaussian prior on log(K) with mean log(1.0) (peak power spectrum for normalized images) and variance 1.0
                    loss_prior_K = (torch.log(K) - np.log(self.K_prior[0]))**2 / self.K_prior[1]**2

                    # Gaussian prior on log(v0) with mean log(0.1) (cutoff frequency for the power spectrum) and variance 1.0
                    loss_prior_v0 = (torch.log(v0) - np.log(self.v0_prior[0]))**2 / self.v0_prior[1]**2

                    # Gaussian prior on log(p) with mean log(2.0) (power law index for the power spectrum) and variance 1.0
                    loss_prior_p = (p - self.p_prior[0])**2 / self.p_prior[1]**2

                    loss_prior = loss_prior_marginal + loss_prior_s2 + loss_prior_K + loss_prior_v0 + loss_prior_p

                    loss_prior *= self.mask_diffraction_th[i][None, :, :]

                    loss = loss_data + loss_prior
                else:
                    loss_prior = torch.tensor(0.0).to(self.device)
                    loss = loss_data
                    
            if self.loss_filter == 'lowpass':                            
                                
                # sigma is acting, indeed, as a regularization, to avoid zero division
                # We could use another number (constant or not) but this one does the trick
                Q = torch.sum(psf_ft[i] * torch.conj(psf_ft[i]), dim=1)                
                t1 = torch.sum(frames_ft[i] * torch.conj(frames_ft[i]), dim=1)
                t2 = torch.sum(torch.conj(frames_ft[i]) * psf_ft[i], dim=1)                
                
                # Use Lofdahl & Scharmer (1994) filter
                if (self.image_filter[i] == 'scharmer'):

                    # We assume, for the moment, that the noise is the same for all frames
                    Sconj_S = torch.sum(torch.conj(psf_ft[i]) * psf_ft[i], dim=1)
                    Sconj_I = torch.sum(torch.conj(psf_ft[i]) * frames_ft[i], dim=1)

                    mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I, sigma[i]) * self.mask_diffraction_th[i][None, :, :]
                                
                    loss = t1 - mask * t2 * torch.conj(t2) / (Q + 1e-10)
                
                # Use simple Wiener filter with tophat prior            
                if (self.image_filter[i] == 'tophat'):                    
                    loss = t1 - self.mask_diffraction_th[i][None, :, :] * t2 * torch.conj(t2) / (Q + 1e-10)
            
            loss_data_total += torch.mean(loss_data).real
            loss_prior_total += torch.mean(loss_prior).real
            loss_total += torch.mean(loss).real
        
        return loss_data_total, loss_prior_total, loss_total

    
    def fft_filter(self, image_ft):
        """
        Applies a Fourier filter to the input image in the frequency domain.

        Parameters:
        -----------
        image_ft : torch.Tensor
            The input image in the frequency domain (Fourier transformed).

        Returns:
        --------
        torch.Tensor
            The filtered image in the frequency domain.
        """
        out = [None] * self.n_o
        for i in range(self.n_o):
            out[i] = image_ft[i] * self.mask_diffraction_th[i][None, :, :]

        return out
            
    def add_frames(self, frames, sigma=None, id_object=0, id_diversity=0, diversity=0.0, XY=None):
        """
        Add frames to the deconvolution object.
        Parameters:
        -----------
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_objects, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        id_object : int, optional
            The object index to which the frames belong (default is 0).
        diversity : torch.Tensor, optional
            The diversity coefficient to use for the deconvolution (n_sequences, n_objects).
            If None, the diversity coefficient is set to zero for all objects.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Adding frames for object {id_object} - diversity {id_diversity} - defocus {diversity}")

        if sigma is None:
            self.logger.info(f"Estimating noise...")
            sigma = noise.compute_noise(frames).to(self.device)
            self.logger.info(f"   * Average noise: {torch.mean(sigma)}")
            # sigma = 1.0 / sigma**2            
            # sigma = torch.tensor(sigma.astype('float32')).to(self.device)

        self.ind_object.append(id_object)        
        self.ind_diversity.append(id_diversity)

        self.frames.append(frames)
        self.sigma.append(sigma)

        # If diversity is a scalar, we need to create a tensor of the same size as the number
        # of sequences
        if np.isscalar(diversity):
            diversity = torch.full(frames.shape[0:1], diversity, dtype=torch.float32).to(self.device)
                
        self.diversity.append(diversity)

        if XY is not None:
            if not torch.is_tensor(XY):
                self.XY = XY.astype('float32')
            else:
                self.XY = XY

    def remove_frames(self):
        """
        Add frames to the deconvolution object.
        Parameters:
        -----------
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_objects, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        id_object : int, optional
            The object index to which the frames belong (default is 0).
        diversity : torch.Tensor, optional
            The diversity coefficient to use for the deconvolution (n_sequences, n_objects).
            If None, the diversity coefficient is set to zero for all objects.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Removing frames for all objects...")

        
        self.ind_object = []
        self.ind_diversity = []

        self.frames = []
        self.sigma = []
        self.diversity = []

        if XY is not None:
            self.XY = XY

    def combine_frames(self):
        """
        Combine the frames from all objects and sequences into a single tensor.
        Observations with different diversity channels are concatenated along the frame axis.
        Returns:
        --------
        torch.Tensor: A tensor of shape (n_sequences, n_objects, n_frames, nx, ny) containing the combined frames.
        """

        self.logger.info(f"Setting up frames...")

        # Get number of objects and number of diversity channels from the added frames
        self.n_bursts = len(self.ind_object)
        self.n_o = max(self.ind_object) + 1

        n_seq, n_f, n_x, n_y = self.frames[0].shape
        
        frames = [None] * self.n_o
        plane = [None] * self.n_o
        diversity = [None] * self.n_o
        sigma = [None] * self.n_o
        index_frames_diversity = [None] * self.n_o

        # Count the number of frames per object, taking into account the diversity channels
        n_frames_per_object = [0] * self.n_o
        n_diversity_per_object = [0] * self.n_o
        for i in range(self.n_bursts):
            n_frames_per_object[self.ind_object[i]] += n_f
            n_diversity_per_object[self.ind_object[i]] += 1
        
        for i in range(self.n_o):
            frames[i] = torch.zeros(n_seq, n_frames_per_object[i], n_x, n_y)
            plane[i] = torch.zeros(n_seq, 1, n_x, n_y)
            diversity[i] = torch.zeros(n_seq, n_frames_per_object[i])
            sigma[i] = torch.zeros(n_seq, n_frames_per_object[i])
            index_frames_diversity[i] = [0] * n_diversity_per_object[i]

        sigma_max = 0.0
        for i in range(self.n_bursts):
            sigma_max = max(sigma_max, torch.max(self.sigma[i]))            

        
        for i in range(self.n_bursts):

            i_obj = self.ind_object[i]
            i_div = self.ind_diversity[i]

            f0 = i_div * n_f
            f1 = (i_div + 1) * n_f

            index_frames_diversity[i_obj][i_div] = f0

            frames[i_obj][:, f0:f1, :, :], subtract = util.apodize(self.frames[i], self.window, gradient=self.remove_gradient_apodization)
            if self.remove_gradient_apodization:
                plane[i_obj][:, :, :, :] = subtract
            
            # Set the diversity for the current object for all frames and for the sequence            
            diversity[i_obj][:, f0:f1] = self.diversity[i][:, None].expand(-1, n_f)
            
            sigma[i_obj][:, f0:f1] = self.sigma[i] #/ sigma_max
                    
        return frames, diversity, index_frames_diversity, sigma, plane
    
    def update_object(self, cutoffs=None):
        """
        Update the object estimate with new cutoffs in the Fourier filter.

        Parameters
        ----------
        cutoffs : list
            A list containing the new cutoffs for each object. Each cutoff contains two numbers, indicating the
            lower and upper frequencies for the transition.
        """

        if self.simultaneous_sequences is None:
            self.logger.info(f"Deconvolution has not been carried out yet")
            return
        
        if cutoffs is None:
            self.logger.info(f"No cutoffs provided")
            return
        
        # Recompute the diffraction masks with the new cutoffs
        self.logger.info('Recalculating object with new cutoffs in the Fourier filter...')

        for i in range(self.n_o):
            self.cutoff[i] = cutoffs[i]
            self.logger.info(f"     - Filter: {self.image_filter[i]} - cutoff: {self.cutoff[i]}...")            
            if self.loss_type == "joint":
                self.logger.info(f"     - s_u: {self.s_u[i]}")
        
        self.compute_diffraction_masks()

        n_seq, _, _, _ = self.frames_apodized[0].shape

        # Split sequences in batches
        ind = np.arange(n_seq)

        n_seq_total = n_seq

        # Split the sequences in groups of simultaneous sequences to be computed in parallel
        ind = np.array_split(ind, np.ceil(n_seq / self.simultaneous_sequences))

        n_sequences = len(ind)
        
        self.psf_seq = [None] * n_sequences        
        self.degraded_seq = [None] * n_sequences
        self.obj_seq = [None] * n_sequences
        self.obj_diffraction_seq = [None] * n_sequences
        
        for i_seq, seq in enumerate(ind):
                            
            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total}")

            frames_apodized_seq = []
            plane_seq = []
            frames_ft = []
            sigma_seq = []
            diversity_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...].to(self.device))
                plane_seq.append(self.plane[i][seq, ...].to(self.device))
                frames_ft.append(torch.fft.fft2(self.frames_apodized[i][seq, ...], norm=self.fft_norm).to(self.device))
                sigma_seq.append(self.sigma[i][seq, ...].to(self.device))
                diversity_seq.append(self.diversity[i][seq, ...].to(self.device))
                
            n_seq = len(seq)

            if self.psf_model.lower() in ['zernike', 'kl']:
                psf, psf_ft = self.compute_psfs(self.modes_seq[i_seq], diversity_seq, jitter=self.jitter_seq[i_seq] if self.use_jitter else None)
            
            if self.psf_model.lower() == 'nmf':
                psf, psf_ft = self.compute_psfs_nmf(self.modes_seq[i_seq])
            
            if (self.infer_object):

                # Compute filtered object from the current estimate
                if (self.config['optimization']['transform'] == 'softplus'):
                    obj_ft = torch.fft.fft2(F.softplus(obj), norm=self.fft_norm)
                else:
                    obj_ft = torch.fft.fft2(obj, norm=self.fft_norm)

                # Filter in Fourier
                obj_filter_ft = self.fft_filter(obj_ft)                

            else:                
                obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, psf_ft, sigma_seq, plane_seq, pars_s0=self.pars_s0_seq[i_seq] if self.loss_type == 'marginal' else None)  
                                   

            obj_filter_diffraction = [None] * self.n_o
            degraded = [None] * self.n_o
            for i in range(self.n_o):                
                obj_filter_diffraction[i] = torch.fft.ifft2(obj_filter_ft[i] * self.psf_diffraction_ft[i][None, :, :]).real
            
                # Compute final degraded images
                degraded_ft = obj_filter_ft[i][:, None, :, :] * psf_ft[i]
                degraded[i] = torch.fft.ifft2(degraded_ft).real
                        
            for i in range(self.n_o):
                psf[i] = psf[i].detach()
                degraded[i] = degraded[i].detach()
                obj_filter[i] = obj_filter[i].detach()
                obj_filter_diffraction[i] = obj_filter_diffraction[i].detach()

# There is a memory leak here!!!!!!!!!!!!!
            # self.psf_seq[i_seq] = psf
            # self.degraded_seq[i_seq] = degraded
            self.obj_seq[i_seq] = obj_filter
            self.obj_diffraction_seq[i_seq] = obj_filter_diffraction

            tfinal = time.time()

            # del psf, degraded, obj_filter, obj_filter_diffraction, degraded_ft, obj_ft, obj_filter_ft, psf_ft
        
        # Concatenate the results from all sequences and all objects independently
        # self.psf = [None] * self.n_o
        # self.degraded = [None] * self.n_o
        self.obj = [None] * self.n_o
        self.obj_diffraction = [None] * self.n_o

        # for i in range(self.n_o):
        self.modes = torch.cat(self.modes_seq, dim=0)
                
        for i in range(self.n_o):
            # tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            # self.psf[i] = torch.cat(tmp, dim=0)

            # tmp = [self.degraded_seq[j][i] for j in range(n_sequences)]
            # self.degraded[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_diffraction_seq[j][i] for j in range(n_sequences)]
            self.obj_diffraction[i] = torch.cat(tmp, dim=0)
        
        return 
    
    def write(self, filename, extra=None):
        """
        Write the deconvolved object to a file.
        Parameters:
        -----------
        filename : str
            The name of the file to which the object will be written.
        Returns:
        --------
        None
        """
        
        self.logger.info(f"Writing object to {filename}...")
        
        hdu = [fits.PrimaryHDU(self.modes.cpu().numpy())]
        for i in range(self.n_o):
            tmp = fits.ImageHDU(self.obj[i].cpu().numpy())
            tmp.header['OBJECT'] = f'Object {i+1}'
            if extra is not None:
                for k, v in extra.items():
                    tmp.header[k] = v
            hdu.append(tmp)
        
        hdu = fits.HDUList(hdu)
        hdu.writeto(filename, overwrite=True)

        return
            
    def deconvolve(self,                                    
                   simultaneous_sequences=1, 
                   infer_object=False, 
                   optimizer='adam', 
                   obj_in=None, 
                   modes_in=None,                    
                   n_iterations=20):
        

        """
        Perform deconvolution on a set of frames using specified parameters.
        Parameters:
        -----------
        frames : torch.Tensor
            The input frames to be deconvolved (n_sequences, n_objects, n_frames, nx, ny).
        sigma : torch.Tensor
            The noise standard deviation for each object.
        simultaneous_sequences : int, optional
            Number of sequences to be processed simultaneously (default is 1).
        infer_object : bool, optional
            Whether to infer the object during optimization (default is False).
        optimizer : str, optional
            The optimizer to use ('adam' for Adam, 'lbfgs' for LBFGS) (default is 'adam').
        obj_in : torch.Tensor, optional
            Initial object to use for deconvolution (default is None).
        modes_in : torch.Tensor, optional
            Initial modes to use for deconvolution (default is None).
        annealing : bool or str, optional
            Annealing schedule to use ('linear', 'sigmoid', 'none') (default is 'linear'').
        n_iterations : int, optional
            Number of iterations for the optimization (default is 20).        
        Returns:
        --------
        None
        """
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        self.simultaneous_sequences = simultaneous_sequences
        self.infer_object = infer_object

        _, self.n_f, self.n_x, self.n_y = self.frames[0].shape

        self.logger.info(f" *****************************************")
        self.logger.info(f" *** SPATIALLY INVARIANT DECONVOLUTION ***")
        self.logger.info(f" *****************************************")

        # Combine all frames        
        self.frames_apodized, self.diversity, self.init_frame_diversity, self.sigma, self.plane = self.combine_frames()

        # Define all basis
        self.define_basis()
        
        # Fill the list of frames and apodize the frames if needed
        # for i in range(self.n_o):
        #     self.frames_apodized[i] = self.frames_apodized[i].to(self.device)
        #     self.diversity[i] = self.diversity[i].to(self.device)
        #     self.sigma[i] = self.sigma[i].to(self.device)
            
        self.logger.info(f"Frames")        
        for i in range(self.n_o):
            n_seq, n_f, n_x, n_y = self.frames_apodized[i].shape
            self.logger.info(f"  * Object {i}")
            self.logger.info(f"     - Number of sequences {n_seq}...")
            self.logger.info(f"     - Number of frames {n_f}...")
            self.logger.info(f"     - Number of diversity channels {len(self.init_frame_diversity[i])}...")
            for j, ind in enumerate(self.init_frame_diversity[i]):
                self.logger.info(f"       -> Diversity {j} = {self.diversity[i][0, ind]} - Noise = {self.sigma[i][0, ind]}...")
            self.logger.info(f"     - Size of frames {n_x} x {n_y}...")
            self.logger.info(f"     - Filter: {self.image_filter[i]} - cutoff: {self.cutoff[i]}...")
            if self.loss_type == "joint":
                self.logger.info(f"     - s_u: {self.s_u[i]}")
                
        self.finite_difference = util.FiniteDifference().to(self.device)
        self.set_regularizations()
                                                    
        # Compute the diffraction masks
        self.compute_diffraction_masks()
        
        # Annealing schedules

        if self.psf_model.lower() in ['zernike', 'kl']:
            modes = np.cumsum(np.arange(2, self.noll_max+1))

        if self.psf_model.lower() == 'nmf':
            n = (self.n_modes - 2) // 5
            modes = np.linspace(2, self.n_modes, n).astype(int)
        
        self.anneal = self.compute_annealing(modes, n_iterations)
                
        # If the regularization parameter is a scalar, we assume that it is the same for all objects
        for reg in self.regularization:
            if reg.type == 'iuwt':                
                if not isinstance(reg.lambda_reg, list):
                    reg.lambda_reg = [reg.lambda_reg] * self.n_o

        # Hyperpriors
        if self.loss_type == 'marginal':
            self.K_prior = [self.config['priors']['K']['mean'], self.config['priors']['K']['sigma']]
            self.v0_prior = [self.config['priors']['v0']['mean'], self.config['priors']['v0']['sigma']]
            self.p_prior = [self.config['priors']['p']['mean'], self.config['priors']['p']['sigma']]

            self.logger.info(f"Normal hyperpriors parameters :")
            self.logger.info(f"  - K: mean = {self.K_prior[0]}, sigma = {self.K_prior[1]}")
            self.logger.info(f"  - v0: mean = {self.v0_prior[0]}, sigma = {self.v0_prior[1]}") 
            self.logger.info(f"  - p: mean = {self.p_prior[0]}, sigma = {self.p_prior[1]}")

        #--------------------------------
        # Start optimization
        #--------------------------------

        # Split sequences in batches
        ind = np.arange(n_seq)

        n_seq_total = n_seq

        # Split the sequences in groups of simultaneous sequences to be computed in parallel
        ind = np.array_split(ind, np.ceil(n_seq / simultaneous_sequences))

        n_sequences = len(ind)
        
        self.modes_seq = [None] * n_sequences
        self.pars_s0_seq = [None] * n_sequences
        self.jitter_seq = [None] * n_sequences
        self.loss = [None] * n_sequences

        self.psf_seq = [None] * n_sequences        
        self.degraded_seq = [None] * n_sequences
        self.obj_seq = [None] * n_sequences
        self.obj_diffraction_seq = [None] * n_sequences

        tinit = time.time()
        tinit_global = time.time()

        self.total_time_convergence = 0.0

        self.psf_diffraction, self.psf_diffraction_ft = self.compute_psf_diffraction()
        
        for i_seq, seq in enumerate(ind):
            
            if i_seq == 0:
                label_time = ''
            else:
                deltat = tfinal - tinit
                tinit = time.time()
                remaining = deltat * (n_sequences - i_seq)
                label_time = f" - Elapsed time {deltat:.2f} s - Remaining time {remaining:.2f} s ({remaining:.2f} s)"
                
            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total} {label_time}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total} {label_time}")

            frames_apodized_seq = []
            plane_seq = []
            frames_ft = []
            sigma_seq = []
            diversity_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...].to(self.device))
                plane_seq.append(self.plane[i][seq, ...].to(self.device))
                frames_ft.append(torch.fft.fft2(self.frames_apodized[i][seq, ...], norm=self.fft_norm).to(self.device))
                sigma_seq.append(self.sigma[i][seq, ...].to(self.device))
                diversity_seq.append(self.diversity[i][seq, ...].to(self.device))

            n_seq = len(seq)
                                                
            if (infer_object):
                
                obj_init = [None] * self.n_o

                # Find frame with best contrast
                for i in range(self.n_o):
                
                    contrast = torch.std(frames_apodized_seq[i], dim=(-1, -2)) / torch.mean(frames_apodized_seq[i], dim=(-1, -2)) * 100.0
                    ind = torch.argsort(contrast[0, :], descending=True)

                    if obj_in is not None:
                        self.logger.info(f"Using provided initial object...")
                        obj_init[i] = obj_in
                        obj_init[i] = obj_init.to(self.device)
                    else:                    
                        if self.config['initialization']['object'] == 'contrast':
                            self.logger.info(f"Selecting initial object as image with best contrast...")
                            obj_init[i] = frames_apodized_seq[i][:, ind[0], :, :]
                        if self.config['initialization']['object'] == 'average':
                            self.logger.info(f"Selecting initial object as average image...")
                            obj_init[i] = torch.mean(frames_apodized_seq, dim=1)
                    
                        # Initialize the object with the inverse softplus
                    if (self.config['optimization']['transform'] == 'softplus'):
                        obj_init[i] = torch.log(torch.exp(obj_init[i]) - 1.0)
            
            # Unknown modes
            if modes_in is not None:
                self.logger.info(f"Using provided initial modes...")
                modes = modes_in.clone().detach().to(self.device).requires_grad_(True)
            else:
                if self.psf_model.lower() in ['zernike', 'kl']:
                    if self.config['initialization']['modes_std'] == 0:
                        self.logger.info(f"Initializing modes with zeros...")
                        modes = torch.zeros((n_seq, self.n_f, self.n_modes), device=self.device, requires_grad=True)
                    else:
                        self.logger.info(f"Initializing modes with random values with standard deviation {self.config['initialization']['modes_std']}")
                        modes = self.config['initialization']['modes_std'] * torch.randn((n_seq, self.n_f, self.n_modes))
                        modes = modes.clone().detach().to(self.device).requires_grad_(True)
                
                if self.psf_model.lower() == 'nmf':                               
                    tmp, _ = optim.nnls(self.basis[0].reshape((self.n_modes, self.npix**2)).T.cpu().numpy(), self.psf_diffraction[0].flatten().cpu().numpy())
                    modes = torch.tensor(tmp[None, None, :].astype('float32')).expand((n_seq, self.n_f, self.n_modes))
                    modes = modes.clone().detach().to(self.device).requires_grad_(True)
                    
            if (infer_object):
                self.logger.info(f"Optimizing object and modes...")

                parameters = [{'params': modes, 'lr': self.lr_modes}]
                obj = [None] * self.n_o

                for i in range(self.n_o):
                    obj[i] = obj_init[i].clone().detach().to(self.device).requires_grad_(True)
                    parameters.append({'params': obj[i], 'lr': self.lr_obj})
                                    
            else:
                self.logger.info(f"Optimizing modes only...")

                parameters = [{'params': modes, 'lr': self.lr_modes}]

                if self.loss_type == 'marginal':
                    # S0(v) = k / (1 + (v/v0)^2)**(p/2)
                    # log k, log v0, p

                    # Initialize the parameters of the power spectral density of the object with some reasonable values
                    pars_s0 = np.zeros((n_seq, self.n_o, 4))

                    # We initialize sigma2 with the average of the noise variance across frames
                    for i in range(self.n_o):
                        pars_s0[:, i, 0] = np.log(1.0)
                        pars_s0[:, i, 1] = np.log(0.1) # log v0
                        pars_s0[:, i, 2] = np.log(2.0) #
                        pars_s0[:, i, 3] = np.log(sigma_seq[i].mean().item()**2)
                    self.pars_s0_out = np.zeros((n_seq, self.n_o, 4))
                    pars_s0_torch = torch.tensor(pars_s0.astype('float32')).to(self.device).requires_grad_(True)
                    self.pars_s0_out = torch.tensor(self.pars_s0_out.astype('float32')).to(self.device)
                    parameters.append({'params': pars_s0_torch, 'lr': self.lr_prior})

                if self.use_jitter:
                    jitter = np.zeros((n_seq, self.n_f, 3))
                    jitter[:, :, 0] = 0.0
                    jitter[:, :, 1] = 0.0
                    jitter[:, :, 2] = 0.01

                    jitter_torch = torch.tensor(jitter.astype('float32')).to(self.device).requires_grad_(True)
                    parameters.append({'params': jitter_torch, 'lr': self.lr_modes})

            # Second order optimizer
            if optimizer == 'lbfgs':
                self.logger.info(f"Using LBFGS optimizer...")
                opt = torch.optim.LBFGS(parameters, lr=0.01)
            if optimizer == 'adam':
                self.logger.info(f"Using Adam optimizer...")
                opt = torch.optim.Adam(parameters)
            if optimizer == 'adamw':
                self.logger.info(f"Using AdamW optimizer...")
                opt = torch.optim.AdamW(parameters)
            if optimizer == 'cg':
                if NGC_OPTIMIZER:
                    self.logger.info(f"Using CG optimizer...")
                    opt = ncg_optimizer.BASIC(parameters, method = 'CD', line_search = 'Strong_Wolfe', c1 = 1e-4, c2 = 0.9, lr = 1, rho = 0.5, eps=1e-8)
                else:
                    self.logger.info(f"Using Adam optimizer...")
                    opt = torch.optim.Adam(parameters)

            if optimizer not in ['lbfgs', 'adam', 'adamw', 'cg']:
                raise ValueError(f"Optimizer {optimizer} not supported")
            
            self.logger.info(f"Starting optimization with {n_iterations} iterations...")
            self.logger.info(f"Learning rates: modes {self.lr_modes} - object {self.lr_obj if infer_object else 'N/A'} - psd prior {self.lr_prior if not infer_object and self.loss_type == 'marginal' else 'N/A'}")
                
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3*n_iterations)

            losses = torch.zeros(n_iterations, device=self.device)

            t = tqdm(range(n_iterations))

            if self.psf_model.lower() in ['zernike', 'kl']:
                n_active = 2
            
            if self.psf_model.lower() == 'nmf':
                n_active = self.n_modes

            modes_previous = modes.clone().detach()

            self.t0_convergence = time.time()
                
            for loop in t:
                                
                def closure():
                            
                    opt.zero_grad(set_to_none=True)

                    if self.psf_model.lower() in ['zernike', 'kl']:
                
                        # Compute PSF from current wavefront coefficients and reference 
                        modes_centered = modes.clone()
                        modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - modes[:, 0:1, 0:2]
                                    
                        # modes -> (n_seq, n_f, self.n_modes)
                        # jitter > (n_seq, n_f, 3)
                        psf, psf_ft = self.compute_psfs(modes_centered[:, :, 0:n_active], diversity_seq, jitter_torch if self.use_jitter else None)
                    
                    if self.psf_model.lower() == 'nmf':
                        psf, psf_ft = self.compute_psfs_nmf(modes[:, :, 0:n_active])
                    
                    if (infer_object):

                        obj_ft = [None] * self.n_o
                        obj_filter_ft = [None] * self.n_o

                        loss_mse = torch.tensor(0.0).to(self.device)
                        
                        for i in range(self.n_o):
                            # Compute filtered object from the current estimate while also clamping negative values
                            if (self.config['optimization']['transform'] == 'softplus'):
                                tmp = torch.clamp(F.softplus(obj[i]), min=0.0)
                                obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                            else:
                                tmp = torch.clamp(obj[i], min=0.0)
                                obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                        
                        # Filter in Fourier
                        obj_filter_ft = self.fft_filter(obj_ft)

                        for i in range(self.n_o):

                            degraded_ft = obj_ft[i][:, None, :, :] * psf_ft[i]

                            # residual = self.weight[:, :, None, None, None] * (degraded_ft - frames_ft)
                            residual =  (degraded_ft - frames_ft[i])
                            loss_mse += torch.mean((residual * torch.conj(residual)).real) / self.npix**2

                    else:                        

                        if self.show_object_info:
                            obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, psf_ft, sigma_seq, plane_seq, pars_s0=pars_s0_torch if self.loss_type == 'marginal' else None)

                        # Compute the loss function using the appropriate filter for the loss
                        loss_data, loss_prior, loss = self.compute_loss(frames_ft, psf_ft, sigma_seq, pars_s0=pars_s0_torch if self.loss_type == 'marginal' else None)

                    # If MOMFBD is used, then the object cannot be regularized. Look for alternatives for the future
                    # Object regularization
                    loss_obj = torch.tensor(0.0).to(self.device)
                    for index in self.index_regularization['object']:
                        loss_obj += self.regularization[index](obj_filter)
                    
                    # Total loss
                    loss += loss_obj
                                                        
                    # Save some information for the progress bar
                    # self.loss_local = loss.detach()
                    # self.obj_filter = [None] * self.n_o
                    # for i in range(self.n_o):
                    #     self.obj_filter[i] = obj_filter[i].detach()
                    # self.loss_mse_local = loss_mse.detach()
                    # self.loss_obj_local = loss_obj.detach()

                    if optimizer == 'cg' and NGC_OPTIMIZER:
                        loss.backward(retain_graph=True)
                    else:
                        loss.backward()

                    self.loss_data_local = loss_data.detach()
                    self.loss_prior_local = loss_prior.detach()
                    self.loss_local = loss.detach()
                    self.loss_obj_local = loss_obj.detach()

                    if self.show_object_info:
                        self.obj_filter_local = obj_filter
                    
                    return loss
                        
                loss = opt.step(closure)

                loss_data = self.loss_data_local
                loss_prior = self.loss_prior_local
                loss = self.loss_local
                loss_obj = self.loss_obj_local
                
                if self.show_object_info:
                    obj_filter = self.obj_filter_local
                
                # scheduler.step()

                if self.handle is not None:
                    gpu_usage = f'{self.handle.gpu_utilization():03d}'
                    memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f}/{self.handle.memory_total() / 1024**2:4.1f} MB'
                    memory_pct = f'{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'
                                   
                tmp = OrderedDict()                
                
                if self.cuda:
                    tmp['gpu'] = f'{gpu_usage} %'
                    # tmp['mem'] = f'{memory_usage} ({memory_pct})'
                    tmp['mem'] = f'{memory_usage}'

                delta_modes = torch.mean(torch.abs(modes.detach() - modes_previous))

                modes_previous = modes.clone().detach()

                tmp['active'] = f'{n_active}'
                if self.show_object_info:
                    tmp['contrast'] = f'{torch.std(obj_filter[0]) / torch.mean(obj_filter[0]) * 100.0:7.4f}'
                    tmp['minmax'] = f'{torch.min(obj_filter[0]):7.4f}/{torch.max(obj_filter[0]):7.4f}'
                tmp['chg'] = f'{delta_modes.item():8.6f}'
                tmp['LOBJ'] = f'{loss_obj.detach().item():8.6f}'
                tmp['LDATA'] = f'{loss_data.detach().item():8.6f}'
                tmp['LPRIOR'] = f'{loss_prior.detach().item():8.6f}'
                if self.loss_type == 'marginal':
                    tmp['K'] = f'{self.pars_s0_avg[0].item() / self.npix:6.2f}'
                    tmp['v0'] = f'{self.pars_s0_avg[1].item():6.2f}'
                    tmp['p'] = f'{self.pars_s0_avg[2].item():6.2f}'
                    tmp['sig'] = f'{np.sqrt(self.pars_s0_avg[3].item()):6.3f}'
                if self.use_jitter:
                    tmp['sx'] = f'{torch.mean(torch.exp(jitter_torch[:, :, 0])).item():6.3f}'
                    tmp['sy'] = f'{torch.mean(torch.exp(jitter_torch[:, :, 1])).item():6.3f}'
                    tmp['rxy'] = f'{torch.mean(torch.tanh(jitter_torch[:, :, 2])).item():6.3f}'
                tmp['L'] = f'{loss.detach().item():8.6f}'
                t.set_postfix(ordered_dict=tmp)

                n_active = self.anneal[loop]

            self.tf_convergence = time.time()

            self.total_time_convergence += self.tf_convergence - self.t0_convergence
            
            if self.psf_model.lower() in ['zernike', 'kl']:
                modes_centered = modes.clone().detach()
                modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - modes_centered[:, 0:1, 0:2]
                psf, psf_ft = self.compute_psfs(modes_centered, diversity_seq, jitter=jitter_torch if self.use_jitter else None)
            
            if self.psf_model.lower() == 'nmf':
                psf, psf_ft = self.compute_psfs_nmf(modes)

            
            if (infer_object):
                
                # Compute filtered object from the current estimate
                obj_ft = [None] * self.n_o
                obj_filter_ft = [None] * self.n_o
                obj_filter = [None] * self.n_o
                
                for i in range(self.n_o):
                    # Compute filtered object from the current estimate while also clamping negative values
                    if (self.config['optimization']['transform'] == 'softplus'):
                        tmp = torch.clamp(F.softplus(obj[i]), min=0.0)
                        obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                    else:
                        tmp = torch.clamp(obj[i], min=0.0)
                        obj_ft[i] = torch.fft.fft2(tmp, norm=self.fft_norm)
                
                # Filter in Fourier
                obj_filter_ft = self.fft_filter(obj_ft)                

                for i in range(self.n_o):
                    obj_filter[i] = torch.fft.ifft2(obj_filter_ft[i]).real

            else:
                obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, psf_ft, sigma_seq, plane_seq, pars_s0=pars_s0_torch if self.loss_type == 'marginal' else None)
                                   

            obj_filter_diffraction = [None] * self.n_o
            degraded = [None] * self.n_o
            for i in range(self.n_o):                
                obj_filter_diffraction[i] = torch.fft.ifft2(obj_filter_ft[i] * self.psf_diffraction_ft[i][None, :, :]).real
            
                # Compute final degraded images
                degraded_ft = obj_filter_ft[i][:, None, :, :] * psf_ft[i]
                degraded[i] = torch.fft.ifft2(degraded_ft).real
            
            # Store the results for the current set of sequences
            self.modes_seq[i_seq] = modes.detach()
            self.jitter_seq[i_seq] = jitter_torch.detach() if self.use_jitter else None
            self.pars_s0_seq[i_seq] = self.pars_s0_out if not infer_object and self.loss_type == 'marginal' else None
            self.loss[i_seq] = losses.detach()

            for i in range(self.n_o):
                psf[i] = psf[i].detach()
                degraded[i] = degraded[i].detach()
                obj_filter[i] = obj_filter[i].detach()
                obj_filter_diffraction[i] = obj_filter_diffraction[i].detach()

# There is a memory leak here!!!!!!!!!!!!!
            # self.psf_seq[i_seq] = psf
            # self.degraded_seq[i_seq] = degraded
            self.obj_seq[i_seq] = obj_filter
            self.obj_diffraction_seq[i_seq] = obj_filter_diffraction

            tfinal = time.time()

            # del psf, degraded, obj_filter, obj_filter_diffraction, degraded_ft, obj_ft, obj_filter_ft, psf_ft
        
        deltat = tfinal - tinit
        deltat_global = tfinal - tinit_global        
        self.total_time = deltat_global        
        self.logger.info(f"Elapsed time {deltat:.2f} s - Total time: {deltat_global:.2f} s")

        # Concatenate the results from all sequences and all objects independently
        # self.psf = [None] * self.n_o
        # self.degraded = [None] * self.n_o
        self.obj = [None] * self.n_o
        self.obj_diffraction = [None] * self.n_o
        self.pars_s0 = [None] * self.n_o        

        # for i in range(self.n_o):
        self.modes = torch.cat(self.modes_seq, dim=0)
        self.loss = torch.cat(self.loss, dim=0)
        self.jitter = torch.cat(self.jitter_seq, dim=0) if self.use_jitter else None
                
        for i in range(self.n_o):
            # tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            # self.psf[i] = torch.cat(tmp, dim=0)

            # tmp = [self.degraded_seq[j][i] for j in range(n_sequences)]
            # self.degraded[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_diffraction_seq[j][i] for j in range(n_sequences)]
            self.obj_diffraction[i] = torch.cat(tmp, dim=0)
            
            if self.loss_type == 'marginal':
                tmp = [self.pars_s0_seq[j][:, i, :] for j in range(n_sequences)]
                if len(tmp) > 0:
                    self.pars_s0[i] = torch.cat(tmp, dim=0)
                else:
                    self.pars_s0[i] = None
                    
        return 
    
    
if __name__ == '__main__':
    pass
