import numpy as np
import scipy.optimize as opt
import torchmfbd.util as util
import torch
import torch.nn.functional as F
from torchmfbd.deconvolution import Deconvolution
from torchmfbd.warp import warp
from tqdm import tqdm
from einops import rearrange
from collections import OrderedDict

class DeconvolutionSV(Deconvolution):
    def __init__(self, config):
                
        super().__init__(config)
        
        # self.ngrid_modes = config['ngrid_modes']
        self.basis_file = self.config['psf']['filename']
        self.psf_model = self.config['psf']['model']
        self.ngrid_modes = self.config['psf']['ngrid_modes']

        # PSF model parameterized with the PSF
        if (self.psf_model.lower() in ['pca', 'nmf']):

            self.logger.info(f"PSF model: linear direct expansion")

            if (self.psf_model.lower() not in ['pca', 'nmf']):
                raise Exception(f"Unknown basis {self.basis}")
            
            # PSF modes       
            self.logger.info(f"  * Using {self.psf_model.lower()} modes")
            self.logger.info(f"     - Reading PSF modes from {self.basis_file}...")
            f = np.load(self.basis_file)

            # Extract wavelength, diameter, pixel size, and central obscuration from the file with the modes
            if self.config['images']['wavelength'] != f['info'][0]:
                raise Exception(f"Wavelength mismatch: {self.config['images']['wavelength']} vs {f['info'][0]}")
            if self.config['telescope']['diameter'] != f['info'][1]:
                raise Exception(f"Diameter mismatch: {self.config['telescope']['diameter']} vs {f['info'][1]}")
            if self.config['images']['pix_size'] != f['info'][2]:
                raise Exception(f"Pixel size mismatch: {self.config['images']['pix_size']} vs {f['info'][2]}")
            if self.config['telescope']['central_obscuration'] != f['info'][3]:
                raise Exception(f"Central obscuration mismatch: {self.config['telescope']['central_obscuration']} vs {f['info'][3]}")

            n_psf = int(np.sqrt(f['basis'].shape[1]))
            self.logger.info(f"  * Size of basis PSF modes {n_psf}x{n_psf}...")        
            self.logger.info(f"  * Number of modes {self.n_modes}...")            
            
            self.psf_modes = f['basis'][0:self.n_modes, :].reshape((self.n_modes, n_psf, n_psf))

            # Pad diffraction PSF to the size of the image and shift
            self.psf_diffraction = f['psf_diffraction']        
            
            if (self.psf_model.lower() == 'pca'):
                self.logger.info(f"Computing PCA coefficients of diffraction PSF")
                self.coeffs_diffraction = np.sum(self.psf_modes * self.psf_diffraction[None, ...], axis=(1, 2))

            if (self.psf_model.lower() == 'nmf'):
                # Normalize the modes to unit area            
                self.psf_modes /= np.sum(self.psf_modes, axis=(1, 2), keepdims=True)

                self.logger.info(f"Computing NMF coefficients of diffraction PSF")            
                                        
                self.coeffs_diffraction, _ = opt.nnls(self.psf_modes.reshape((self.n_modes, n_psf**2)).T, self.psf_diffraction.flatten())
                
                # Since the diffraction PSF is already normalized to unit area, the sum of the inferred coefficients must be 1
                # We renormalize the coefficients to ensure that
                self.coeffs_diffraction /= np.sum(self.coeffs_diffraction)
                            
                min_coeff = np.min(self.coeffs_diffraction[self.coeffs_diffraction > 0])
                ind = np.where(self.coeffs_diffraction == 0)[0]            
                self.modes_init = np.copy(self.coeffs_diffraction)
                self.modes_init[ind] = 0.1 * min_coeff * np.random.rand(len(ind))            
        else:
            raise Exception(f"Spatially variant deconvolution cannot be used with PSF model {self.psf_model}")
                          
                    
        if (self.ngrid_modes != self.npix):
            self.logger.info(f"  * Using tiptilt and modes of size {self.ngrid_modes}x{self.ngrid_modes} and bilinear interpolation")
        else:
            self.logger.info(f"  * Using tiptilt and modes of size {self.ngrid_modes}x{self.ngrid_modes}")        

        # We use self.psf_modes to refer to both the PCA modes or the NMF modes
        # Pad the PSF modes to the size of the image
        pad = (self.npix - n_psf) // 2
        self.psf_modes = np.pad(self.psf_modes, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
                
        # Shift the PCA to later compute FFTs
        self.psf_modes = np.fft.ifftshift(self.psf_modes, axes=(1, 2))

        self.psf_diffraction = np.pad(self.psf_diffraction, ((pad, pad), (pad, pad)), mode='constant') 
        self.psf_diffraction = np.fft.ifftshift(self.psf_diffraction) 
                                                            
        # Move all relevant quantities to GPU
        self.psf_modes = torch.tensor(self.psf_modes.astype('float32')).to(self.device)
        self.psf_modes_fft = torch.fft.fft2(self.psf_modes)                            
        self.modes_init = torch.tensor(self.modes_init.astype('float32'))

        # Learning rates
        self.lr_tiptilt = self.config['optimization']['lr_tt']
        self.logger.info(f"Optimization")
        self.logger.info(f"  * lr_obj  ={self.lr_obj}")
        self.logger.info(f"  * lr_tt   ={self.lr_tiptilt}")
        self.logger.info(f"  * lr_modes={self.lr_modes}")
        self.logger.info(f"  * Scale of softplus {self.config['optimization']['softplus_scale']}...")
        
    def define_basis(self):

        self.rho = [None] * self.n_o
        self.diffraction_limit = [None] * self.n_o
        self.cutoff = [None] * self.n_o
        self.image_filter = [None] * self.n_o

        # First locate unique wavelengths. We will use the same basis for the same wavelength
        ind_wavelengths = []
        unique_wavelengths = []

        for i in range(self.n_o):
            self.cutoff[i] = self.config[f'object{i+1}']['cutoff']
            self.image_filter[i] = self.config[f'object{i+1}']['image_filter']
            w = self.config[f'object{i+1}']['wavelength']
            if w not in unique_wavelengths:
                unique_wavelengths.append(w)
            
            ind_wavelengths.append(unique_wavelengths.index(w))

        # Now iterate over all unique wavelengths and associate the basis
        # to the corresponding object
        for i in range(len(unique_wavelengths)):

            wavelength = unique_wavelengths[i]
                    
            # Compute the diffraction limit and the frequency grid
            cutoff = self.config['telescope']['diameter'] / (wavelength * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.npix, d=self.config['images']['pix_size']) / cutoff
            
            xx, yy = np.meshgrid(freq, freq)
            rho = np.sqrt(xx ** 2 + yy ** 2)

            diffraction_limit = wavelength * 1e-8 / self.config['telescope']['diameter'] * 206265.0

            self.logger.info(f"Wavelength {i} ({wavelength} A)")
            self.logger.info(f"  * Diffraction: {diffraction_limit} arcsec")
            self.logger.info(f"  * Diffraction (x1.22): {1.22 * diffraction_limit} arcsec")

            for j in range(self.n_o):
                if ind_wavelengths[j] == i:
                    self.rho[j] = rho
                    self.diffraction_limit[j] = diffraction_limit                

        return

    def fft_filter_image(self, obj, mask_diffraction):
        """
        Filter the object in Fourier space
        It simply multiplies the FFT of the object (properly apodized) by the Fourier filter
        and returns the inverse FFT of the result. This can be used to avoid structures above
        the diffraction limit.
        """
        
        # Apodize the object
        
        mean_val = torch.mean(obj, dim=(-1, -2), keepdims=True)
        obj1 = obj - mean_val        
        obj2 = obj1 * self.window[None, :, :]
        obj3 = obj2 + mean_val

        del obj1, obj2

        # Apply the mask
        objf1 = torch.fft.fft2(obj3)        
        objf2 = objf1 * mask_diffraction[None, ...]
        obj = torch.fft.ifft2(objf2).real

        del objf1, objf2

        return obj
    
    def active_annealing(self, iter):
        if (iter < self.config['annealing']['start_pct'] * self.config['gradient_steps']):
            y = 2
        elif (iter > self.config['annealing']['end_pct'] * self.config['gradient_steps']):
            y = self.config['n_modes']
        else:
            x0 = self.config['annealing']['start_pct'] * self.config['gradient_steps']
            x1 = self.config['annealing']['end_pct'] * self.config['gradient_steps']
            y0 = 2
            y1 = self.config['n_modes']
            y = np.clip((y1 - y0) / (x1 - x0) * (iter - x0) + y0, y0, y1)
        
        return int(y)
    
    def compute_syn(self, im, obj_filtered, tiptilt_infer, modes_infer, infer_tiptilt, infer_modes, i_o):
        """
        Compute the synthetic image based on the inferred object, tip-tilt, and modes.

        Parameters:
        im (torch.Tensor): Input image tensor of shape (ns, no, nf, nx, ny).
        obj_infer (torch.Tensor): Inferred object tensor.
        tiptilt_infer (torch.Tensor): Inferred tip-tilt tensor.
        modes_infer (torch.Tensor): Inferred modes tensor.
        infer_modes (bool): Flag indicating whether to infer modes or simply apply tip-tilt.

        Returns:
        torch.Tensor: Synthetic image tensor.
        """

        ns, nf, nx, ny = im.shape

        # Interpolate the tiptilt and modes to the appropriate sizes in case they
        # are defined in a coarser grid
        if (self.ngrid_modes != self.npix):                
            tmp = rearrange(tiptilt_infer, 'b nf d nx ny -> (b nf) d nx ny')
            tiptilt = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
            tiptilt = rearrange(tiptilt, '(b nf) d nx ny -> b nf d nx ny', b=ns)

            del tmp
            
            tmp = rearrange(modes_infer, 'b nf nm nx ny -> (b nf) nm nx ny')
            modes = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
            modes = rearrange(modes, '(b nf) nm nx ny -> b nf nm nx ny', b=ns)

            del tmp        
        else:
            tiptilt = tiptilt_infer
            modes = modes_infer
                
        # Apply tip-tilt to the object by applying a warp with the pixel-dependent optical flow
        if infer_tiptilt:
            tmp_obj = obj_filtered[:, None, None, :, :].expand(ns, nf, 1, self.npix, self.npix)
            tmp_obj = rearrange(tmp_obj, 'b nf no nx ny -> (b nf) no nx ny')
            tmp_tt = rearrange(tiptilt, 'b nf d nx ny -> (b nf) d nx ny')
            obj_tt = warp(tmp_obj, tmp_tt)
            obj_tt = rearrange(obj_tt, '(b nf) no nx ny -> b no nf nx ny', b=ns)

            del tmp_obj, tmp_tt
        else:
            obj_tt = obj_filtered[:, :, None, :, :].expand(1, no, nf, self.npix, self.npix)
            
        # Now apply the effect of the high order modes
        if infer_modes:
            
            # If we are using NMF, we need to force the coefficients to be non-negative
            # and also normalize by their sum to force the PSF to have unit area
            if self.psf_model.lower() == 'nmf':                
                # modes = F.relu(modes, inplace=False)
                modes = F.softplus(self.config['optimization']['softplus_scale'] * modes) / self.config['optimization']['softplus_scale']
                                
                modes_unitarea = modes / torch.sum(modes, dim=2, keepdims=True)
                
                del modes
            
            # Compute the product of the object and the modes
            obj_tt = obj_tt[:, :, :, None, :, :] * modes_unitarea[:, None, :, :, :, :]
            
            # Apply the apodization window                            
            mn = torch.mean(obj_tt, dim=(-1, -2), keepdims=True)
            obj_tt -= mn
            obj_tt *= self.window[None, None, None, None, :, :]
            obj_tt += mn

            # Compute convolution of current object with the PCA modes                               
            obj_tt_modes_fft = torch.fft.fft2(obj_tt)
            obj_convolved_with_psf_modes = torch.fft.ifft2(obj_tt_modes_fft * self.psf_modes_fft[None, None, None, ...]).real
            
            # Now sum convolved objects weighted by the modes
            syn = torch.sum(obj_convolved_with_psf_modes, dim=3)

            del obj_convolved_with_psf_modes, obj_tt_modes_fft
            
            # In the case of the PCA basis, we need to normalize the modes by
            # using the convolution of the modes with the PCA basis

            if self.psf_model.lower() == 'pca':
                # Normalization                
                # Apply the apodization window                            
                mn = torch.mean(modes, dim=(-1, -2), keepdims=True)
                modes_norm = modes - mn
                modes_norm *= self.window[None, None, None, :, :]
                modes_norm += mn
                
                # Compute convolution of current object with the PCA modes                               
                modes_norm_fft = torch.fft.fft2(modes_norm)
                norm = torch.fft.ifft2(modes_norm_fft * self.psf_modes_fft[None, None, None, ...]).real
                norm = torch.sum(norm, dim=3)
                        
                syn = syn / norm

                del norm
                del modes_norm_fft, modes_norm            

        else:
            syn = obj_tt
            modes_unitarea = None
        
        return syn[:, 0, ...], tiptilt, modes_unitarea
                            
    def deconvolve(self,                   
                   simultaneous_sequences=1,
                   infer_tiptilt=True, 
                   infer_modes=True, 
                   tiptilt_init=None,
                   n_iterations=20,
                   batch_size=64):
        
        """
        Perform spatially variant deconvolution on a set of frames.
        
        Parameters:
        -----------
        simultaneous_sequences : int, optional
            The number of sequences to process simultaneously. Default is 1.
        infer_tiptilt : bool, optional
            Whether to infer tip-tilt. Default is True.
        infer_modes : bool, optional
            Whether to infer modes. Default is True.
        tiptilt_init : torch.Tensor, optional
            Initial tip-tilt values. Default is None.
        n_iterations : int, optional
            Number of iterations for the optimization. Default is 20.
        batch_size : int, optional
            Batch size for processing frames. Default is 64.
        
        Returns:
        --------
        obj : numpy.ndarray
            The deconvolved object.
        tiptilt_infer : numpy.ndarray
            The inferred tip-tilt values.
        tiptilt : numpy.ndarray
            The final tip-tilt values.
        modes_infer : numpy.ndarray
            The inferred modes.
        modes : numpy.ndarray
            The final modes.
        coeffs_diffraction : numpy.ndarray
            The coefficients of diffraction.
        losses : list
            The list of loss values during optimization.
        psf_modes : numpy.ndarray
            The point spread function modes.
        """
        
        _, self.n_f, self.n_x, self.n_y = self.frames[0].shape

        self.logger.info(f" ***************************************")
        self.logger.info(f" *** SPATIALLY VARIANT DECONVOLUTION ***")
        self.logger.info(f" ***************************************")

        # Combine all frames
        self.frames_apodized, self.diversity, self.init_frame_diversity, self.sigma = self.combine_frames()

        # Define all basis
        self.define_basis()

        # Fill the list of frames and apodize the frames if needed
        for i in range(self.n_o):
            self.frames_apodized[i] = self.frames_apodized[i].to(self.device)
            self.diversity[i] = self.diversity[i].to(self.device)
            self.sigma[i] = self.sigma[i].to(self.device)
            
        self.logger.info(f"Frames")        
        for i in range(self.n_o):
            n_seq, n_f, n_x, n_y = self.frames_apodized[i].shape
            self.logger.info(f"  * Object {i}")            
            self.logger.info(f"     - Number of sequences {n_seq}...")
            self.logger.info(f"     - Number of frames {n_f}...")
            self.logger.info(f"     - Number of diversity channels {len(self.diversity[i])}...")
            for j, ind in enumerate(self.init_frame_diversity[i]):
                self.logger.info(f"       -> Diversity {j} = {self.diversity[i][ind]}...")
            self.logger.info(f"     - Size of frames {n_x} x {n_y}...")
                
        self.finite_difference = util.FiniteDifference().to(self.device)
        self.set_regularizations()

        # Compute the diffraction masks
        self.compute_diffraction_masks()
        
        # Annealing schedules        
        modes = np.arange(self.n_modes)
        self.anneal = self.compute_annealing(modes, n_iterations)

        # If the regularization parameter is a scalar, we assume that it is the same for all objects
        for reg in self.regularization:
            if reg.type == 'iuwt':                
                if not isinstance(reg.lambda_reg, list):
                    reg.lambda_reg = [reg.lambda_reg] * self.n_o


        self.batch_size = batch_size
        
        # self.sigma = sigma
        # self.weight = 1.0 / sigma
      
        
        #--------------------------------
        # Start optimization
        #--------------------------------

        # Split sequences in batches
        ind = np.arange(n_seq)

        n_seq_total = n_seq

        # Split the sequences in groups of simultaneous sequences to be computed in parallel
        ind = np.array_split(ind, np.ceil(n_seq / simultaneous_sequences))        

        n_sequences = len(ind)

        self.modes_lr_seq = [None] * n_sequences
        self.modes_hr_seq = [None] * n_sequences
        self.tiptilt_lr_seq = [None] * n_sequences
        self.tiptilt_hr_seq = [None] * n_sequences
        self.loss = [None] * n_sequences        
        self.obj_seq = [None] * n_sequences
                
        for i_seq, seq in enumerate(ind):

            if len(seq) > 1:
                self.logger.info(f"Processing sequences [{seq[0]+1},{seq[-1]+1}]/{n_seq_total}")
            else:
                self.logger.info(f"Processing sequence {seq[0]+1}/{n_seq_total}")

            frames_apodized_seq = []
            frames_ft = []
            sigma_seq = []
            weight_seq = []
            for i in range(self.n_o):
                frames_apodized_seq.append(self.frames_apodized[i][seq, ...])
                frames_ft.append(torch.fft.fft2(self.frames_apodized[i][seq, ...]))
                sigma_seq.append(self.sigma[i][seq, ...])
                weight_seq.append(1.0 / self.sigma[i][seq, ...])
                        
            # Initial estimation of the object: average of all frames
            if self.config['initialization']['object'].lower() == 'average':
                self.logger.info(f"Using the mean of the frames as initialization")
                obj = []
                for i in range(self.n_o):
                    obj.append(torch.mean(frames_apodized_seq[i], dim=1))
                
            # Select the frame with the highest contrast
            if self.config['initialization']['object'].lower() == 'contrast':
                self.logger.info(f"Using the frame with highest contrast as initialization")

                obj = []
                for i in range(self.n_o):
                    contrast = torch.std(frames_apodized_seq[i], dim=(-1, -2)) / torch.mean(frames_apodized_seq[i], dim=(-1, -2))
                    obj.append(frames_apodized_seq[i][:, torch.argmax(contrast), ...])
                                            
            
            if infer_tiptilt and infer_modes:
                self.logger.info("Optimizing object, tip-tilt, and modes...")
            if infer_tiptilt and not infer_modes:
                self.logger.info("Optimizing object and tip-tilt...")
                
            # Initialization of the tip-tilt            
            if tiptilt_init is None:
                tiptilt = torch.zeros((n_seq, self.n_f, 2, self.ngrid_modes, self.ngrid_modes))
            else:
                tiptilt = tiptilt_init[seq, ...]
            
            # Initialization of the modes
            modes = torch.ones((n_seq, self.n_f, self.n_modes, self.ngrid_modes, self.ngrid_modes))        
            modes *= self.modes_init[None, None, :, None, None]
            
            # Variables to be optimized
            obj_infer = []
            for i in range(self.n_o):
                obj_infer.append(obj[i].clone().detach().to(self.device).requires_grad_(True))
            modes_infer = modes.clone().detach().to(self.device).requires_grad_(True)
            tiptilt_infer = tiptilt.clone().detach().to(self.device).requires_grad_(True)
            
            # Parameters to be optimized
            parameters = [{'params': obj_infer, 'lr': self.lr_obj}, {'params': modes_infer, 'lr': self.lr_modes}, {'params': tiptilt_infer, 'lr': self.lr_tiptilt}]
            
            # Define the optimizer and the scheduler
            optimizer = torch.optim.AdamW(parameters)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 3*n_iterations)            
            
            self.logger.info("Starting optimization...")

            self.loss = []
        
            t = tqdm(range(n_iterations))
                    
            for loop in t:
                                    
                optimizer.zero_grad(set_to_none=True)
            
                # Partition the frames in batches
                indices = np.arange(self.n_f)
                
                indices = np.array_split(indices, np.ceil(self.n_f / self.batch_size))               

                # Accumulate the MSE loss for all batches
                loss_mse = torch.tensor(0.0).to(self.device)
                loss_tiptilt = torch.tensor(0.0).to(self.device)
                loss_modes = torch.tensor(0.0).to(self.device)
                loss_obj = torch.tensor(0.0).to(self.device)

                obj_filtered = [None] * self.n_o

                for i in range(self.n_o):

                    # Filter the object in the Fourier domain
                    if self.image_filter[i] == 'tophat':                        
                        obj_filtered[i] = self.fft_filter_image(obj_infer[i], self.mask_diffraction_th[i])
                    else:
                        obj_filtered[i] = obj_infer
                    
                    for j, ind in enumerate(indices):
                                                
                        # Compute the synthetic images taking into account the tip-tilt and modes
                        syn, tiptilt, modes = self.compute_syn(frames_apodized_seq[i][:, ind, :, :], 
                                                                             obj_filtered[i], 
                                                                             tiptilt_infer[:, ind, ...],
                                                                             modes_infer[:, ind, ...],
                                                                             infer_tiptilt, 
                                                                             infer_modes,
                                                                             i)
                                                
                        # Compute the MSE loss                        
                        loss_mse += torch.mean(weight_seq[i][:, ind, None, None] * (syn - frames_apodized_seq[i][:, ind, :, :])**2)
                
                # Regularization
                # Tip-tilt regularization                
                for index in self.index_regularization['tiptilt']:                    
                    loss_tiptilt += self.regularization[index](tiptilt_infer)

                # Modes regularization                
                for index in self.index_regularization['modes']:
                    loss_modes += self.regularization[index](modes_infer)

                # Object regularization
                for index in self.index_regularization['object']:
                    loss_obj += self.regularization[index](obj_filtered)
                
                # Total loss
                loss = loss_mse + loss_tiptilt + loss_modes + loss_obj

                # Backward pass
                loss.backward()
                
                # Update the parameters
                optimizer.step()
                
                # Update the learning rate
                scheduler.step()
                
                
                # Do some printing
                gpu_usage = f'{self.handle.gpu_utilization()}'            
                memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f}/{self.handle.memory_total() / 1024**2:4.1f} MB'
                memory_pct = f'{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'
                
                current_lr_obj = optimizer.param_groups[0]['lr']
                current_lr_modes = optimizer.param_groups[1]['lr']

                tmp = OrderedDict()
                if self.cuda:
                    tmp['gpu'] = f'{gpu_usage} %'
                    tmp['mem'] = f'{memory_usage} ({memory_pct})'
                tmp['lrm'] = f'{current_lr_modes:8.6f}'
                tmp['lro'] = f'{current_lr_obj:8.6f}'                
                tmp['contrast'] = f'{torch.std(obj_filtered[0]) / torch.mean(obj_filtered[0]) * 100.0:7.4f}'
                tmp['minmax'] = f'{torch.min(obj_filtered[0]):7.4f}/{torch.max(obj_filtered[0]):7.4f}'
                tmp['LMSE'] = f'{loss_mse.item():8.6f}'
                tmp['LTT'] = f'{loss_tiptilt.item():8.6f}'
                tmp['LMOD'] = f'{loss_modes.item():8.6f}'
                tmp['LOBJ'] = f'{loss_obj.item():8.6f}'
                tmp['L'] = f'{loss.item():8.6f}'

                t.set_postfix(ordered_dict = tmp)

                self.loss.append(loss)

            if self.psf_model.lower() == 'nmf':
                modes_infer = F.softplus(self.config['optimization']['softplus_scale'] * modes_infer) / self.config['optimization']['softplus_scale']
                # modes_infer = F.relu(modes_infer, inplace=False)
                modes_infer /= torch.sum(modes_infer, dim=2, keepdims=True)

            # Final interpolation of the tiptilt and modes to the appropriate sizes
            if (self.ngrid_modes != self.npix):                
                tmp = rearrange(tiptilt_infer, 'b nf d nx ny -> (b nf) d nx ny')
                tiptilt = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
                tiptilt = rearrange(tiptilt, '(b nf) d nx ny -> b nf d nx ny', b=n_seq)

                del tmp

                tmp = rearrange(modes_infer, 'b nf nm nx ny -> (b nf) nm nx ny')
                modes = F.interpolate(tmp, size=(self.npix, self.npix), mode='bilinear')
                modes = rearrange(modes, '(b nf) nm nx ny -> b nf nm nx ny', b=n_seq)

                del tmp
            else:
                tiptilt = tiptilt_infer
                modes = modes_infer

            # Denormalize the object and the frames and move to the CPU                    
            self.obj_seq[i_seq] = [obj_filtered[i].detach() for i in range(self.n_o)]                                    
            self.tiptilt_lr_seq[i_seq] = tiptilt_infer.detach()
            self.tiptilt_hr_seq[i_seq] = tiptilt.detach()
            self.modes_lr_seq[i_seq] = modes_infer.detach()
            self.modes_hr_seq[i_seq] = modes.detach()
                                    
        self.tiptilt_lr = torch.cat(self.tiptilt_lr_seq, dim=0)
        self.tiptilt_hr = torch.cat(self.tiptilt_hr_seq, dim=0)
        self.modes_lr = torch.cat(self.modes_lr_seq, dim=0)
        self.modes_hr = torch.cat(self.modes_hr_seq, dim=0)
        
        self.obj = [None] * self.n_o

        for i in range(self.n_o):
            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)        
            
        return
    
if __name__ == '__main__':
    pass