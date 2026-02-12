import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import matplotlib.pyplot as pl
import torchmfbd.zern as zern
import torchmfbd.util as util
from collections import OrderedDict
from tqdm import tqdm
from skimage.morphology import flood
import scipy.ndimage as nd
from nvitop import Device
from torchmfbd.deconvolution import Deconvolution
import logging
import torchmfbd.kl_modes as kl_modes
import torchmfbd.noise as noise
from torchmfbd.reg_smooth import RegularizationSmooth
from torchmfbd.reg_iuwt import RegularizationIUWT
import glob
import pathlib
import yaml
import torchmfbd.configuration as configuration
import torchmfbd.projection as projection

class DeconvolutionMetapupils(Deconvolution):
    def __init__(self, config):
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
        super().__init__(config, add_piston=True)

        self.XY = []

        if 'atmosphere' not in self.config:
            raise ValueError("atmosphere is mandatory in metapupils")
        self.heights = self.config['atmosphere']['heights']

        self.n_modes_height = self.n_modes
                            
    def compute_psfs(self, modes, n_active, M, diversity):
        """
        Compute the Point Spread Functions (PSFs) from the given modes.
        Parameters:
        modes (torch.Tensor): A tensor of shape (nheights, nframes, nmodes) representing the modes.
        M (torch.Tensor): A tensor of shape (nseq, nheights, nmodes, npix, npix) representing the projection matrices.
        Returns:
        tuple: A tuple containing:
            - wavefront (torch.Tensor): The computed wavefronts from the estimated modes.
            - psf_norm (torch.Tensor): The normalized PSFs.
            - psf_ft (torch.Tensor): The FFT of the normalized PSFs.
        """
                
        n_f = modes[0].shape[0]
                                        
        psf_norm = [None] * self.n_o
        psf_ft = [None] * self.n_o        
                
        for i in range(self.n_o):

            # Compute the projection of the metapupils on the footprints at all heights by multiplying by the
            # projection matrix and sum over all heights                  
                        
            for j in range(self.n_heights):
                # if j == 0:                    
                    # modes_directions = torch.einsum('ijk,mj->imk', M[i][j][:, 0:n_active, 0:n_active], modes[j][:, 0:n_active])
                # else:
                    # modes_directions += torch.einsum('ijk,mj->imk', M[i][j][:, 0:n_active, 0:n_active], modes[j][:, 0:n_active])

                n_active_height = min(n_active, self.n_modes_height[j])
                
                if j == 0:
                    modes_directions = torch.einsum('ijk,mj->imk', M[i][j][:, 0:n_active_height, 0:n_active_height], modes[j][:, 0:n_active_height])
                    
                    # Compute wavefronts from estimated modes                                
                    wavefront = torch.einsum('ijk,klm->ijlm', modes_directions, self.basis_meta[j][i][0:n_active_height, :, :])
                else:                    
                    modes_directions = torch.einsum('ijk,mj->imk', M[i][j][:, 0:n_active_height, 0:n_active_height], modes[j][:, 0:n_active_height])

                    # Compute wavefronts from estimated modes            
                    wavefront += torch.einsum('ijk,klm->ijlm', modes_directions, self.basis_meta[j][i][0:n_active_height, :, :])
                                    
            # Reuse the same wavefront per object but add the diversity
            # We use any of the basis, which surely contains the defocus mode
            wavef = []            
            for j in range(len(self.init_frame_diversity[i])):                
                div = diversity[i][:, j:j+n_f, None, None] * self.basis_meta[0][i][2, :, :][None, None, :, :]
                wavef.append(wavefront + div)
            
            wavef = torch.cat(wavef, dim=1)
            
            # Compute the complex phase
            phase = self.pupil[i][None, None, :, :] * torch.exp(1j * wavef)

            # Compute FFT of the pupil function and compute autocorrelation
            ft = torch.fft.fft2(phase)
            psf = (torch.conj(ft) * ft).real
            
            # Normalize PSF        
            psf_norm[i] = psf / torch.sum(psf, [-1, -2], keepdim=True)

            # FFT of the PSF
            psf_ft[i] = torch.fft.fft2(psf_norm[i])
        
        return psf_norm, psf_ft
    
        
    
    def define_metapupils(self):
        
        self.n_heights = len(self.heights)
        self.DMetapupil = np.zeros(self.n_heights)

        self.pupil_meta = [None] * self.n_heights
        self.basis_meta = [None] * self.n_heights

        self.projection_matrices = []
        zernike_projection = projection.ZernikeProjectionMatrix()

        for i, h in enumerate(self.heights):
            self.logger.info(f"*** Layer {i} at {h} km")            

            # We need to take piston into account always
            n_modes = self.n_modes_height[i]
            self.define_basis(n_modes)
            self.pupil_meta[i] = self.pupil
            self.basis_meta[i] = self.basis

            self.logger.info(f"Projection matrix")
        
            M, DMetapupil = zernike_projection.zernikeProjectionMatrix(self.n_modes, 
                                                   self.XY.cpu().numpy(), 
                                                   self.config['telescope']['diameter'] / 100, 
                                                   h,
                                                   cm_per_pix=5.0)            
            self.DMetapupil[i] = DMetapupil

            M = torch.tensor(M.astype('float32'), device=self.device)
            self.projection_matrices.append(M)
               
    def plot_metapupils(self):
        zernike_projection = projection.ZernikeProjectionMatrix()
        zernike_projection.plotMetapupils(self.XY.cpu().numpy(),
                                          self.config['telescope']['diameter'] / 100, 
                                          self.heights)                   
            
    def deconvolve(self,                   
                   simultaneous_sequences=1, 
                   infer_object=False, 
                   optimizer='first', 
                   modes_in=None,                    
                   n_iterations=20):
        

        """
        Perform deconvolution on a set of frames using specified parameters.
        Parameters:
        -----------        
        simultaneous_sequences : int, optional
            Number of sequences to be processed simultaneously (default is 1).
        infer_object : bool, optional
            Whether to infer the object during optimization (default is False).
        optimizer : str, optional
            The optimizer to use ('first' for Adam, 'second' for LBFGS) (default is 'first').
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

        self.n_seq, self.n_f, self.n_x, self.n_y = self.frames[0].shape

        self.logger.info(f" ********************************************************")
        self.logger.info(f" *** SPATIALLY VARIANT DECONVOLUTION WITH METAPUPILS  ***")
        self.logger.info(f" ********************************************************")

        # Combine all frames
        self.frames_apodized, self.diversity, self.init_frame_diversity, self.sigma, self.plane = self.combine_frames()
        
        # Define metapupils and associated basis according to the number of modes
        self.define_metapupils()

        
        # Fill the list of frames and apodize the frames if needed
        for i in range(self.n_o):
            self.frames_apodized[i] = self.frames_apodized[i].to(self.device)
            self.diversity[i] = self.diversity[i].to(self.device)
            self.sigma[i] = self.sigma[i].to(self.device)
            
        self.logger.info(f"Frames")        
        for i in range(self.n_o):
            n_s, n_f, n_x, n_y = self.frames_apodized[i].shape
            self.logger.info(f"  * Object {i}")
            self.logger.info(f"     - Number of sequences {n_s}...")
            self.logger.info(f"     - Number of frames {n_f}...")
            self.logger.info(f"     - Number of diversity channels {len(self.init_frame_diversity[i])}...")
            for j, ind in enumerate(self.init_frame_diversity[i]):
                self.logger.info(f"       -> Diversity {j} = {self.diversity[i][0, ind]}...")
            self.logger.info(f"     - Size of frames {n_x} x {n_y}...")
            self.logger.info(f"     - Filter: {self.image_filter[i]} - cutoff: {self.cutoff[i]}...")
                
        self.finite_difference = util.FiniteDifference().to(self.device)
        self.set_regularizations()
                                                    
        # Compute the diffraction masks
        self.compute_diffraction_masks()
        
        # Annealing schedules        
        modes = np.cumsum(np.arange(1, self.noll_max+1))[1:]
        self.anneal = self.compute_annealing(modes, n_iterations)
                
        # If the regularization parameter is a scalar, we assume that it is the same for all objects
        for reg in self.regularization:
            if reg.type == 'iuwt':                
                if not isinstance(reg.lambda_reg, list):
                    reg.lambda_reg = [reg.lambda_reg] * self.n_o

        # Unknown modes
        if modes_in is not None:
            self.logger.info(f"Using provided initial modes...")
            modes = modes_in.clone().detach().to(self.device).requires_grad_(True)
        else:
            if self.config['initialization']['modes_std'] == 0:
                self.logger.info(f"Initializing modes with zeros...")
                modes = []
                for i in range(self.n_heights):

                    # Always add piston mode
                    modes.append(torch.zeros((self.n_f, self.n_modes_height[i]+1), device=self.device, requires_grad=True))                
            else:
                self.logger.info(f"Initializing modes with random values with standard deviation {self.config['initialization']['modes_std']}")

                modes = []
                for i in range(self.n_heights):
                    tmp = self.config['initialization']['modes_std'] * torch.randn((self.n_f, self.n_modes_height[i]+1))
                    modes.append(tmp.clone().detach().to(self.device).requires_grad_(True))

        # first order/Second order optimizer
        parameters = [{'params': modes, 'lr': self.lr_modes}]

        if optimizer == 'lbfgs':
            self.logger.info(f"Using LBFGS optimizer...")
            opt = torch.optim.LBFGS(parameters, lr=0.01)
        if optimizer == 'adam':
            self.logger.info(f"Using Adam optimizer...")
            opt = torch.optim.Adam(parameters)
        if optimizer == 'adamw':
            self.logger.info(f"Using AdamW optimizer...")
            opt = torch.optim.AdamW(parameters)

        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3*n_iterations)

        losses = torch.zeros(n_iterations, device=self.device)

        t = tqdm(range(n_iterations))

        _, self.psf_diffraction_ft = self.compute_psf_diffraction()

        n_active = 2

        active_per_height = [n_active] * self.n_heights
        
        # Split sequences in batches
        ind_reference = np.arange(n_s)

        # Split the sequences in groups of simultaneous sequences to be computed in parallel
        ind = np.array_split(ind_reference, np.ceil(n_s / simultaneous_sequences))
                
        n_sequences = len(ind)
        
        self.modes = [None] * n_sequences        
        self.loss = [None] * n_sequences

        self.psf_seq = [None] * n_sequences        
        self.obj_seq = [None] * n_sequences
        
        #--------------------------------
        # Start optimization
        #--------------------------------
        for loop in t:

            # Split the sequences in groups of simultaneous sequences to be computed in parallel
            # But we need to shuffle the sequences to avoid that the same sequences are always computed together
            # In the last iteration, we do not shuffle the sequences to end up with the original order
            ind = ind_reference.copy()
            if loop < n_iterations-1:            
                np.random.shuffle(ind)
            
            ind = np.array_split(ind, np.ceil(n_s / simultaneous_sequences))
            
            for i_seq, seq in enumerate(ind):
                
                frames_apodized_seq = []
                plane_seq = []
                frames_ft = []
                sigma_seq = []
                diversity_seq = []
                projection_seq = []
                for i in range(self.n_o):
                    frames_apodized_seq.append(self.frames_apodized[i][seq, ...])
                    plane_seq.append(self.plane[i][seq, ...].to(self.device))
                    frames_ft.append(torch.fft.fft2(self.frames_apodized[i][seq, ...]))
                    sigma_seq.append(self.sigma[i][seq, ...])
                    diversity_seq.append(self.diversity[i][seq, ...].to(self.device))
                    
                    projection_h = []
                    for j in range(self.n_heights):
                        projection_h.append(self.projection_matrices[j][seq, :, :])
                    
                    projection_seq.append(projection_h)
                            
                opt.zero_grad(set_to_none=True)
            
                # Compute PSF from current wavefront coefficients and reference 
                # Set piston of the metapupil to zero. We are not sensitive to piston
                # modes_nopiston = modes.clone()
                # modes_nopiston[:, :, 0] = 0.0
                            
                # modes -> (n_seq, n_f, self.n_modes)                                    
                psf, psf_ft = self.compute_psfs(modes, n_active, projection_seq, diversity_seq)
                
                if self.show_object_info or loop == n_iterations - 1:
                    obj_ft, obj_filter_ft, obj_filter = self.compute_object(frames_ft, psf_ft, sigma_seq, plane_seq)                        

                loss_mse = torch.tensor(0.0).to(self.device)

                # Sum over objects, frames and diversity channels
                for i in range(self.n_o):
                    Q = torch.mean(self.sigma[i]) + torch.sum(psf_ft[i] * torch.conj(psf_ft[i]), dim=1)
                    t1 = torch.sum(frames_ft[i] * torch.conj(frames_ft[i]), dim=1)
                    t2 = torch.sum(torch.conj(frames_ft[i]) * psf_ft[i], dim=1)                        
                    loss_mse += torch.mean(t1 - t2 * torch.conj(t2) / Q).real / self.npix**2
                                                                                
                # Object regularization
                loss_obj = torch.tensor(0.0).to(self.device)
                for index in self.index_regularization['object']:
                    loss_obj += self.regularization[index](obj_filter)                        
                
                # Total loss
                loss = loss_mse + loss_obj
                                                    
                # Save some information for the progress bar
                self.loss_local = loss.detach()
                self.obj_filter = [None] * self.n_o                
                self.loss_mse_local = loss_mse.detach()
                self.loss_obj_local = loss_obj.detach()

                loss.backward()
                        
                opt.step()

                # scheduler.step()

                if self.cuda:
                    gpu_usage = f'{self.handle.gpu_utilization()}'            
                    memory_usage = f'{self.handle.memory_used() / 1024**2:4.1f}/{self.handle.memory_total() / 1024**2:4.1f} MB'
                    memory_pct = f'{self.handle.memory_used() / self.handle.memory_total() * 100.0:4.1f}%'
                                   
                tmp = OrderedDict()                
                
                if self.cuda:
                    tmp['gpu'] = f'{gpu_usage} %'
                    tmp['mem'] = f'{memory_usage} ({memory_pct})'

                active_per_height = [int(min(n_active, self.n_modes_height[i])) for i in range(self.n_heights)]

                tmp['active'] = f'{active_per_height}'
                if self.show_object_info:
                    tmp['contrast'] = f'{torch.std(self.obj_filter[0]) / torch.mean(self.obj_filter[0]) * 100.0:7.4f}'
                    tmp['minmax'] = f'{torch.min(self.obj_filter[0]):7.4f}/{torch.max(self.obj_filter[0]):7.4f}'
                tmp['LMSE'] = f'{self.loss_mse_local.item():8.6f}'                
                tmp['LOBJ'] = f'{self.loss_obj_local.item():8.6f}'
                tmp['L'] = f'{self.loss_local.item():7.4f}'
                t.set_postfix(ordered_dict=tmp)

                n_active = self.anneal[loop]
                    
                # Store the results for the current set of sequences
                self.loss[i_seq] = losses.detach()

                if (loop == n_iterations - 1):

                    for i in range(self.n_o):
                        psf[i] = psf[i].detach()                    
                        obj_filter[i] = obj_filter[i].detach()
                        
                    self.psf_seq[i_seq] = psf                
                    self.obj_seq[i_seq] = obj_filter
                        
        # Concatenate the results from all sequences and all objects independently
        self.psf = [None] * self.n_o        
        self.obj = [None] * self.n_o
        self.wavefront_metapupil = [None] * self.n_heights
        
        # self.modes = modes.detach()
        
                
        for i in range(self.n_o):            
            tmp = [self.psf_seq[j][i] for j in range(n_sequences)]
            self.psf[i] = torch.cat(tmp, dim=0)

            tmp = [self.obj_seq[j][i] for j in range(n_sequences)]
            self.obj[i] = torch.cat(tmp, dim=0)

        for i in range(self.n_heights):
            self.wavefront_metapupil[i] = modes[i].detach()
                    
        return 
    
if __name__ == '__main__':
    pass