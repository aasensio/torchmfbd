import torch
from torchmfbd.regularization import Regularization
import torchmfbd.iuwt_torch as iuwt_torch

class RegularizationIUWT(Regularization):

    def __init__(self, lambda_reg=0.0, variable=None, nbands=1, n_pixel=128):
        super().__init__('iuwt', variable, lambda_reg)

        self.n_pixel = n_pixel
        self.nbands = nbands

        # Compute noise characteristics if using IUWT    
        noise = torch.randn(1, 1, self.n_pixel, self.n_pixel)
        out = iuwt_torch.starlet_transform(noise, num_bands = self.nbands, gen2 = False)

        self.std_iuwt = torch.zeros(len(out))
        for i in range(len(out)):
            self.std_iuwt[i] = torch.std(out[i])
        
    def print(self):
        self.logger.info(f"  * Adding {self.type} regularization - Variable: {self.variable} - lambda={self.lambda_reg} - nbands={self.nbands}")
        self.logger.info(f"    - Noise characteristics: {self.std_iuwt}")
        

    def iuwt_loss(self, image, scale):
        """
        Compute the loss using the isotropic undecimated wavelet transform (IUWT).
        This function calculates the loss by performing the IUWT on the input image
        and then summing the L1 norms of all coefficients, scaled by their respective
        standard deviations.
        Args:
            image (torch.Tensor): The input image tensor on which the IUWT is performed.
            scale (list or torch.Tensor): A list or tensor of scaling factors for each wavelet band.
        Returns:
            torch.Tensor: The computed loss value.
        """
        

        # Compute the isotropic undecimated wavelet transform
        coefs = iuwt_torch.starlet_transform(image, num_bands = self.nbands, gen2 = False)
        
        # Now compute the loss by addding L1 norms of all coefficients scaled by their standard deviation
        nlev = len(coefs)
        loss = 0.0
        for i in range(nlev-1):            
            loss += scale[i] * torch.mean(torch.abs(coefs[i]))

        return loss        
    
    def __call__(self, x):

        n_o = len(x)

        loss = 0.0#torch.tensor(0.0).to(x[0].device)

        for i in range(n_o):            
            n_s, _, _ = x[i].shape
            
            for j in range(n_s):
                loss += self.lambda_reg[i] * self.iuwt_loss(x[i][j:j+1, ...], scale=self.std_iuwt)

        return loss