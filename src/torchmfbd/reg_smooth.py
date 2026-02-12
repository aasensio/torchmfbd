import torch
from torchmfbd.regularization import Regularization
import torchmfbd.util as util
from einops import rearrange

class RegularizationSmooth(Regularization):

    def __init__(self, lambda_reg=0.0, variable=None):
        super().__init__('smooth', variable, lambda_reg)
        
        self.finite_difference = util.FiniteDifference()

    def __call__(self, x):
                   
        if self.active:

            # Tip-tilts and modes have the additional dimension nf
            if self.variable == 'tiptilt' or self.variable == 'modes':
                tmp = rearrange(x, 'b nf nm nx ny -> (b nf) nm nx ny')                
                loss = self.lambda_reg * torch.mean(self.finite_difference(tmp)**2)
            else:
                # Objects are passed as a list
                n_o = len(x)
                loss = 0.0
                for i in range(n_o):
                    loss += self.lambda_reg * torch.mean(self.finite_difference(x[i])**2)

        return loss