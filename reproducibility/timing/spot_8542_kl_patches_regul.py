import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd

class MyRegularization(torchmfbd.Regularization):
    def __init__(self, lambda_reg, variable, value):
        super(MyRegularization, self).__init__('external', lambda_reg, variable)

        self.variable = variable
        self.lambda_reg = lambda_reg
        self.value = value

        # Deal here with any additional parameter needed by the regularization. They are passed through kwargs

    def __call__(self, x):

        # Add your regularization term here
        n_o = len(x)
        loss = 0.0
        for i in range(n_o):
            loss += self.lambda_reg * torch.sum((x[i] - self.value)**2)

        return loss


if __name__ == '__main__':

    xy0 = [200, 200]
    lam = 7
    npix = 512
    obs_file = f"../obs/spot_20200727_083509_8542_npix512_original.h5"

    if (os.path.exists(obs_file)):
        print(f'Reading observations from {obs_file}...')
        f = h5py.File(obs_file, 'r')
        im = f['im'][:]
        im_d = None
        f.close()
    else:
        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres/spot_20200727_083509_8542'
        label = '20200727_083509_8542_nwav_al'
        print(f'Reading wavelength point {lam}...')
        wb, nb = readsst(root, 
                         label, 
                         cam=0, 
                         lam=lam, 
                         mod=0, 
                         seq=[0, 1], 
                         xrange=[xy0[0], xy0[0]+npix], 
                         yrange=[xy0[1], xy0[1]+npix], 
                         destretch=False,
                         instrument='CRISP')

        ns, nf, nx, ny = wb.shape

        # ns, no, nf, nx, ny
        im = np.concatenate((wb[:, None, ...], nb[:, None, ...]), axis=1)
        im_d = None

        print(f"Saving observations to {obs_file}...")
        f = h5py.File(obs_file, 'w')
        f.create_dataset('im', data=im)
        f.close()
    
    frames = im[:, :, :, 0:npix, 0:npix]

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape
    
    decSI = torchmfbd.Deconvolution('spot_8542_kl_patches.yaml')
    myregularization = MyRegularization(lambda_reg=0.01, variable='object', value=0.0)
    decSI.add_external_regularizations(myregularization)

    # Patchify and add the frames
    for i in range(2):        
        frames_patches = patchify.patchify(frames[:, i, :, :, :], patch_size=64, stride_size=50, flatten_sequences=True)
        decSI.add_frames(frames_patches, id_object=i, id_diversity=0, diversity=0.0)
            
    
    decSI.deconvolve(infer_object=False, 
                     optimizer='first', 
                     simultaneous_sequences=16,
                     n_iterations=100)
        
    # modes = decSI.modes.cpu().numpy()
    # # psf = patchify.unpatchify(decSI.psf).cpu().numpy()
    # wavefront = decSI.wavefront.cpu().numpy()
    # degraded = patchify.unpatchify(decSI.degraded, apodization=6).cpu().numpy()
    obj = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=6).cpu().numpy())        
    # obj_diffraction = patchify.unpatchify(decSI.obj_diffraction, apodization=6).cpu().numpy()    
    # frames = patchify.unpatchify(frames_patches).cpu().numpy()

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(2):
        ax[0, i].imshow(frames[0, i, 0, :, :])
        ax[1, i].imshow(obj[i][0, :, :])