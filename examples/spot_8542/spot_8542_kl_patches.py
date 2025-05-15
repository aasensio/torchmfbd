import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd


if __name__ == '__main__':

    xy0 = [200, 200]
    lam = 7
    npixx = 256
    npixy = 512
    obs_file = f"spot_20200727_083509_8542_npix512_original.h5"

    print(f'Reading observations from {obs_file}...')
    f = h5py.File(obs_file, 'r')
    im = f['im'][:]    
    f.close()
    
    frames = im[:, :, :, 0:npixx, 0:npixy]

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape
    
    decSI = torchmfbd.Deconvolution('kl.yaml')

    # Patchify and add the frames
    for i in range(2):        
        frames_patches = patchify.patchify(frames[:, i, :, :, :], patch_size=64, stride_size=50, flatten_sequences=True)
        decSI.add_frames(frames_patches, id_object=i, id_diversity=0, diversity=0.0)
                
    decSI.deconvolve(infer_object=False, 
                     optimizer='adamw', 
                     simultaneous_sequences=90,
                     n_iterations=150)
            
    obj = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())        
    
    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for i in range(2):
        ax[i, 0].imshow(frames[0, i, 0, :, :], cmap='gray')
        ax[i, 1].imshow(obj[i][0, :, :], cmap='gray')


    decSI.update_object(cutoffs=[[0.3, 0.35], [0.3, 0.35]])

    # Unpatchify
    obj = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())        
        
    for i in range(2):
        ax[i, 2].imshow(obj[i][0, :, :], cmap='gray')

    ax[0, 1].set_title('Reconstructed object')
    ax[0, 2].set_title('Reconstructed object (updated cutoffs)')

    decSI.write('test.fits')