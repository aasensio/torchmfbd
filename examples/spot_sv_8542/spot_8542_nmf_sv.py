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
    npix = 256
    obs_file = f"../spot_8542/spot_20200727_083509_8542_npix512_original.h5"

    print(f'Reading observations from {obs_file}...')
    f = h5py.File(obs_file, 'r')
    im = f['im'][:]
    im_d = None
    f.close()
        
    frames = im[:, :, :, 0:npix, 0:npix]

    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)
    
    frames = torch.tensor(frames.astype('float32'))
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape
    
    decSI = torchmfbd.DeconvolutionSV('nmf_sv.yaml')

    # Patchify and add the frames
    for i in range(2):                
        decSI.add_frames(frames[:, i, :, :, :], id_object=i, id_diversity=0, diversity=0.0)


    decSI.deconvolve(simultaneous_sequences=1,
                     n_iterations=20,
                     batch_size=12)
        
    obj = [None] * 2
    for i in range(2):
        obj[i] = decSI.obj[i].cpu().numpy()

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(2):
        ax[0, i].imshow(frames[0, i, 0, :, :])
        ax[1, i].imshow(obj[i][0, :, :])
    pl.savefig('nmf.png')    
