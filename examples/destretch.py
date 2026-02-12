import numpy as np
import os
import h5py
import torch
from einops import rearrange
import matplotlib.pyplot as pl
from astropy.io import fits
from readsst import readsst
import torchmfbd


if __name__ == '__main__':

    xy0 = [200, 200]
    lam = 7
    npix = 128
    obs_file = f"spot_8542/spot_20200727_083509_8542_npix512_original.h5"

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

    n_scans, n_obj, n_frames, nx, ny = frames.shape

    warped, tt = torchmfbd.destretch(frames,
              ngrid=8, 
              lr=0.50,
              reference_frame=0,
              border=6,
              n_iterations=200,
              lambda_tt=0.01,
              mode='bilinear')
    
    warped_WB, tt = torchmfbd.destretch(frames[:, 0:1, ...],
              ngrid=8, 
              lr=0.50,
              reference_frame=0,
              border=6,
              n_iterations=200,
              lambda_tt=0.01,
              mode='bilinear')
    
    warped_NB = torchmfbd.apply_destretch(frames[:, 1:2, ...], tt, mode='bilinear')
    
    
    print("Generating movie...")
    # torchmfbd.gen_movie(frames[0, 1, ...], warped[0, 1, ...], fps=8, filename='movie.mp4')
    # torchmfbd.gen_movie(frames[0, 1, ...], warped[0, 1, ...], fps=8, filename='movie2.mp4')
    