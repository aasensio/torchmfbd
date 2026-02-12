import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
sys.path.append('../')
from readsst import readsst
import torchmfbd
from astropy.io import fits


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

    contrast = np.std(frames, axis=(-1,-2)) / np.mean(frames, axis=(-1,-2))
    ind_best_contrast = np.argmax(contrast[0, 0, :])

    frames = torch.tensor(frames.astype('float32'))

    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = frames.shape

    sizes = [32, 64, 128]

    s = 64
    values = [4, 8, 16, 32, 75, 125, 250, 500, 1000]

    times_all = np.zeros(len(values))
    times_all_convergence = np.zeros(len(values))
    mem_all = np.zeros(len(values))
        
    decSI = torchmfbd.Deconvolution(f'spot_8542_kl_{s}.yaml')

    # Patchify and add the frames
    frames_patches = [None] * 2
    for k in range(2):        
        frames_patches[k] = patchify.patchify(frames[:, k, :, :, :], patch_size=int(s), stride_size=int(s)//2, flatten_sequences=True)
        decSI.add_frames(frames_patches[k], id_object=k, id_diversity=0, diversity=0.0)

    breakpoint()
                            
    for j, v in enumerate(values):
        decSI.deconvolve(infer_object=False, 
                        optimizer='first',                      
                        simultaneous_sequences=v,
                        n_iterations=100)
        times_all[j] = decSI.total_time
        times_all_convergence[j] = decSI.total_time_convergence
        mem_all[j] = decSI.total_mem
    
    del decSI

    np.savez(f'times_{s}.npz', times=times_all, times_convergence=times_all_convergence, mem=mem_all, values=values)
