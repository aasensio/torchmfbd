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

    # pos = ['-1755', '-845', '-390', '-195', '-130', '-65', '+0', '+65', '+130', '+195', '+390', '+845', '+1755']
    lam = 7  # +65 mA
    xy0 = [200, 200]
    npix = 512
    obs_file = f"../obs/qs_20190801_081547_8542_npix512_original.h5"

    if (os.path.exists(obs_file)):
        print(f'Reading observations from {obs_file}...')
        f = h5py.File(obs_file, 'r')
        im = f['im'][:]
        im_d = None
        f.close()
    else:
        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres/qs_20190801_081547'
        label = '20190801_081547_nwav_al'
        print(f'Reading wavelength point {lam}...')
        wb, nb = readsst(root, 
                         label, 
                         cam=0, 
                         lam=lam, 
                         mod=0, 
                         seq=[10, 11], 
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
    
    decSI = torchmfbd.Deconvolution('qs_8542_kl_patches.yaml')

    # Patchify and add the frames
    frames_patches = [None] * 2
    for i in range(2):        
        frames_patches[i] = patchify.patchify(frames[:, i, :, :, :], patch_size=88, stride_size=40, flatten_sequences=True)
        decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0)
            
    
    decSI.deconvolve(infer_object=False, 
                     optimizer='adam', 
                     simultaneous_sequences=200,
                     n_iterations=250)
            
    obj = []    
    best_frame = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())        
        best_frame.append(patchify.unpatchify(frames_patches[i][:, ind_best_contrast, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())

    npix = obj[0][0, :, :].shape[0]
    
    mfbd = [None] * 2
    mfbd[0] = fits.open('../aux/camXX_2019-08-01T08:15:47_00010_8542_8542_+65_lc0.fits')[0].data[None, :, ::-1]    
    mfbd[1] = fits.open('../aux/camXXV_2019-08-01T08:15:47_00010_8542_8542_+65_lc0.fits')[0].data[None, :, ::-1]
    
    # Save the object as a fits file
    best_frame = np.concatenate([best_frame[0][0:1, ...], best_frame[1][0:1, ...]], axis=0)
    obj = np.concatenate([obj[0][0:1, ...], obj[1][0:1, ...]], axis=0)    
    mfbd = np.concatenate(mfbd, axis=0)
    hdu0 = fits.PrimaryHDU(best_frame)    
    hdu1 = fits.ImageHDU(obj)    
    hdu2 = fits.ImageHDU(mfbd)
    hdu3 = fits.ImageHDU(decSI.obj[0].cpu().numpy())
    hdu4 = fits.ImageHDU(decSI.obj[1].cpu().numpy())
    hdu5 = fits.ImageHDU(frames_patches[0].cpu().numpy())
    hdu6 = fits.ImageHDU(frames_patches[1].cpu().numpy())
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
    hdul.writeto(f'qs_8542.fits', overwrite=True)