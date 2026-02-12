import numpy as np
import os
import h5py
import torch
import matplotlib.pyplot as pl
import sys
from astropy.io import fits
import torchmfbd


if __name__ == '__main__':
    
    f = fits.open('imax_focus.fits')
    im_focus = f[0].data
    f.close()

    f = fits.open('imax_defocus.fits')
    im_defocus = f[0].data
    f.close()

    # ns, nf, nx, ny
    nx, ny = im_focus.shape
    frames_focus = np.zeros((1, 1, nx, ny))
    frames_defocus = np.zeros((1, 1, nx, ny))

    frames_focus[0, 0, :, :] = im_focus
    frames_defocus[0, 0, :, :] = im_defocus

    mn = np.mean(frames_focus, axis=(-1, -2), keepdims=True)
    frames_focus /= mn
    frames_defocus /= mn

    frames_focus = torch.tensor(frames_focus.astype('float32'))
    frames_defocus = torch.tensor(frames_defocus.astype('float32'))
    
    sigma = torchmfbd.compute_noise(frames_focus)
        
    decSI = torchmfbd.Deconvolution('config_imax.yaml')     


    FD = 45.0
    wavelength = 5250e-8
    Delta = 1.0 * wavelength # 1 lambda
    
    d_cm = 8 * FD**2 * Delta

    div = np.pi * d_cm / (8.0 * np.sqrt(3) * wavelength * FD**2)
   
    decSI.add_frames(frames_focus, id_object=0, id_diversity=0, diversity=0.0)
    decSI.add_frames(frames_defocus, id_object=0, id_diversity=1, diversity=div)

    decSI.deconvolve(infer_object=False, 
                     optimizer='first',                     
                     simultaneous_sequences=1,
                     n_iterations=100)
            
    obj = decSI.obj[0].cpu().numpy()
    
    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(15, 5))

    ax[0, 0].imshow(frames_focus[0, 0, :, :])
    ax[0, 1].imshow(frames_defocus[0, 0, :, :])
    ax[0, 2].imshow(obj[0, :, :])

    ax[1, 0].imshow(frames_focus[0, 0, 200:400, 200:400])
    ax[1, 1].imshow(frames_defocus[0, 0, 200:400, 200:400])
    ax[1, 2].imshow(obj[0, 200:400, 200:400])

    # Save the object as a fits file
    hdu0 = fits.PrimaryHDU(frames_focus[0, 0, :, :])
    hdu1 = fits.ImageHDU(frames_defocus[0, 0, :, :])
    hdu2 = fits.ImageHDU(obj[0, :, :])    
    hdul = fits.HDUList([hdu0, hdu1, hdu2])
    hdul.writeto(f'imax.fits', overwrite=True)