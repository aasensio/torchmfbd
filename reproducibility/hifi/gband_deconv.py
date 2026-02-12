import numpy as np
import torch
import matplotlib.pyplot as pl
from tqdm import tqdm
import torchmfbd
from astropy.io import fits


if __name__ == '__main__':

    n_frames = 80

    f = fits.open('gband_bluec/gband_aligned.fits')

    # Convert to tensor
    frames = torch.tensor(f[0].data.astype('float32'))

    # Create the deconvolution object                
    decSI = torchmfbd.Deconvolution('gband_kl.yaml')

    # Patchify and add the frames
    patchify = torchmfbd.Patchify4D()
    frames_patches = patchify.patchify(frames, patch_size=32, stride_size=10, flatten_sequences=True)
    noise = torchmfbd.compute_noise(frames_patches[0:1, 0:1, ...])
    decSI.add_frames(frames_patches, id_object=0, id_diversity=0, diversity=0.0, sigma=noise)

    # Deconvolve
    decSI.deconvolve(infer_object=False, 
                     optimizer='adam', 
                     simultaneous_sequences=1000,
                     n_iterations=250)
            
    # Unpatchify
    obj = patchify.unpatchify(decSI.obj[0], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy()

    f = fits.open('gband_bluec/hifiplus1_20230721_085151_sd_speckle.fts')
    speckle = f[0].data

    npix = obj.shape[-1]
    
    # Do plots
    fig, ax = pl.subplots(nrows=2, ncols=3, figsize=(25, 15))
    for i in range(2):
        ax[0, 0].imshow(frames[0, 0, 0:npix, 0:npix])
        ax[0, 1].imshow(obj[0, :, :])
        ax[0, 2].imshow(speckle[0:npix, 0:npix])

        ax[1, 0].imshow(frames[0, 0, 400:600, 300:500])
        ax[1, 1].imshow(obj[0, 400:600, 300:500])
        ax[1, 2].imshow(speckle[0:npix, 0:npix][400:600, 300:500])

    
    # Save the object as a fits file
    hdu0 = fits.PrimaryHDU(frames[0, 0, :, :])    
    hdu1 = fits.ImageHDU(obj[0, :, :])    
    hdul = fits.HDUList([hdu0, hdu1])
    hdul.writeto(f'gband.fits', overwrite=True)
