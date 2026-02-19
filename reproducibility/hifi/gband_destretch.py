import numpy as np
import torch
import matplotlib.pyplot as pl
from tqdm import tqdm
import torchmfbd
from astropy.io import fits

def read_frames(npix, nframes, filename, initial):
    f = fits.open(filename)    

    nx, ny = f[initial].data.shape

    nx = min(nx, npix)
    ny = min(ny, npix)

    frames = np.zeros((1, 100, nx, ny), dtype='float32')

    # Read all frames
    for i in tqdm(range(100)):
        frames[0, i, :, :] = f[initial + 2*i].data[0:npix, 0:npix]
    
    frames /= np.mean(frames, axis=(-1, -2), keepdims=True)
    
    # Sort by descending contrast
    contrast = np.std(frames[0, ...], axis=(-1, -2)) / np.mean(frames[0, ...], axis=(-1, -2))
    ind = np.argsort(contrast)[::-1]

    # Choose the nframes with highest contrast
    frames = frames[:, ind[0:nframes], :, :]

    return frames


if __name__ == '__main__':

    # Size of the image and number of frames to use
    npix = 1024
    nframes = 100
    destrecht = True
    
    # Bluec images start at extension 2    
    initial = 1

    frames = read_frames(npix, nframes, '../obs/gband_bluec/hifiplus1_20230721_085151_sd.fts', initial)

    
    # Convert to tensor
    frames = torch.tensor(frames.astype('float32')).to('cuda')
        
    # Destretch
    warped, tt = torchmfbd.destretch(frames[:, None, :, :, :],
            ngrid=128, 
            lr=0.20,
            reference_frame=0,
            border=6,
            n_iterations=60,
            lambda_tt=0.01)
            
    frames = warped[:, 0, :, :, :].cpu().numpy()

    hdu = fits.PrimaryHDU(data=frames)

    # Save the frames
    hdu.writeto('../obs/gband_bluec/gband_aligned.fits', overwrite=True)