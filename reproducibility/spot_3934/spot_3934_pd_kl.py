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
from einops import rearrange

if __name__ == '__main__':

    lam = 14
    xy0 = [100, 550]
    npix = 1024
    obs_file = f"../obs/spot_20200727_083509_3934_npix1024_original.h5"
    obs_file_cont = f"../obs/spot_20200727_083509_3934_npix1024_cont.h5"

    if (os.path.exists(obs_file)):
        print(f'Reading observations from {obs_file}...')
        f = h5py.File(obs_file, 'r')
        im = f['im'][:]
        im_d = f['im_d'][:]
        f.close()

        print(f'Reading continuum observations from {obs_file_cont}...')
        f = h5py.File(obs_file_cont, 'r')
        im_cont = f['im'][:]        
        f.close()
    else:
        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres/spot_20200727_083509_3934'
        label = '20200727_083509_3934_nwav_al'
        print(f'Reading wavelength point {lam}...')
        wb, nb, db = readsst(root, 
                         label, 
                         cam=0, 
                         lam=lam,
                         mod=0, 
                         seq=[0, 1], 
                         xrange=[xy0[0], xy0[0]+npix], 
                         yrange=[xy0[1], xy0[1]+npix], 
                         destretch=False,
                         instrument='CHROMIS')

        ns, nf, nx, ny = wb.shape

        # ns, no, nf, nx, ny
        im = np.concatenate((wb[:, None, ...], nb[:, None, ...]), axis=1)
        im_d = db[:, None, ...]

        print(f"Saving observations to {obs_file}...")
        f = h5py.File(obs_file, 'w')
        f.create_dataset('im', data=im)
        if (im_d is not None):
            f.create_dataset('im_d', data=im_d)
        f.close()

        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres/spot_20200727_083509_3934'
        label = '20200727_083509_3934_nwav_al'
        print(f'Reading wavelength point {lam}...')
        wb, nb, db = readsst(root, 
                         label, 
                         cam=0, 
                         lam=0,
                         mod=0, 
                         seq=[0, 1], 
                         xrange=[xy0[0], xy0[0]+npix], 
                         yrange=[xy0[1], xy0[1]+npix], 
                         destretch=False,
                         instrument='CHROMIS')

        ns, nf, nx, ny = wb.shape

        # ns, no, nf, nx, ny
        im = np.concatenate((wb[:, None, ...], nb[:, None, ...]), axis=1)
        im_d = db[:, None, ...]
        
        print(f"Saving observations to {obs_file_cont}...")
        f = h5py.File(obs_file, 'w')
        f.create_dataset('im', data=im)
        if (im_d is not None):
            f.create_dataset('im_d', data=im_d)
        f.close()

    # Align pinholes, using the pinhole for wavelentgh 14    
    NB = fits.open('../obs/camXXX_3934_wheel00002_hrz33671.pinh.fits')[0].data[::-1, ::-1]
    # NB = NB[xy0[0]:xy0[0]+npix, xy0[1]:xy0[1]+npix]

    WB = fits.open('../obs/camXXVIII_3950_wheel00002_hrz33671.pinh.fits')[0].data[:, :]
    # WB = WB[xy0[0]:xy0[0]+npix, xy0[1]:xy0[1]+npix]

    PD = fits.open('../obs/camXXVII_3950_wheel00002_hrz33671.pinh.fits')[0].data[:, ::-1]
    # PD = PD[xy0[0]:xy0[0]+npix, xy0[1]:xy0[1]+npix]
    
    # WB and NB
    frames = np.concatenate([WB[None, :, :], NB[None, :, :]], axis=0)
    print("Estimating alignment between WB and NB...")
    # frames /= np.mean(frames, axis=(-1, -2), keepdims=True)
    frames = torch.tensor(frames.astype('float32'))
    warped_NB, affine_WB_NB = torchmfbd.align(frames,
                lr=0.0050,
                border=1,
                n_iterations=100)
    print("Affine matrix NB to WB:")
    print(affine_WB_NB)
    
    # WB and PD
    frames = np.concatenate([WB[None, :, :], PD[None, :, :]], axis=0)
    print("Estimating alignment between WB and PD...")
    # frames /= np.mean(frames, axis=(-1, -2), keepdims=True)
    frames = torch.tensor(frames.astype('float32'))
    warped_PD, affine_WB_PD = torchmfbd.align(frames,
                lr=0.0050,
                border=1,
                n_iterations=100)
    print("Affine matrix PD to WB:")
    print(affine_WB_PD)
       
    # Align all frames
    frames_WB_NB = im[:, :, :, 0:npix, 0:npix]
    frames_PD = im_d[:, :, :, 0:npix, 0:npix]
    frames = np.concatenate([frames_WB_NB, frames_PD], axis=1)
    frames /= np.mean(frames, axis=(-1, -2, -3), keepdims=True)   

    contrast = np.std(frames, axis=(-1,-2)) / np.mean(frames, axis=(-1,-2))
    ind_best_contrast = np.argmax(contrast[0, 0, :])

    frames_aligned = torch.tensor(frames.astype('float32'))
        
    warpedNB = torchmfbd.apply_align(frames_aligned[:, 1, :, :, :], affine_WB_NB)
    warpedPD = torchmfbd.apply_align(frames_aligned[:, 2, :, :, :], affine_WB_PD)
    frames_aligned[:, 1, :, :, :] = warpedNB
    frames_aligned[:, 2, :, :, :] = warpedPD

    # Align PD and WB channels
    print("Destretching PD channels...")
    WB_PD = frames_aligned[:, [0, 2], :, :, :]
    WB_PD = rearrange(WB_PD, 'nb no nf nx ny -> nb nf no nx ny')
    warped_PD2, tt = torchmfbd.destretch(WB_PD[0, 0, :, :, :],
            ngrid=64, 
            lr=0.50,
            reference_frame=0,
            border=6,
            n_iterations=50,
            lambda_tt=0.01)
    
    # Align all PD frames
    out = torchmfbd.apply_destretch(WB_PD, tt)
    frames_aligned[:, 2, :, :, :] = out[0, :, 1, :, :]
    
    # Destretch the frames
    print("Destretching...")
    warped, tt = torchmfbd.destretch(frames_aligned,
            ngrid=64, 
            lr=0.50,
            reference_frame=ind_best_contrast,
            border=6,
            n_iterations=20,
            lambda_tt=0.01)
                    
    patchify = torchmfbd.Patchify4D()    
            
    n_scans, n_obj, n_frames, nx, ny = warped.shape
    
    decSI = torchmfbd.Deconvolution('spot_3934_pd_kl_patches.yaml')

    # Patchify and add the frames
    frames_patches = [None] * 2    
    for i in range(2):        
        frames_patches[i] = patchify.patchify(warped[:, i, :, 0:512, 0:512], patch_size=128, stride_size=40, flatten_sequences=True)
        noise = torchmfbd.compute_noise(frames_patches[i][0:1, 0:1, ...])
        decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0, sigma=noise)

    # Add diversity channel
    FD = 32.8
    wavelength = 3934e-8
    Delta = 1.0 * wavelength # 1 lambda    
    d_cm = 8 * FD**2 * Delta
    div = np.pi * d_cm / (8.0 * np.sqrt(3) * wavelength * FD**2)
    frames_PD = patchify.patchify(warped[:, 2, :, 0:512, 0:512], patch_size=128, stride_size=40, flatten_sequences=True)
    noise = torchmfbd.compute_noise(frames_PD[0:1, 0:1, ...])
    decSI.add_frames(frames_PD, id_object=0, id_diversity=1, diversity=div, sigma=noise)
                
    decSI.deconvolve(infer_object=False, 
                     optimizer='adam',                      
                     simultaneous_sequences=100,
                     n_iterations=450)
            
    best_frame = []
    obj = []
    for i in range(2):
        obj.append(patchify.unpatchify(decSI.obj[i], apodization=18, weight_type='cosine', weight_params=30).cpu().numpy())
        best_frame.append(patchify.unpatchify(frames_patches[i][:, ind_best_contrast, :, :], apodization=6, weight_type='cosine', weight_params=30).cpu().numpy())
    

    mfbd = [None] * 2
    mfbd[0] = fits.open('../aux/camXXVIII_2020-07-27T08:35:09_00000_12.00ms_G10.00_3934_3934_+65.fits')[0].data[None, :, :]
    mfbd[1] = fits.open('../aux/camXXX_2020-07-27T08:35:09_00000_12.00ms_G10.00_3934_3934_+65.fits')[0].data[None, :, :]

    # Save the object as a fits file
    best_frame = np.concatenate([best_frame[0][0:1, ...], best_frame[1][0:1, ...]], axis=0)
    obj = np.concatenate([obj[0][0:1, ...], obj[1][0:1, ...]], axis=0)    
    mfbd = np.concatenate(mfbd, axis=0)
    
    hdu0 = fits.PrimaryHDU(best_frame)
    hdu1 = fits.ImageHDU(obj)
    hdu2 = fits.ImageHDU(mfbd)
    hdu3 = fits.ImageHDU(decSI.obj[0].cpu().numpy())
    hdu4 = fits.ImageHDU(decSI.obj[1].cpu().numpy())
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])
    hdul.writeto(f'spot_3934_pd.fits', overwrite=True)