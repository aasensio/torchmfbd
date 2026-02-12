import numpy as np
import torch
import torch.nn.functional as F


def bspline_star(x, step):
    """
    This implements the starlet kernel. Application to different scales is
    accomplished via the step parameter.
    """    
    C1 = 1./16.
    C2 = 4./16.
    C3 = 6./16.
    KSize = 4*step+1
    KS2 = int(KSize/2)    
    kernel = torch.zeros((KSize))
    if KSize == 1:
        kernel[0] = 1.0
    else:
        kernel[0] = C1
        kernel[KSize-1] = C1
        kernel[KS2+step] = C2
        kernel[KS2-step] = C2
        kernel[KS2] = C3
    
    kernelx = kernel[None, None, :, None].to(x.device)
    kernely = kernel[None, None, None, :].to(x.device)
    x = F.conv2d(x, kernelx, padding='same')
    x = F.conv2d(x, kernely, padding='same')
    
    return x


def starlet_transform(input_image, num_bands = None, gen2 = True):
    '''
    Computes the starlet transform of an image (i.e. undecimated isotropic
    wavelet transform).

    The output is a python list containing the sub-bands. If the keyword Gen2 is set,
    then it is the 2nd generation starlet transform which is computed: i.e. g = Id - h*h
    instead of g = Id - h.

    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
    Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the STAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html

    '''

    if num_bands == None:
        num_bands = int(np.ceil(np.log2(np.min(input_image.shape))) - 3)
        assert num_bands > 0


    im_in = input_image
    step_trou = 1
    im_out = None
    WT = []

    for band in range(num_bands):
        im_out = bspline_star(im_in, step_trou)
        if gen2:  # Gen2 starlet applies smoothing twice
            WT.append(im_in - bspline_star(im_out, step_trou))
        else:            
            WT.append(im_in - im_out)
        im_in = im_out
        step_trou *= 2

    WT.append(im_out)
    return WT

def inverse_starlet_transform(coefs, gen2 = True):
    '''
    Computes the inverse starlet transform of an image (i.e. undecimated
    isotropic wavelet transform).

    The input is a python list containing the sub-bands. If the keyword Gen2 is
    set, then it is the 2nd generation starlet transform which is computed: i.e.
    g = Id - h*h instead of g = Id - h.

    REFERENCES:
    [1] J.L. Starck and F. Murtagh, "Image Restoration with Noise Suppression Using the Wavelet Transform",
        Astronomy and Astrophysics, 288, pp-343-348, 1994.

    For the modified STARLET transform:
    [2] J.-L. Starck, J. Fadili and F. Murtagh, "The Undecimated Wavelet Decomposition
        and its Reconstruction", IEEE Transaction on Image Processing,  16,  2, pp 297--309, 2007.

    This code is based on the ISTAR2D IDL function written by J.L. Starck.
            http://www.multiresolutions.com/sparsesignalrecipes/software.html
    '''

    # Gen1 starlet can be reconstructed simply by summing the coefficients at each scale.
    if not gen2:
        recon_img = torch.zeros_like(coefs[0])
        for i in range(len(coefs)):
            recon_img += coefs[i]

    # Gen2 starlet requires more careful reconstruction.
    else:
        num_bands = len(coefs)-1
        recon_img = coefs[-1]
        step_trou = np.power(2, num_bands - 1)

        for i in reversed(range(num_bands)):            
            im_temp = bspline_star(recon_img, int(step_trou))
            recon_img = im_temp + coefs[i]
            step_trou /= 2

    return recon_img


    
if __name__ == '__main__':
    from astropy.io import fits
    import matplotlib.pyplot as pl
    from astropy.visualization import (PercentileInterval, LogStretch, ImageNormalize)
    
    f1 = fits.open('NGC0521_g_raw.fits')
    image = f1[1].data[None, None, :, :].astype(np.float32)
    image[image < -0.1] = 0.0
    
    device = torch.device("cuda:0") 

    image_pt = torch.tensor(image).to(device)

    out = starlet_transform(image_pt, num_bands = 5, gen2 = False)
    recons = inverse_starlet_transform(out, gen2 = False)

    fig, ax = pl.subplots(nrows=1, ncols=6, figsize=(20, 5))
    for i in range(6):
        ax[i].imshow(out[i][0, 0, :, :].cpu().numpy(), norm=ImageNormalize(out[i][0, 0, :, :].cpu().numpy(), interval=PercentileInterval(99.5), stretch=LogStretch()))


    noise = torch.randn_like(image_pt) * 1.0
    out = starlet_transform(noise, num_bands = 5, gen2 = False)

    std = torch.zeros(len(out))
    for i in range(len(out)):
        std[i] = torch.std(out[i])