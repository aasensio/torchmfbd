.. include:: ../code_name
.. _configuration:


Phase diversity
===============

Introduction
------------

If only one object is used but observed with two cameras, one of them with a defocus, the phase diversity technique can be 
used to recover the phase of the object. 

Example
-------

This example shows the deconvolution of IMaX data. Since the telescope is barely affected by the Earth atmosphere turbulence,
we deconvolve the whole field of view with a single PSF. Mosaicking can be used to allow for changes in the PSF across the 
field of view. First we define a 
configuration file that controls the general behavior of the deconvolution. This configuration file can also
be defined in the code as a dictionary:

::

    telescope:
        diameter: 100.0
        central_obscuration : 0.0

    images:
        n_pixel : 952    
        pix_size : 0.055
        apodization_border : 0
        
    object1:
        wavelength: 5250.0
        image_filter: tophat
        cutoff: 0.75

    optimization:
        gpu : 0
        transform : softplus
        softplus_scale : 1.0    
        lr_obj : 0.02
        lr_modes : 0.08

    regularization:
        iuwt1:
            variable : object
            lambda : 0.00
            nbands : 5

    psf:
        model : zernike
        nmax_modes : 44

    initialization:
        object : contrast
        modes_std : 0.0

    annealing:
        type: sigmoid
        start_pct : 0.0
        end_pct : 0.85

The deconvolution can be carried out by the following code, where we calculate the diversity as 

::

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
    sigma = torch.tensor(sigma.astype('float32'))
        
    decSI = torchmfbd.Deconvolution('imax.yaml')     

    div = torchmfbd.get_diversity(defocus=1.0, FD=45.0, wavelength=5250.0)
       
    decSI.add_frames(frames_focus, sigma, id_object=0, id_diversity=0, diversity=0.0)
    decSI.add_frames(frames_defocus, sigma, id_object=0, id_diversity=1, diversity=div)

    decSI.deconvolve(infer_object=False, 
                     optimizer='first', 
                     simultaneous_sequences=1,
                     n_iterations=100)

We have used the ``torchmfbd.get_diversity`` function to return the diversity coefficient