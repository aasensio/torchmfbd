.. include:: ../code_name
.. _configuration:


Spatially invariant deconvolution
=================================

Introduction
------------

Deconvolutions when the PSF is constant across the field of view can be carried out using the ``torchmfbd.Deconvolution`` class. 
The following example shows how to deconvolve two objects with a constant PSF. When the anisoplanatic patch is smaller than
the field of view, the deconvolution can be carried out in parallel for all anisoplanatic patches. To this end, the field of view is
mosaicked using the functions provided by |torchmfbd|. 

Using the following figure from van Noort et al. (2005), ``torchmfbd.Deconvolution`` can be used to deconvolve a burst
of images taken for different objects and with potentially several phase diversity channels. All objects and diversity 
images are assumed to be obtained strictly simultaneously, so that the wavefront is the same for all of them. Diversity
channels differ in a known defocus. 

.. figure:: ../frames.png

Example
-------

The following example shows how to deconvolve a field of view with two objects without a diversity channel. The field of view
is first mosaicked into patches, and the deconvolution is carried out in parallel for each patch. First we define a 
configuration file that controls the general behavior of the deconvolution. This configuration file can also
be defined in the code as a dictionary:

::

    telescope:
        diameter: 100.0
        central_obscuration : 0.0

    images:
        n_pixel : 64
        pix_size : 0.059
        apodization_border : 6

    object1:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : [0.75, 0.80]

    object2:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : [0.75, 0.80]
        
    optimization:
        gpu : 0
        transform : softplus
        softplus_scale : 1.0        
        lr_obj : 0.02
        lr_modes : 0.08
        shod_object_info : False

    regularization:
        iuwt1:
            variable : object
            lambda : 0.0
            nbands : 5

    psf:
        model : kl
        nmax_modes : 44

    initialization:
        object : contrast
        modes_std : 0.0

    annealing:
        type: sigmoid
        start_pct : 0.0
        end_pct : 0.85

The deconvolution can be carried out by the following code:

::

    deconv = torchmfbd.Deconvolution('qs_8542_kl_patches.yaml')

    # Patchify and add the frames
    frames_patches = [None] * 2
    for i in range(2):        
        frames_patches[i] = patchify.patchify(frames[:, i, :, :, :], patch_size=64, stride_size=50, flatten_sequences=True)
        decSI.add_frames(frames_patches[i], id_object=i, id_diversity=0, diversity=0.0)
            
    
    deconv.deconvolve(infer_object=False, 
                     optimizer='first', 
                     simultaneous_sequences=16,
                     n_iterations=20)
            
    obj = []
    frames_back = []
    for i in range(2):
        obj.append(patchify.unpatchify(deconv.obj[i], apodization=6).cpu().numpy())
        frames_back.append(patchify.unpatchify(frames_patches[i], apodization=0).cpu().numpy())

    npix = obj[0][0, :, :].shape[0]
    
    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(2):
        ax[0, i].imshow(frames[0, i, 0, 0:npix, 0:npix])
        ax[1, i].imshow(obj[i][0, :, :])

First, we instantiate the ``torchmfbd.Deconvolution`` class with the configuration file. We then patchify the frames and 
add them to the deconvolution object. The field of view is mosaicked into patches of size 64x64 with a stride of 50 pixels, so 
that they overlap. The frames are added object by object, indicating the index of the object ``id_object`` and the index of the
diversity channel ``id_diversity``. In this case, we have two objects and no diversity channel, so we set ``id_diversity=0`` and
``diversity=0.0``. You can add as many objects and diversity channels per object as you want.

The deconvolution is carried out by calling the ``deconvolve`` method. The method takes several arguments:

* ``infer_object``: If ``False``, the object is inferred using the analytic solution given by the Wiener filter (see, e.g., van Noort et al. (2005)). Otherwise, the object is inferred by the optimizer.
* ``optimizer``: The optimizer to use. The optimizer can be either ``first`` (first order Adam) or ``second`` (second order L-BFGS, that is more memory and time consuming but more efficient in terms of number of iterations).
* ``simultaneous_sequences``: The number of patches to deconvolve simultaneously. If you have plenty of VRAM, you can increase this number to speed up the deconvolution.
* ``n_iterations``: The number of iterations to carry out.

Output
------

::

    2025-02-04 10:17:32,102 - deconvolution  - Using configuration file qs_8542_kl_patches.yaml
    2025-02-04 10:17:32,125 - deconvolution  - Computing in NVIDIA GeForce RTX 4090 (free 22.14 GB) - cuda:0
    2025-02-04 10:17:32,125 - deconvolution  - Using apodization mask with a border of 6 pixels
    2025-02-04 10:17:32,168 - deconvolution  - Telescope
    2025-02-04 10:17:32,168 - deconvolution  -   * D: 100.0 m
    2025-02-04 10:17:32,168 - deconvolution  -   * pix: 0.059 arcsec
    2025-02-04 10:17:32,407 - deconvolution  - Adding frames for object 0 - diversity 0 - defocus 0.0
    2025-02-04 10:17:32,407 - deconvolution  - Estimating noise...
    2025-02-04 10:17:33,002 - deconvolution  - Adding frames for object 1 - diversity 0 - defocus 0.0
    2025-02-04 10:17:33,002 - deconvolution  - Estimating noise...
    2025-02-04 10:17:33,542 - deconvolution  -  *****************************************
    2025-02-04 10:17:33,543 - deconvolution  -  *** SPATIALLY INVARIANT DECONVOLUTION ***
    2025-02-04 10:17:33,543 - deconvolution  -  *****************************************
    2025-02-04 10:17:33,543 - deconvolution  - Setting up frames...
    2025-02-04 10:17:33,590 - deconvolution  - PSF model: wavefront expansion
    2025-02-04 10:17:33,590 - deconvolution  - Found basis file with 44 modes that can be used for 44 modes
    2025-02-04 10:17:33,590 - deconvolution  - Loading precomputed KL basis: basis/kl_100cm_64px_8542A_44.npz
    2025-02-04 10:17:33,598 - deconvolution  - Wavefront
    2025-02-04 10:17:33,598 - deconvolution  -   * Using 44 modes...
    2025-02-04 10:17:33,598 - deconvolution  - Wavelength 0 (8542.0 A)
    2025-02-04 10:17:33,598 - deconvolution  -   * Diffraction: 0.176191563 arcsec
    2025-02-04 10:17:33,598 - deconvolution  -   * Diffraction (x1.22): 0.21495370685999998 arcsec
    2025-02-04 10:17:33,607 - deconvolution  - Frames
    2025-02-04 10:17:33,607 - deconvolution  -   * Object 0
    2025-02-04 10:17:33,607 - deconvolution  -      - Number of sequences 81...
    2025-02-04 10:17:33,607 - deconvolution  -      - Number of frames 12...
    2025-02-04 10:17:33,607 - deconvolution  -      - Number of diversity channels 1...
    2025-02-04 10:17:33,608 - deconvolution  -        -> Diversity 0 = 0.0...
    2025-02-04 10:17:33,608 - deconvolution  -      - Size of frames 64 x 64...
    2025-02-04 10:17:33,608 - deconvolution  -   * Object 1
    2025-02-04 10:17:33,608 - deconvolution  -      - Number of sequences 81...
    2025-02-04 10:17:33,608 - deconvolution  -      - Number of frames 12...
    2025-02-04 10:17:33,608 - deconvolution  -      - Number of diversity channels 1...
    2025-02-04 10:17:33,608 - deconvolution  -        -> Diversity 0 = 0.0...
    2025-02-04 10:17:33,608 - deconvolution  -      - Size of frames 64 x 64...
    2025-02-04 10:17:33,608 - deconvolution  - Regularization
    2025-02-04 10:17:33,608 - deconvolution  - Adding modes using sigmoid schedule
    2025-02-04 10:17:33,608 - deconvolution  - Processing sequences [1,14]/81
    2025-02-04 10:17:33,629 - deconvolution  - Initializing modes with zeros...
    2025-02-04 10:17:33,630 - deconvolution  - Optimizing modes only...
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 125.04it/s, gpu=16 %, mem=2192.8/24564.0 MB ( 8.9%), active=44, contrast=6.8165, minmax=0.7718/ 1.6871, LMSE=0.040005, LOBJ=0.000000, L=0.0400]
    2025-02-04 10:17:33,792 - deconvolution  - Processing sequences [15,28]/81
    2025-02-04 10:17:33,792 - deconvolution  - Initializing modes with zeros...
    2025-02-04 10:17:33,792 - deconvolution  - Optimizing modes only...
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 304.56it/s, gpu=16 %, mem=2236.8/24564.0 MB ( 9.1%), active=44, contrast=7.7578, minmax=0.7042/ 1.7335, LMSE=0.045078, LOBJ=0.000000, L=0.0451]
    2025-02-04 10:17:33,858 - deconvolution  - Processing sequences [29,42]/81
    2025-02-04 10:17:33,858 - deconvolution  - Initializing modes with zeros...
    2025-02-04 10:17:33,858 - deconvolution  - Optimizing modes only...
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 306.41it/s, gpu=16 %, mem=2258.8/24564.0 MB ( 9.2%), active=44, contrast=6.5759, minmax=0.7459/ 1.6867, LMSE=0.042037, LOBJ=0.000000, L=0.0420]
    2025-02-04 10:17:33,924 - deconvolution  - Processing sequences [43,55]/81
    2025-02-04 10:17:33,936 - deconvolution  - Initializing modes with zeros...
    2025-02-04 10:17:33,936 - deconvolution  - Optimizing modes only...
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 243.11it/s, gpu=35 %, mem=2260.8/24564.0 MB ( 9.2%), active=44, contrast=5.5843, minmax=0.7154/ 1.4883, LMSE=0.038261, LOBJ=0.000000, L=0.0383]
    2025-02-04 10:17:34,019 - deconvolution  - Processing sequences [56,68]/81
    2025-02-04 10:17:34,019 - deconvolution  - Initializing modes with zeros...
    2025-02-04 10:17:34,019 - deconvolution  - Optimizing modes only...
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 319.36it/s, gpu=35 %, mem=2262.8/24564.0 MB ( 9.2%), active=44, contrast=5.4903, minmax=0.7996/ 1.3550, LMSE=0.036806, LOBJ=0.000000, L=0.0368]
    2025-02-04 10:17:34,083 - deconvolution  - Processing sequences [69,81]/81
    2025-02-04 10:17:34,083 - deconvolution  - Initializing modes with zeros...
    2025-02-04 10:17:34,083 - deconvolution  - Optimizing modes only...
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 315.16it/s, gpu=35 %, mem=2284.6/24564.0 MB ( 9.3%), active=44, contrast=5.4909, minmax=0.7334/ 1.3897, LMSE=0.036584, LOBJ=0.000000, L=0.0366]

Once the deconvolution is finished, several attributes are available in the deconvolution object:

* ``obj``: The deconvolved objects.
* ``obj_diffraction``: The deconvolved objects convolved with the diffraction-limited PSF.
* ``psf``: The inferred PSFs
* ``degraded``: The object convolved with the inferred PSFs. They can be used to check the quality of the deconvolution because they should be similar to the input frames.

The mosaicking is undone by calling the ``unpatchify`` function.

