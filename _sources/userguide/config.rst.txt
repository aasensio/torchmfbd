.. include:: ../code_name
.. _configuration:


Configuration file
==================

Configuration files are YAML human-readable files. Note that some of them are optional
and can be absent from the configuration file. If this is the case, a default
value will be used.

Spatially invariant
-------------------

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
        cutoff : 0.75

    object2:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : 0.75
        
    optimization:
        gpu : 0
        transform : softplus
        softplus_scale : 1.0        
        lr_obj : 0.02
        lr_modes : 0.08

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

telescope
^^^^^^^^^

* ``diameter``: diameter of the telescope in cm
* ``central_obscuration`` : central obscuration of the telescope in cm

images
^^^^^^

* ``n_pixel`` : size in pixels of the images to be deconvolved. In case the images are mosaicked to take into account the size of the anisoplanatic patch, this keyword gives the size of the patches
* ``pix_size`` : pixel size in arcsec
* ``apodization_border`` : border of the apodization in pixels to avoid edge effects in the Fourier transform used for the convolution

objects
^^^^^^^

Since several objects can be deconvolved simultaneously, one needs to define the parameters for each object. Objects are defined as ``objectx``, where ``x`` is an index
starting from 1.

* ``wavelength`` : wavelength of the object in angstroms
* ``image_filter`` : Fourier filter to be used for the image. Options are ``tophat`` (a simple tophat filter to avoid frequencies above the cutoff) or ``scharmer`` (the filter from Lofdahl & Scharmer 1994).
* ``cutoff`` : cutoff frequency indicated in units of the diffraction limit

optimization
^^^^^^^^^^^^
* ``gpu`` : index of the GPU to be used for the optimization. If you want to use the CPU, just put ``-1``.
* ``transform`` : in case the object is optimized simultaneously with the wavefronts, a transformation can be applied to ensure its positivity. Options are ``softplus`` (which uses the softplus function) or ``none`` (which uses the identity function).
* ``softplus_scale`` : scale of the softplus function
* ``lr_obj`` : learning rate for the object
* ``lr_modes`` : learning rate for the modes

regularization
^^^^^^^^^^^^^^
Different regularizations can be used simultaneously. Current options are ``smooth`` (an L2 regularization applied on the spatial derivative of the images) or ``iuwt`` (an L1 regularization on the isotropic undecimated wavelet transform).

* ``iuwt1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter
    * ``nbands`` : number of bands for the iuwt

* ``smooth1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter    

psf
^^^
* ``model`` : model to be used for the PSF. Options are ``kl`` (which uses Karhunen-Loeve modes) or ``zernike`` (which uses Zernike modes)
* ``nmax_modes`` : number of modes to be used for the PSF

initialization
^^^^^^^^^^^^^^
* ``object`` : initialization of the object. Options are ``contrast`` (which initializes the object with the contrast) or ``random`` (which initializes the object with random values). This has impact only if the ``l2``loss is used.
* ``modes_std`` : standard deviation of the modes at initialization

annealing
^^^^^^^^^

The modes are annealed during the optimization. This means modes are added slowly while the optimization proceeds. The specific annealing :

* ``type`` : type of annealing. Options are ``sigmoid`` (which uses a sigmoid function), ``linear`` (which uses a linear function) or ``none``
* ``start_pct`` : percentage of the total number of iterations at which the annealing starts
* ``end_pct`` : percentage of the total number of iterations at which the annealing ends


Spatially variant
-------------------

::

    telescope:
        diameter: 100.0
        central_obscuration : 0.0

    images:
        n_pixel : 256
        wavelength : 8542.0
        pix_size : 0.059
        apodization_border : 6

    object1:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : 0.75

    object2:
        wavelength : 8542.0
        image_filter: tophat
        cutoff : 0.75
            
    optimization:
        gpu : 0
        transform : softplus
        softplus_scale : 1000.0        
        lr_obj : 0.02
        lr_modes : 0.08
        lr_tt : 0.003

    regularization:
        smooth1:
            variable: tiptilt
            lambda: 0.01
        smooth2:
            variable: modes
            lambda: 0.01
        iuwt1:
            variable : object
            lambda : 0.00
            nbands : 5

    psf:
        model : nmf        
        nmax_modes : 150
        filename: '../basis/nmf_8542_n_150_r0_5_30.npz'
        ngrid_modes: 8

    initialization:
        object : contrast
        modes_std : 0.0

    annealing:
        type: none
        start_pct : 0.0
        end_pct : 0.85

telescope
^^^^^^^^^

* ``diameter``: diameter of the telescope in cm
* ``central_obscuration`` : central obscuration of the telescope in cm

images
^^^^^^

* ``n_pixel`` : size in pixels of the images to be deconvolved. In case the images are mosaicked to take into account the size of the anisoplanatic patch, this keyword gives the size of the patches
* ``pix_size`` : pixel size in arcsec
* ``apodization_border`` : border of the apodization in pixels to avoid edge effects in the Fourier transform used for the convolution

objects
^^^^^^^

Since several objects can be deconvolved simultaneously, one needs to define the parameters for each object. Objects are defined as ``objectx``, where ``x`` is an index
starting from 1.

* ``wavelength`` : wavelength of the object in angstroms
* ``image_filter`` : Fourier filter to be used for the image. Options are ``tophat`` (a simple tophat filter to avoid frequencies above the cutoff) or ``scharmer`` (the filter from Lofdahl & Scharmer 1994).
* ``cutoff`` : cutoff frequency indicated in units of the diffraction limit

optimization
^^^^^^^^^^^^
* ``gpu`` : index of the GPU to be used for the optimization. If you want to use the CPU, just put ``-1``.
* ``transform`` : in case the object is optimized simultaneously with the wavefronts, a transformation can be applied to ensure its positivity. Options are ``softplus`` (which uses the softplus function) or ``none`` (which uses the identity function).
* ``softplus_scale`` : scale of the softplus function
* ``lr_obj`` : learning rate for the object
* ``lr_tt`` : learning rate for the tip-tilt
* ``lr_modes`` : learning rate for the modes

regularization
^^^^^^^^^^^^^^
Different regularizations can be used simultaneously. Current options are ``smooth`` (an L2 regularization applied on the spatial derivative of the images) or ``iuwt`` (an L1 regularization on the isotropic undecimated wavelet transform).

* ``iuwt1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter
    * ``nbands`` : number of bands for the iuwt

* ``smooth1`` : iuwt regularization to be used. The parameters are:
    * ``variable`` : variable to be regularized. Options are ``object`` (which regularizes the object) or ``modes`` (which regularizes the modes)
    * ``lambda`` : regularization parameter    

psf
^^^
* ``model`` : model to be used for the PSF. Options are ``nmf`` (which uses non-negative matrix factorization) or ``pca`` (which uses principal component analysis)
* ``nmax_modes`` : number of modes to be used for the PSF
* ``filename`` : filename of the basis to be used for the PSF
* ``ngrid_modes`` : number of grid points where to infer the PSF modes. They will be interpolated to the full size of the image

initialization
^^^^^^^^^^^^^^
* ``object`` : initialization of the object. Options are ``contrast`` (which initializes the object with the contrast) or ``random`` (which initializes the object with random values). This has impact only if the object is inferred.
* ``modes_std`` : standard deviation of the modes at initialization

annealing
^^^^^^^^^

The modes are annealed during the optimization. This means modes are added slowly while the optimization proceeds. The specific annealing :

* ``type`` : type of annealing. Options are ``sigmoid`` (which uses a sigmoid function), ``linear`` (which uses a linear function) or ``none``
* ``start_pct`` : percentage of the total number of iterations at which the annealing starts
* ``end_pct`` : percentage of the total number of iterations at which the annealing ends