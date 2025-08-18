.. include:: code_name

|torchmfbd|
=================================

Introduction
------------
|torchmfbd| is a Python 3 package to carry out multi-object multi-frame blind deconvolution (MOMFBD) of point-like or 
extended objects, specially taylored for solar images. It is built on top of PyTorch and provides a high-level interface for adding observations,
defining phase diversity channels and adding regularization. It can deal with spatially variant PSFs either by mosaicking the images or by
defining a spatially variant PSF.


Features
--------

- |torchmfbd| 
- User-friendly API.
- Easy to use configuration file.
- Spatially invariant and variant PSFs.
- Easy-to-use regularization. The current version supports smooth solutions, and solutions based on the :math:`\ell_1` penalization of the isotropic undecimated wavelet transform of the object. Regularizations are easily extendable.
- Phase diversity.


.. toctree::
   :numbered:   
   :maxdepth: 2
   :hidden:
      
   userguide   
   api
   disclaimer

