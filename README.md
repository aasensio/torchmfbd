# torchmfbd

## Introduction
``torchmfbd`` is a Python 3 package to carry out multi-object multi-frame blind deconvolution (MOMFBD) of point-like or 
extended objects, specially taylored for solar images. It is built on top of PyTorch and provides a high-level interface for adding observations,
defining phase diversity channels and adding regularization. It can deal with spatially variant PSFs either by mosaicking the images or by
defining a spatially variant PSF.


## Features

- User-friendly API.
- Easy to use configuration file.
- Spatially invariant and variant PSFs.
- Easy-to-use regularization. The current version supports smooth solutions, and solutions based on the $\ell_1$ penalization of the isotropic
undecimated wavelet transform of the object. Regularizations are easily extendable.
- Phase diversity.

## Installation

Install it using ``pip install torchmfbd``.

## Documentation

Visit the [documentation](https://aasensio.github.io/torchmfbd/) for detailed instructions of installation and use.

## Reproducibility
All figures of the accompanying paper can be reproduced using the code in the `reproducibility` directory. The
observations can be download from [here](https://cloud.iac.es/index.php/s/EqMGsqBeyfq6Bnr).