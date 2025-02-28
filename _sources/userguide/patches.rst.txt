.. include:: ../code_name
.. _configuration:


Mosaicking
==========

When dealing with spatially variant PSFs, the deconvolution is often done via mosaicking. ``torchmfbd`` provides a simple tool
for simplifying the mosaicking and recombination process. The mosaicking is done by instantiating the ``torchmfbd.Patchify4D`` class, which
has two methods, ``patchify`` and ``unpatchify``. The ``patchify`` method takes a tensor of shape ``(n_seq, n_frames, n_x, n_y)`` and
returns another tensor of shape ``(n_seq, n_patches, n_frames, n_x_patch, n_y_patch)``. The ``unpatchify`` method takes a tensor of shape
``(n_seq, n_patches, n_frames, n_x_patch, n_y_patch)`` previously patchified with ``patchify`` and returns a tensor of shape 
``(n_seq, n_frames, n_x, n_y)``. The patchifying is done by splitting the frames into patches of size ``patch_size`` and 
stride ``stride_size``. By default, the patches are flattened along the patches dimension.

::

    import torchmfbd
    patchify = torchmfbd.Patchify4D()
    frames = torch.zeros((4, 12, 256, 256))
    patches = patchify.patchify(frames, patch_size=64, stride_size=64, flatten_sequences=True)
    
In this example, ``patches`` has shape ``(64, 12, 64, 64)``. If we use a different stride, the number of patches will change accordingly:

::
    
    patches = patchify.patchify(frames, patch_size=64, stride_size=32, flatten_sequences=True)

generates a tensor of shape ``(196, 12, 64, 64)``. The patches can be unpatchified using the ``unpatchify`` method:

::

    frames_back = patchify.unpatchify(patches, apodization=0, weight_type='gaussian', weight_param=1.0)

where an apodization can be applied to each patch, often used to remove border effects during the deconvolution.

The ``patchify`` function has the following options:

* ``frames``: tensor
    Tensor of shape ``(n_seq, n_frames, n_x, n_y)`` with the frames to be patchified.
* ``patch_size``: int
    Size of the patches. Default is 64.
* ``stride_size``: int
    Stride of the patches. Default is 64.
* ``flatten_sequences``: bool
    If True, the patches are flattened along the patches dimension. Default is True.

The ``unpatchify`` function has the following options:

* ``patches``: tensor
    Tensor of shape ``(n_seq, n_patches, n_frames, n_x_patch, n_y_patch)`` with the patches to be unpatchified.
* ``apodization``: int
    Size of the apodization to be applied to each patch. Default is 0.
* ``weight_type``: str
    Type of apodization to be applied. Default is None. Options are (None, 'cosine', 'gaussian').
* ``weight_param``: float
    Parameter of the apodization. Default is 1.0. The parameter is given in normalized units for 'gaussian' and in pixels for 'cosine'.

The weighting window can always be accessed on the property ``weight`` of the ``Patchify4D`` class. This allows the user to correctly define the weight of the patches to reduce artifacts.