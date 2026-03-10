.. include:: ../code_name
.. _configuration:


Destretching
============

A burst of images affected by differential seeing in the field-of-view can be prealigned using the ``torchmfbd.destretch`` function. 
The function uses a grid of control points to warp the images to a reference frame. The function returns the warped images and the transformation tensor.
The same destretching is applied to all objects in the burst. The destretching is performed by computing the optical flow for all frames
in the field-of-view to align the frames to a reference frame. To this end, it uses the correlation between the reference frame and the warped frames
as defined in "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization" by Georgios D. Evangelidis and Emmanouil Z. Psarakis.

If ``frames`` is a tensor of shape ``(n_seq, n_objects, n_frames, n_x, n_y)``, the destretching is performed in the ``frames`` axis, using
a frame as reference:

::

    warped, opt_flow = torchmfbd.destretch(frames,
              ngrid=8, 
              lr=0.50,
              reference_frame=0,
              border=6,
              n_iterations=200,
              lambda_tt=0.01,
              mode='bilinear')

The options of the function are:

* ``ngrid``: int, optional
    Number of control points in the x and y directions. Default is 32.
* ``lr``: float, optional
    Learning rate for the optimization. The optimization is done via an Adam optimizer. Default is 0.50.
* ``reference_frame``: int, optional
    Index of the reference frame. Default is 0.
* ``border``: int, optional
    Border of the images to be ignored. This can be used to avoid apodization, if done in advance, or to remove the effect of boundary effects in the camera. Default is 6.
* ``n_iterations``: int, optional
    Number of iterations for the optimization. Default is 200.
* ``lambda_tt``: float, optional
    Regularization parameter for the transformation tensor. The regularization is :math:`\lambda |\nabla_x \mathbf{v} + \nabla_y \mathbf{v}|^2`, the L2 norm of the gradient of the horizontal and vertical optical flow. Default is 0.01.
* ``mode``: str, optional
    Interpolation mode for the warping. Default is 'bilinear' (Options are 'bilinear' or 'nearest').

The previous function returns the warped frames and the optical flow. Once computed, the optical flow
can be applied to an other observed frames. For instance, one can compute the warp using the wideband
images and apply the same warp to the narrowband images. This can be done using the following function:

::

    warped_NB = torchmfbd.apply_destretch(frames_NB, opt_flow, mode='bilinear')


Alignment
==========

A simpler version of the destretching is available using the ``torchmfbd.align`` function. This function infers the affine
transformation between the reference frame and the other frames. The function returns the warped images and the transformation tensor.
The input frames are assumed to be of shape ``(n_frames, n_x, n_y)``.

::

    warped, affine_matrix = torchmfbd.align(frames,
                lr=0.0050,
                border=1,
                n_iterations=100)

Again, the affine transformation can be applied to other frames using the ``torchmfbd.apply_align`` function:

::

    warped2 = torchmfbd.apply_align(frames2, affine_matrix, mode='bilinear')