.. include:: ../code_name
.. _configuration:


Destretching
============

A burst of images affected by differential seeing in the field-of-view can be prealigned using the ``torchmfbd.destretch`` function. 
The function uses a grid of control points to warp the images to a reference frame. The function returns the warped images and the transformation tensor.
The same destretching is applied to all objects in the burst. The destretching is performed by computing the optical flow for all frames
in the field-of-view to align the frames to a reference frame. To this end, it uses the correlation between the reference frame and the warped frames
as defined in "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization" by Georgios D. Evangelidis and Emmanouil Z. Psarakis.

If ``frames`` is a tensor of shape ``(n_seq, n_objects, n_frames, n_x, n_y)``, the destretching can be performed as follows:

::

    warped, tt = torchmfbd.destretch(frames,
              ngrid=8, 
              lr=0.50,
              reference_frame=0,
              border=6,
              n_iterations=200,
              lambda_tt=0.01)

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
