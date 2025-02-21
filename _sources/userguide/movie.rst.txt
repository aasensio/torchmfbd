.. include:: ../code_name
.. _configuration:


Movie
=====

You can easily generate a movie from a set of frames by using the ``torchmfbd.gen_movie`` function. The function takes as input the 
frames and, optionally, another set of frames, and generates a movie with the frames and the warped frames side by side. The function uses the ``matplotlib`` library to generate the movie. The function returns the movie as a file,
either in ``mp4`` or ``gif`` format. 

::

    torchmfbd.gen_movie(frames, warped, fps=8, filename='movie.mp4')

The function has the following options:

* ``frames``: tensor
    Tensor of shape ``(n_frames, n_x, n_y)`` with the frames to be shown in the movie.
* ``warped``: tensor, optional
    Tensor of shape ``(n_frames, n_x, n_y)`` with the warped frames to be shown in the movie. Default is None.
* ``fps``: int, optional
    Frames per second of the movie. Default is 1.
* ``filename``: str, optional
    Name of the file to save the movie. Default is 'movie.gif'.
* ``deltat``: int, optional
    Time interval between frames in milliseconds. Default is 300.