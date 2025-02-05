.. include:: ../code_name

Installation
============

Since |torchmfbd| is a pure Python module, it should be pretty easy to install.

Package managers
----------------

The simplest and recommended way to install |torchmfbd| is by using ``pip``. We recommend to install
it into a virtual environment, using any of the available options in Python, like 
`pip <http://www.virtualenv.org>`_ or `conda <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.
This makes everything much safer, plus making sure that all packages are installed for the code.
For example, once you have installed `Miniconda <https://conda.io/miniconda.html>`_, you can generate
a new environment and install the dependencies (you can install whatever version of Python 3 you desire).

For the installation, just type:

::

    conda create -n torchmfbd
    conda activate torchmfbd
    pip install torchmfbd

The code relies on the ``pytorch`` package. Take a look at the documentation for `PyTorch <https://pytorch.org/>`_ for more information.

From source
-----------

|torchmfbd| is developed on `GitHub <https://github.com/aasensio/torchmfbd>`_. If you want to install the latest version
from the repository, you can clone the repository and install it manually once you have installed
the requirements. For example, you can do:

::

    git clone https://github.com/aasensio/torchmfbd.git
    cd torchmfbd
    python setup.py install

Requirements
------------
|torchmfbd| depends on the following external packages, that should be
pretty straightforward to install:

* ``numpy``
* ``torch``
* ``scipy``
* ``matplotlib``
* ``tqdm``
* ``skimage``
* ``sklearn``
* ``nvitop``
* ``yaml``
* ``einops``
* ``pkgutil``

