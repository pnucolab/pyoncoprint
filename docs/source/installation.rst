Installation
============

Requirements
------------

PyOncoPrint requires Python 3.6+ and the following dependencies:

* numpy
* pandas
* matplotlib

Install from PyPI
-----------------

The easiest way to install PyOncoPrint is using pip:

.. code-block:: bash

   pip install pyoncoprint

Install from Source
-------------------

To install the latest development version from GitHub:

.. code-block:: bash

   git clone https://github.com/pjb7687/pyoncoprint.git
   cd pyoncoprint
   pip install -e .

Or directly via pip:

.. code-block:: bash

   pip install git+https://github.com/pjb7687/pyoncoprint.git

Verify Installation
-------------------

To verify that PyOncoPrint is installed correctly:

.. code-block:: python

   import pyoncoprint
   print(pyoncoprint.__version__)