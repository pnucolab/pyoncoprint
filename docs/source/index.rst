.. PyOncoPrint documentation master file

PyOncoPrint Documentation
=========================

.. image:: https://img.shields.io/pypi/v/pyoncoprint.svg
   :target: https://pypi.org/project/pyoncoprint/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/dm/pyoncoprint.svg
   :target: https://pypi.org/project/pyoncoprint/
   :alt: Downloads

PyOncoPrint is a Python library for creating OncoPrint visualizations, commonly used in cancer genomics to display mutation and copy number alteration data across multiple samples and genes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Features
--------

* Create OncoPrint visualizations from pandas DataFrames or numpy arrays
* Support for multiple mutation types with customizable markers
* Add clinical annotations and heatmaps
* Flexible sorting options for genes and samples
* Export high-quality figures for publication

Quick Example
-------------

.. code-block:: python

   import pyoncoprint
   import pandas as pd

   # Create your mutation matrix
   data = pd.DataFrame({
       'Sample1': ['Missense', '', 'Amplification'],
       'Sample2': ['', 'Truncating', ''],
       'Sample3': ['Missense', 'Missense', 'Deep Deletion']
   }, index=['Gene1', 'Gene2', 'Gene3'])

   # Define mutation markers
   markers = {
       'Missense': {'marker': 'fill', 'color': 'green'},
       'Truncating': {'marker': 'fill', 'color': 'black'},
       'Amplification': {'marker': 'fill', 'color': 'red'},
       'Deep Deletion': {'marker': 'fill', 'color': 'blue'}
   }

   # Create OncoPrint
   op = pyoncoprint.OncoPrint(data)
   fig, axes = op.oncoprint(markers)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

