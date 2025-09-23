Quick Start Guide
=================

This guide will walk you through creating your first OncoPrint visualization.

Basic Usage
-----------

1. Import the library and prepare your data:

.. code-block:: python

   import pyoncoprint
   import pandas as pd
   import matplotlib.pyplot as plt

   # Create a mutation matrix (genes x samples)
   data = pd.DataFrame({
       'Sample1': ['Missense', '', 'Amplification', ''],
       'Sample2': ['', 'Truncating', '', 'Missense'],
       'Sample3': ['Missense', 'Missense', 'Deep Deletion', ''],
       'Sample4': ['Amplification', '', '', 'Truncating']
   }, index=['TP53', 'KRAS', 'EGFR', 'BRAF'])

2. Define mutation markers:

.. code-block:: python

   markers = {
       'Missense': {
           'marker': 'fill',
           'color': 'green',
           'height': 0.5
       },
       'Truncating': {
           'marker': 'fill',
           'color': 'black',
           'height': 0.5
       },
       'Amplification': {
           'marker': 'fill',
           'color': 'red'
       },
       'Deep Deletion': {
           'marker': 'fill',
           'color': 'blue'
       }
   }

3. Create the OncoPrint:

.. code-block:: python

   # Initialize OncoPrint object
   op = pyoncoprint.OncoPrint(data)

   # Generate the plot
   fig, axes = op.oncoprint(
       markers,
       figsize=[20, 10],
       title="My First OncoPrint"
   )

   plt.show()

Adding Clinical Annotations
----------------------------

You can add clinical data as annotation tracks:

.. code-block:: python

   # Create annotation data
   import numpy as np

   # Categorical annotation
   cancer_type = pd.DataFrame(
       ['Lung', 'Lung', 'Breast', 'Colon'],
       columns=['Cancer Type'],
       index=['Sample1', 'Sample2', 'Sample3', 'Sample4']
   ).T

   # Numerical annotation
   age = pd.DataFrame(
       [65, 72, 58, 61],
       columns=['Sample1', 'Sample2', 'Sample3', 'Sample4'],
       index=['Age']
   )

   # Define annotation styles
   annotations = {
       'Cancer Type': {
           'annotations': cancer_type,
           'colors': {
               'Lung': 'lightblue',
               'Breast': 'pink',
               'Colon': 'lightgreen'
           },
           'order': 0
       },
       'Age': {
           'annotations': age,
           'color': 'orange',
           'order': 1
       }
   }

   # Create OncoPrint with annotations
   fig, axes = op.oncoprint(
       markers,
       annotations=annotations,
       figsize=[20, 12]
   )

Adding Heatmaps
---------------

You can include expression or other continuous data as heatmaps:

.. code-block:: python

   # Create expression data
   expression_data = pd.DataFrame(
       np.random.randn(4, 4),
       index=['TP53', 'KRAS', 'EGFR', 'BRAF'],
       columns=['Sample1', 'Sample2', 'Sample3', 'Sample4']
   )

   # Define heatmap configuration
   heatmaps = {
       'Gene Expression': {
           'heatmap': expression_data,
           'cmap': 'RdBu_r',  # Red-Blue colormap
           'vmin': -2,
           'vmax': 2
       }
   }

   # Create OncoPrint with heatmap
   fig, axes = op.oncoprint(
       markers,
       heatmaps=heatmaps,
       figsize=[20, 15]
   )

Customization Options
---------------------

The `oncoprint()` method supports many customization options:

.. code-block:: python

   fig, axes = op.oncoprint(
       markers,
       annotations=annotations,
       heatmaps=heatmaps,
       title="Customized OncoPrint",
       figsize=[30, 20],
       gene_sort_method='default',      # Sort genes by frequency
       sample_sort_method='default',     # Sort samples by mutations
       topplot=True,                     # Show mutation frequency plot on top
       rightplot=True,                   # Show gene alteration plot on right
       legend=True,                      # Show legend
       cell_background="#dddddd",        # Background color for empty cells
       gap=[0.3, 0.1],                  # Gap between cells (x, y)
       ratio_template="{0:.0%}"        # Format for gene frequency labels
   )