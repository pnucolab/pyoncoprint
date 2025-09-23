API Reference
=============

.. currentmodule:: pyoncoprint

OncoPrint Class
---------------

.. autoclass:: OncoPrint
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: oncoprint

Parameters
----------

OncoPrint.__init__
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - recurrence_matrix
     - pd.DataFrame or np.ndarray
     - Matrix of mutations/alterations (genes x samples)
   * - genes
     - array-like, optional
     - Gene names for rows
   * - samples
     - array-like, optional
     - Sample names for columns
   * - separator
     - str, default=","
     - Separator for multiple mutations in same cell

OncoPrint.oncoprint
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - markers
     - dict
     - Dictionary mapping mutation types to marker styles
   * - annotations
     - dict, optional
     - Clinical annotations to display
   * - heatmaps
     - dict, optional
     - Continuous data heatmaps
   * - title
     - str, default=""
     - Plot title
   * - gene_sort_method
     - str, default='default'
     - Method for sorting genes ('default', 'unsorted')
   * - sample_sort_method
     - str, default='default'
     - Method for sorting samples ('default', 'unsorted')
   * - figsize
     - list, default=[50, 20]
     - Figure size [width, height]
   * - topplot
     - bool, default=True
     - Show sample mutation frequency plot
   * - rightplot
     - bool, default=True
     - Show gene alteration frequency plot
   * - legend
     - bool, default=True
     - Show legend
   * - cell_background
     - str, default="#dddddd"
     - Background color for empty cells
   * - gap
     - float or list, default=0.3
     - Gap between cells (ratio)
   * - ratio_template
     - str, default="{0:.0%}"
     - Format string for gene frequency labels

Marker Specifications
---------------------

Markers define how different mutation types are displayed:

Rectangle/Fill Markers
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       'marker': 'fill',  # or 'rect'
       'color': 'red',
       'width': 1.0,      # Optional, 0-1 ratio
       'height': 0.5,     # Optional, 0-1 ratio
       'zindex': 1,       # Optional, z-order
       'linewidth': 0     # Optional
   }

Custom Patch Markers
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from matplotlib.patches import Polygon

   {
       'marker': Polygon([[0, 0], [1, 1], [1, 0]]),
       'color': 'green',
       'linewidth': 0,
       'zindex': 1
   }

Scatter Markers
~~~~~~~~~~~~~~~

.. code-block:: python

   {
       'marker': '*',     # Any matplotlib marker
       'color': 'purple',
       's': 100,          # Size
       'lw': 0,           # Line width
       'zindex': 2
   }

Annotation Specifications
-------------------------

Categorical Annotations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       'annotations': pd.DataFrame(...),  # Categorical data
       'colors': {
           'Category1': 'color1',
           'Category2': 'color2'
       },
       'order': 0  # Display order
   }

Numerical Annotations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       'annotations': pd.DataFrame(...),  # Numerical data
       'color': 'orange',  # Single color for gradient
       'order': 1
   }

Heatmap Specifications
----------------------

.. code-block:: python

   {
       'heatmap': pd.DataFrame(...),  # Expression/continuous data
       'cmap': 'RdBu_r',              # Colormap name or object
       'vmin': -3,                    # Optional, minimum value
       'vmax': 3                      # Optional, maximum value
   }

Return Values
-------------

The `oncoprint()` method returns:

.. code-block:: python

   fig, (ax, ax2, ax_top, ax_annot, ax_right, ax_legend)

Where:
- ``fig``: matplotlib Figure object
- ``ax``: Main oncoprint axis
- ``ax2``: Twin axis for gene labels
- ``ax_top``: Top barplot axis (sample frequencies)
- ``ax_annot``: Annotation tracks axis
- ``ax_right``: Right barplot axis (gene frequencies)
- ``ax_legend``: Legend axis