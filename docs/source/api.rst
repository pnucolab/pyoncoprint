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
     - Matrix of mutations/alterations (genes x samples). If DataFrame, genes should be index and samples should be columns
   * - genes
     - array-like, optional
     - Gene names for rows. If not provided, extracted from DataFrame index or auto-generated
   * - samples
     - array-like, optional
     - Sample names for columns. If not provided, extracted from DataFrame columns or auto-generated
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

Markers define how different mutation types are displayed. Each marker must have a 'color' field.

Rectangle/Fill Markers
~~~~~~~~~~~~~~~~~~~~~~

For rectangular patches that fill the cell:

.. code-block:: python

   {
       'marker': 'fill',  # or 'rect'
       'color': 'red',    # Required
       'width': 1.0,      # Optional, 0-1 ratio (default: 1.0)
       'height': 0.5,     # Optional, 0-1 ratio (default: 1.0)
       'zindex': 1,       # Optional, z-order for layering (default: 1)
       'linewidth': 0     # Optional, border width (default: 0)
   }

Custom Patch Markers
~~~~~~~~~~~~~~~~~~~~

For custom shapes using matplotlib patches:

.. code-block:: python

   from matplotlib.patches import Polygon

   {
       'marker': Polygon([[0, 0], [1, 1], [1, 0]]),  # Triangle
       'color': 'green',      # Required
       'width': 1.0,          # Optional, scaling factor
       'height': 1.0,         # Optional, scaling factor
       'linewidth': 0,        # Optional
       'zindex': 1            # Optional
   }

Scatter Markers
~~~~~~~~~~~~~~~

For point markers using matplotlib scatter:

.. code-block:: python

   {
       'marker': '*',     # Any matplotlib marker symbol
       'color': 'purple', # Required
       's': 100,          # Optional, marker size
       'lw': 0,           # Optional, edge line width
       'zindex': 2        # Optional
   }

Note: The 'zindex' parameter controls layering - lower values are drawn first (background), higher values on top.

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
- ``ax2``: Twin axis for gene labels (right side)
- ``ax_top``: Top barplot axis (sample frequencies), None if topplot=False
- ``ax_annot``: Annotation tracks axis, None if no annotations
- ``ax_right``: Right barplot axis (gene frequencies), None if rightplot=False
- ``ax_legend``: Legend axis, None if legend=False

Deprecated Parameters
---------------------

The following parameters are deprecated but still supported for backward compatibility:

- ``is_topplot``: Use ``topplot`` instead
- ``is_rightplot``: Use ``rightplot`` instead
- ``is_legend``: Use ``legend`` instead

These will print a warning message if used.