Examples
========

Example Data
------------

The following example datasets are available for download and testing:

* `TCGA Lung Adenocarcinoma Data <https://raw.githubusercontent.com/pnucolab/pyoncoprint/main/example_data/tcga.tsv>`_ - Real cancer genomics data from TCGA
* `Test Dataset 1 <https://raw.githubusercontent.com/pnucolab/pyoncoprint/main/example_data/test1.tsv>`_ - Sample mutation data for testing
* `Test Dataset 2 <https://raw.githubusercontent.com/pnucolab/pyoncoprint/main/example_data/test2.tsv>`_ - Additional test data
* `Example Notebook <https://raw.githubusercontent.com/pnucolab/pyoncoprint/main/example.ipynb>`_ - Complete tutorial notebook

Download these files to follow along with the examples below.

Basic OncoPrint
---------------

Creating a simple OncoPrint with common mutation types:

.. code-block:: python

   import pyoncoprint
   import pandas as pd
   import matplotlib.pyplot as plt

   # Sample mutation data
   data = pd.DataFrame({
       'TCGA-01': ['Missense', '', 'Amplification', '', 'Missense'],
       'TCGA-02': ['Truncating', 'Missense', '', '', ''],
       'TCGA-03': ['', 'Amplification', 'Deep Deletion', 'Missense', ''],
       'TCGA-04': ['Missense', '', '', 'Truncating', 'Amplification'],
       'TCGA-05': ['', 'Missense', 'Amplification', '', 'Truncating']
   }, index=['TP53', 'EGFR', 'PTEN', 'KRAS', 'PIK3CA'])

   # Define visual markers
   markers = {
       'Missense': {'marker': 'fill', 'color': 'forestgreen', 'height': 0.5},
       'Truncating': {'marker': 'fill', 'color': 'black', 'height': 0.5},
       'Amplification': {'marker': 'fill', 'color': 'firebrick'},
       'Deep Deletion': {'marker': 'fill', 'color': 'royalblue'}
   }

   # Create OncoPrint
   op = pyoncoprint.OncoPrint(data)
   fig, axes = op.oncoprint(markers, figsize=[15, 8], title="Cancer Gene Mutations")
   plt.show()

Multiple Mutations per Cell
----------------------------

PyOncoPrint supports multiple mutations in the same cell by separating them with commas:

.. code-block:: python

   import pyoncoprint
   import pandas as pd

   # Data with multiple mutations per cell
   data = pd.DataFrame({
       'Sample1': ['Missense,Amplification', '', 'Truncating'],
       'Sample2': ['Missense', 'Deep Deletion', ''],
       'Sample3': ['', 'Missense,Splice', 'Amplification']
   }, index=['TP53', 'EGFR', 'PTEN'])

   # Define markers - zindex controls layering
   markers = {
       'Amplification': {'marker': 'fill', 'color': 'red', 'zindex': 0},
       'Deep Deletion': {'marker': 'fill', 'color': 'blue', 'zindex': 0},
       'Missense': {'marker': 'fill', 'color': 'green', 'height': 0.5, 'zindex': 1},
       'Truncating': {'marker': 'fill', 'color': 'black', 'height': 0.5, 'zindex': 1},
       'Splice': {'marker': 'fill', 'color': 'orange', 'height': 0.5, 'zindex': 1}
   }

   op = pyoncoprint.OncoPrint(data)
   fig, axes = op.oncoprint(markers, figsize=[12, 8])

TCGA-style OncoPrint
--------------------

Recreating a TCGA-style visualization with multiple data types:

.. code-block:: python

   import numpy as np
   from matplotlib.patches import Polygon

   # More complex mutation data
   mutation_data = pd.DataFrame({
       'Sample_' + str(i): np.random.choice(
           ['', 'Missense', 'Truncating', 'Splice', 'Inframe'],
           size=10,
           p=[0.5, 0.2, 0.1, 0.1, 0.1]
       ) for i in range(20)
   }, index=['Gene_' + str(i) for i in range(10)])

   # CNA data (overlapping with mutations)
   for sample in mutation_data.columns[:10]:
       for gene in ['Gene_0', 'Gene_2', 'Gene_5']:
           if np.random.random() < 0.3:
               current = mutation_data.loc[gene, sample]
               if current:
                   mutation_data.loc[gene, sample] = current + ',Amplification'
               else:
                   mutation_data.loc[gene, sample] = 'Amplification'

   # TCGA-style markers
   markers = {
       'Amplification': {
           'marker': 'fill',
           'color': 'red',
           'zindex': 0
       },
       'Deep Deletion': {
           'marker': 'fill',
           'color': 'blue',
           'zindex': 0
       },
       'Missense': {
           'marker': Polygon([[0, 0], [1, 1], [1, 0]]),
           'color': 'green',
           'linewidth': 0,
           'zindex': 1
       },
       'Truncating': {
           'marker': 'fill',
           'color': 'black',
           'height': 0.5,
           'zindex': 1
       },
       'Splice': {
           'marker': 'fill',
           'color': 'darkorange',
           'height': 0.5,
           'zindex': 1
       },
       'Inframe': {
           'marker': 'fill',
           'color': 'brown',
           'height': 0.5,
           'zindex': 1
       }
   }

   op = pyoncoprint.OncoPrint(mutation_data)
   fig, axes = op.oncoprint(
       markers,
       figsize=[25, 12],
       title="TCGA-style OncoPrint",
       gap=[0.3, 0.1]
   )
   plt.show()

Multi-track Visualization
-------------------------

Combining mutations with clinical data and expression:

.. code-block:: python

   # Mutation data
   mutations = pd.DataFrame({
       f'Patient_{i}': np.random.choice(
           ['', 'Missense', 'Truncating'],
           size=8,
           p=[0.6, 0.3, 0.1]
       ) for i in range(30)
   }, index=['BRAF', 'NRAS', 'NF1', 'CDKN2A', 'TP53', 'PTEN', 'ARID2', 'KIT'])

   # Clinical annotations
   stage = pd.DataFrame(
       np.random.choice(['I', 'II', 'III', 'IV'], size=30),
       columns=mutations.columns,
       index=['Stage']
   )

   mutation_count = pd.DataFrame(
       np.random.poisson(50, size=30),
       columns=mutations.columns,
       index=['TMB']
   )

   # Expression heatmap
   expression = pd.DataFrame(
       np.random.randn(8, 30),
       index=mutations.index,
       columns=mutations.columns
   )

   # Methylation heatmap
   methylation = pd.DataFrame(
       np.random.beta(2, 5, size=(8, 30)),
       index=mutations.index,
       columns=mutations.columns
   )

   # Configure visualizations
   markers = {
       'Missense': {'marker': 'fill', 'color': 'green', 'height': 0.5},
       'Truncating': {'marker': 'fill', 'color': 'black', 'height': 0.5}
   }

   annotations = {
       'Cancer Stage': {
           'annotations': stage,
           'colors': {
               'I': 'lightgreen',
               'II': 'yellow',
               'III': 'orange',
               'IV': 'red'
           },
           'order': 0
       },
       'Tumor Mutation Burden': {
           'annotations': mutation_count,
           'color': 'purple',
           'order': 1
       }
   }

   heatmaps = {
       'Expression (Z-score)': {
           'heatmap': expression,
           'cmap': 'RdBu_r',
           'vmin': -3,
           'vmax': 3
       },
       'Methylation (Beta)': {
           'heatmap': methylation,
           'cmap': 'Blues',
           'vmin': 0,
           'vmax': 1
       }
   }

   # Create comprehensive OncoPrint
   op = pyoncoprint.OncoPrint(mutations)
   fig, axes = op.oncoprint(
       markers,
       annotations=annotations,
       heatmaps=heatmaps,
       figsize=[40, 20],
       title="Multi-track Cancer Genomics Visualization"
   )
   plt.tight_layout()
   plt.show()

Custom Sorting
--------------

Controlling gene and sample ordering:

.. code-block:: python

   # Create data with specific patterns
   data = pd.DataFrame({
       'S1': ['Missense', 'Missense', '', ''],
       'S2': ['Missense', '', 'Truncating', ''],
       'S3': ['', '', 'Truncating', 'Missense'],
       'S4': ['', 'Missense', '', 'Missense'],
       'S5': ['Missense', 'Missense', 'Truncating', 'Missense']
   }, index=['Gene_A', 'Gene_B', 'Gene_C', 'Gene_D'])

   markers = {
       'Missense': {'marker': 'fill', 'color': 'green'},
       'Truncating': {'marker': 'fill', 'color': 'black'}
   }

   op = pyoncoprint.OncoPrint(data)

   # Default sorting (by frequency)
   fig1, _ = op.oncoprint(
       markers,
       figsize=[10, 6],
       title="Default Sorting",
       gene_sort_method='default',
       sample_sort_method='default'
   )

   # No sorting (original order)
   fig2, _ = op.oncoprint(
       markers,
       figsize=[10, 6],
       title="Original Order",
       gene_sort_method='unsorted',
       sample_sort_method='unsorted'
   )

   plt.show()

Publication-ready Figure
------------------------

Creating a high-quality figure for publication:

.. code-block:: python

   # Load your actual data here
   data = pd.DataFrame({
       'P1': ['Missense', '', 'Amplification', ''],
       'P2': ['', 'Truncating', '', 'Missense'],
       'P3': ['Missense', 'Missense', 'Deep Deletion', ''],
       'P4': ['Amplification', '', '', 'Truncating'],
       'P5': ['', 'Missense', 'Amplification', ''],
       'P6': ['Truncating', '', '', 'Missense'],
       'P7': ['', '', 'Deep Deletion', ''],
       'P8': ['Missense', 'Amplification', '', '']
   }, index=['TP53', 'EGFR', 'PTEN', 'KRAS'])

   # Publication-quality markers
   markers = {
       'Missense': {
           'marker': 'fill',
           'color': '#2E7D32',  # Dark green
           'height': 0.5,
           'zindex': 1
       },
       'Truncating': {
           'marker': 'fill',
           'color': '#000000',  # Black
           'height': 0.5,
           'zindex': 1
       },
       'Amplification': {
           'marker': 'fill',
           'color': '#B71C1C',  # Dark red
           'zindex': 0
       },
       'Deep Deletion': {
           'marker': 'fill',
           'color': '#0D47A1',  # Dark blue
           'zindex': 0
       }
   }

   # Create figure
   op = pyoncoprint.OncoPrint(data)
   fig, axes = op.oncoprint(
       markers,
       figsize=[12, 6],
       title="",  # No title for publication
       gap=[0.2, 0.15],
       cell_background="#f5f5f5",
       ratio_template="{0:.0%}"
   )

   # Customize for publication
   fig.patch.set_facecolor('white')

   # Save as high-resolution image
   plt.savefig('oncoprint_publication.pdf', dpi=300, bbox_inches='tight')
   plt.savefig('oncoprint_publication.png', dpi=300, bbox_inches='tight')
   plt.show()

Loading Data from Files
-----------------------

Working with real data files:

.. code-block:: python

   # Download and load the TCGA example data
   import pandas as pd
   import numpy as np

   # Load TCGA data (download from the link above)
   df = pd.read_csv('tcga.tsv', sep='\t', header=0)

   # Extract oncoprint data (mutations and CNAs)
   df_oncoprint = df[df['track_type'].isin(['MUTATIONS', 'CNA'])].drop(columns=['track_type']).set_index('track_name').fillna('')

   # Clean up mutation names
   df_oncoprint.replace('amp_rec', 'Amplification', inplace=True)
   df_oncoprint.replace('homdel_rec', 'Deep Deletion', inplace=True)
   df_oncoprint.replace('splice', 'Splice Mutation', inplace=True)

   # Define markers for TCGA data
   markers = {
       'Amplification': {'marker': 'fill', 'color': 'red', 'zindex': 0},
       'Deep Deletion': {'marker': 'fill', 'color': 'blue', 'zindex': 0},
       'Missense Mutation': {'marker': 'fill', 'color': 'green', 'height': 0.5, 'zindex': 1},
       'Truncating mutation': {'marker': 'fill', 'color': 'black', 'height': 0.5, 'zindex': 1},
       'Splice Mutation': {'marker': 'fill', 'color': 'orange', 'height': 0.5, 'zindex': 1}
   }

   # Create OncoPrint
   op = pyoncoprint.OncoPrint(df_oncoprint)
   fig, axes = op.oncoprint(markers, figsize=[30, 15], title="TCGA Lung Adenocarcinoma")

Working with other data formats:

.. code-block:: python

   # From CSV file
   mutation_df = pd.read_csv('mutations.csv', index_col=0)

   # From TSV file (general format)
   data = pd.read_csv('data.tsv', sep='\\t', index_col=0)

   # Process and clean data
   processed_data = data.replace({'NaN': '', np.nan: ''})

   # Create OncoPrint
   op = pyoncoprint.OncoPrint(processed_data)
   fig, axes = op.oncoprint(markers, figsize=[30, 15])