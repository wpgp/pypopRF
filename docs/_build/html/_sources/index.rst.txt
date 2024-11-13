# docs/index.rst

pypopRF Documentation
===================

``pypopRF`` is a Python package for population prediction and dasymetric mapping using machine learning techniques.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   examples

Installation
-----------

Install the package using pip:

.. code-block:: bash

   pip install pypoprf

Quick Start
----------

1. Initialize a new project:

   .. code-block:: bash

      pypoprf init my_project

2. Add your data files to the `data` directory:

   .. code-block:: text

      my_project/
      ├── data/
      │   ├── mastergrid.tif
      │   ├── buildingCount.tif
      │   ├── buildingSurface.tif
      │   ├── buildingVolume.tif
      │   └── census.csv
      └── config.yaml

3. Run the analysis:

   .. code-block:: bash

      pypoprf run -c my_project/config.yaml

For more detailed instructions, see :doc:`usage`.