Installation Guide
================

Prerequisites
-----------

Before installing pypopRF, ensure you have the required system dependencies:

Linux (Ubuntu/Debian)
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y gdal-bin libgdal-dev gcc

MacOS
^^^^^

.. code-block:: bash

   brew install gdal

Windows
^^^^^^^

Install OSGeo4W from https://trac.osgeo.org/osgeo4w/

Installation Methods
-----------------

From PyPI
^^^^^^^^

.. code-block:: bash

   pip install pypoprf

From Source
^^^^^^^^^

.. code-block:: bash

   git clone https://github.com/wpgp/pypopRF.git
   cd pypopRF
   pip install -e .