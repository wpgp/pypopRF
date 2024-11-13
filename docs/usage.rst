# docs/usage.rst

Usage Guide
==========

Project Setup
------------

1. Initialize Project
^^^^^^^^^^^^^^^^^^^

Create a new project with default settings:

.. code-block:: bash

   pypoprf init my_project

The command will create:
- A project directory
- A configuration file (config.yaml)
- Data and output directories

2. Prepare Data
^^^^^^^^^^^^^

Place your input files in the data directory:

- Covariate rasters (GeoTIFF format)
- Mastergrid raster
- Census data (CSV format)

3. Configure Project
^^^^^^^^^^^^^^^^^

Edit config.yaml to match your data:

.. code-block:: yaml

   work_dir: "."
   data_dir: "data"

   # Input data paths
   covariates:
     cnt: "buildingCount.tif"
     srf: "buildingSurface.tif"
     vol: "buildingVolume.tif"
   mastergrid: "mastergrid.tif"
   census_data: "census.csv"

   # Census data columns
   census_pop_column: "pop"
   census_id_column: "id"

4. Run Analysis
^^^^^^^^^^^^^

Basic run:

.. code-block:: bash

   pypoprf run -c my_project/config.yaml

With verbose output:

.. code-block:: bash

   pypoprf run -c my_project/config.yaml -v

Skip visualization:

.. code-block:: bash

   pypoprf run -c my_project/config.yaml --no-viz

Output Files
-----------

The analysis creates several output files in the output directory:

- features.csv: Extracted features for model training
- model.pkl.gz: Trained Random Forest model
- prediction.tif: Raw population probability surface
- normalized_census.tif: Normalized census populations
- dasymetric.tif: Final high-resolution population distribution
- visualization.png: Multi-panel visualization of results

Common Issues
------------

1. GDAL Import Errors:
   - Ensure GDAL is installed system-wide
   - Check Python GDAL bindings match system version

2. Memory Issues:
   - Adjust block_size in config.yaml
   - Reduce max_workers for parallel processing

3. CRS Mismatches:
   - Ensure all input rasters share same CRS
   - Use gdalwarp to reproject if needed