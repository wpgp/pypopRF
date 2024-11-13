# docs/examples.rst

Examples
========

Basic Usage
----------

1. Simple Project
^^^^^^^^^^^^^^^

Initialize and run a basic project:

.. code-block:: bash

   # Create project
   pypoprf init basic_project

   # Add your data files to basic_project/data/

   # Run analysis
   pypoprf run -c basic_project/config.yaml

2. Custom Covariates
^^^^^^^^^^^^^^^^^^^

Initialize project with specific covariates:

.. code-block:: bash

   pypoprf init custom_project \
     --covariates population \
     --covariates elevation \
     --covariates slope

3. Large Dataset
^^^^^^^^^^^^^^

For large datasets, adjust processing parameters in config.yaml:

.. code-block:: yaml

   # Reduce memory usage with smaller blocks
   block_size: [256, 256]

   # Control parallel processing
   max_workers: 4

Common Workflows
--------------

1. Population Density Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Example config.yaml for population density analysis:

.. code-block:: yaml

   work_dir: "."
   data_dir: "data"

   covariates:
     pop: "population.tif"
     bld: "buildings.tif"
     rds: "roads.tif"

   mastergrid: "zones.tif"
   census_data: "census.csv"
   census_pop_column: "total_pop"
   census_id_column: "zone_id"

2. Urban Population Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^

Example workflow for urban areas:

.. code-block:: bash

   # Create project
   pypoprf init urban_project

   # Add urban-specific covariates
   cp building_density.tif urban_project/data/
   cp road_density.tif urban_project/data/
   cp nightlights.tif urban_project/data/

   # Update config.yaml with urban covariates
   # Run analysis
   pypoprf run -c urban_project/config.yaml