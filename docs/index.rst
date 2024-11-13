# docs/index.rst

pypopRF Documentation
===================

.. image:: https://img.shields.io/pypi/v/pypoprf
   :target: https://pypi.org/project/pypoprf/
   :alt: PyPI version

.. image:: https://img.shields.io/github/license/wpgp/pypopRF
   :target: https://github.com/wpgp/pypopRF/blob/main/LICENSE
   :alt: License

.. image:: https://readthedocs.org/projects/pypoprf/badge/?version=latest
   :target: https://pypoprf.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Overview
--------

``pypopRF`` is a Python package for population prediction and dasymetric mapping using machine learning techniques. It provides a comprehensive toolkit for processing geospatial data, training models, and generating high-resolution population distribution maps.

Key Features
-----------

* **Feature Extraction**: Process multiple geospatial covariates
* **Machine Learning**: Random Forest-based population prediction
* **Dasymetric Mapping**: High-resolution population redistribution
* **Performance**: Parallel processing support for large datasets
* **Visualization**: Tools for analysis and validation
* **CLI**: Command-line interface for easy project management

Quick Installation
----------------

.. code-block:: bash

   pip install pypoprf

Quick Start
----------

.. code-block:: bash

   # Create new project
   pypoprf init my_project

   # Run analysis
   pypoprf run -c my_project/config.yaml

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage
   examples

Support
-------

* Issue Tracker: https://github.com/wpgp/pypopRF/issues
* Source Code: https://github.com/wpgp/pypopRF

License
-------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/wpgp/pypopRF/blob/main/LICENSE>`_ file for details.

Citation
--------

If you use pypopRF in your research, please cite:

.. code-block:: text

   @software{pypoprf2024,
     author = {WorldPop SDI},
     title = {pypopRF: Population Prediction and Dasymetric Mapping Tool},
     year = {2024},
     publisher = {GitHub},
     url = {https://github.com/wpgp/pypopRF}
   }