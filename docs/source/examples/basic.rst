Basic Usage
==========

Initialize Project
----------------

.. code-block:: bash

   # Create new project
   popupy init my_project

   # Create project with custom settings
   popupy init my_project --data-dir input --prefix data

Train Model
----------

.. code-block:: python

   from popupy import Settings, Model

   # Load settings
   settings = Settings.from_file('config.yaml')

   # Create and train model
   model = Model(settings)
   model.train(features)