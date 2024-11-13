# docs/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'pypopRF'
copyright = '2024, WorldPop SDI'
author = 'Borys Nosatiuk, Rhorom Priyatikanto'
version = '0.1.0'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']