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
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    'display_version': True,
}

# Custom CSS
def setup(app):
    app.add_css_file('custom.css')

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'wpgp',
    'github_repo': 'pypopRF',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# Footer info
html_show_sphinx = False
html_show_copyright = True