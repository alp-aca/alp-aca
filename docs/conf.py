import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import alpaca
import alpaca.plotting.mpl

project = 'ALPaca'
author = 'Jorge Alda, Marta Fuentes Zamoro, Luca Merlo, Xavier Ponce Diaz, Stefano Rigolin'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []
html_static_path = ['_static']
html_theme = 'sphinx_rtd_theme'  # You can change this to your preferred theme
html_title = project
html_short_title = project
html_logo = '_static/logo.png'  # Path to your logo file
html_favicon = '_static/logo.png'
html_theme_options = {}
master_doc = "contents"
copyright = '%Y, ' + author