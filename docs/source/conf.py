# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JAX-KR'
copyright = '2023, JAX/Flax Lab'
author = 'JAX/Flax Lab'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



extensions = [
    'sphinx.ext.mathjax',
    'myst_nb',
    'sphinx_design'
]
myst_enable_extensions = ['dollarmath']
source_suffix = ['.rst', '.md', '.ipynb']



templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_logo = '_static/jax-kr.png'
html_theme = 'sphinx_book_theme'  # in conf.py file
html_static_path = ['_static']


# LaTex UNICODE Options --------------------------------
latex_engine = 'xelatex'
latex_elements = {
    'fontpkg': r'''
    \setmainfont{DejaVu Serif}
    \setsansfont{DejaVu Sans}
    \setmonofont{DejaVu Sans Mono}
    ''',
    'inputenc': '',
    'utf8extra': '',
}

