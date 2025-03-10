# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from recommonmark.parser import CommonMarkParser
sys.path.insert(0, os.path.abspath('../../python'))
# sys.path.insert(0, os.path.abspath('../'))
#sys.path.insert(0, os.path.abspath('../../python/hetu'))
# 指向项目根目录（根据实际结构调整）

# 验证路径
print("Current Python path:", sys.path)


source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']
autodoc_default_options = {'enabled': False}
autosummary_generate = False
#autoclass_content = 'init'

# -- Project information -----------------------------------------------------

project = 'Hetu'
copyright = '2021, Peking University'
author = 'hsword'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary', 
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
autosummary_generate = True

# Register the theme as an extension to generate a sitemap.xml
# extensions.append("sphinx_rtd_theme")
# logo
html_logo = 'figure/logo.png'
html_favicon = 'figure/favicon.ico'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    'style_nav_header_background': 'white',
    'logo_only': True,
    'collapse_navigation': False,
    # 新增以下两个关键参数
    'sticky_navigation': True,      # 固定侧边栏
    'navigation_depth': 4,         # 增加导航层级
}


def setup(app):
    app.add_css_file('css/customized.css')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static'] 
extensions.extend([
     'recommonmark',
     'sphinx_markdown_tables'
 ])
