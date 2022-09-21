# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Yolov5 Optimization '
copyright = '2022, chentzj'
author = 'chentzj'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_material'

html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'Yolov5 Optimization',

    # Set you GA account ID to enable tracking
    #'google_analytics_account': 'UA-136029994-1',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    #'base_url': 'https://nni.readthedocs.io/',

    # Set the color and the accent color
    # Remember to update static/css/material_custom.css when this is updated.
    # Set those colors in layout.html.
    'color_primary': 'custom',
    'color_accent': 'custom',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/Raychen0617/yolov5_optimization',
    'repo_name': 'GitHub',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 5,

    # Expand all toc so that they can be dynamically collapsed
    'globaltoc_collapse': False,

    'version_dropdown': True,
    # This is a placeholder, which should be replaced later.
    'version_info': {
        'current': '/'
    },

    # Text to appear at the top of the home page in a "hero" div.
    #'heroes': {
    #    'index': 'An open source AutoML toolkit for hyperparameter optimization, neural architecture search, '
    #             'model compression and feature engineering.'
    #}
}

html_static_path = ['_static']
