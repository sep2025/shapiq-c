"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

from pathlib import Path
import sys

# It only works if conf.py is run from the docs/source directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


project = "shapiq-c"
copyright_ = "2025, Heeso Park"
author = "Heeso Park"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "myst_parser",
    "myst_nb",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
]


templates_path = ["_templates"]
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/SEP_Logo.png"
autodoc_typehints = "description"
html_title = "[SEP_SoSe2025 / Gruppe C] Game Theoretic Explainable Artificial Intelligence"
html_short_title = "shapiq-c"
