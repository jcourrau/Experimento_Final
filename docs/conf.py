"""Sphinx configuration for the hand gesture recognition project."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

project = "Clasificación de Gestos de Mano con CNN"
author = "Luis Gabriel Corrales Mora, Jason Courrau Madrigal"
copyright = "2026, Luis Gabriel Corrales Mora, Jason Courrau Madrigal"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["config.cache"]

autodoc_mock_imports = ["cv2", "numpy", "torch", "torch.nn"]
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

myst_heading_anchors = 3
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

html_theme = "furo"
html_title = "Gestos de Mano con CNN"
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#007f73",
        "color-brand-content": "#005c53",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#99f6e4",
    },
}

html_extra_path = [
    "../outputs/models/gesture_recognition_(cnn_b).pt",
]
