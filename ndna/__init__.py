"""
nDNA Library
============

A mini library for calculating thermodynamic length and spectral curvature metrics
for any Large Language Model from Hugging Face transformers library.

Metrics included:
- Thermodynamic Length: Parameter strain and Fisher-Rao distance methods
- Spectral Curvature: Geometric curvature analysis using logit lens

Usage:
    from ndna import ThermodynamicCalculator, SpectralCalculator
    from ndna.utils import ModelHandler, DatasetHandler
    
    # Or use the high-level API
    from ndna import calculate_all_metrics
"""

from .thermodynamic import ThermodynamicCalculator
from .spectral import SpectralCalculator
from .utils import ModelHandler, DatasetHandler
from .api import calculate_all_metrics, nDNA
from . import datasets

__version__ = "0.1.0"
__author__ = "nDNA Team"

__all__ = [
    "ThermodynamicCalculator", 
    "SpectralCalculator", 
    "ModelHandler", 
    "DatasetHandler",
    "calculate_all_metrics",
    "nDNA",
    "datasets"
]
