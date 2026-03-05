"""
Fractal Analyzer Package

A comprehensive toolkit for fractal and multifractal analysis with specialized
applications for various domains.
"""

from .main import FractalAnalyzer
from .core import FractalBase
from .analysis import BoxCounter
from .visualization import FractalVisualizer

# Optionally expose RT analyzer if available
try:
    from .applications.rt import RTAnalyzer
    __all__ = ["FractalAnalyzer", "FractalBase", "BoxCounter", "FractalVisualizer", "RTAnalyzer"]
except ImportError:
    # RT module may not be available if dependencies are missing
    __all__ = ["FractalAnalyzer", "FractalBase", "BoxCounter", "FractalVisualizer"]

__version__ = "0.25.0"
