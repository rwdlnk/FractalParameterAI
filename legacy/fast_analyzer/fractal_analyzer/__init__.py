"""
FastFractalAnalyzer v3: High-performance fractal dimension analysis.

A complete rewrite focused on vectorized NumPy operations for maximum performance.
"""

__version__ = "3.0.0"
__author__ = "Rod Douglass"

from .core.fractal_analyzer import FastFractalAnalyzer
from .core.data_types import FractalResult, SegmentArray
from .core.box_counting import VectorizedBoxCounter

__all__ = [
    "FastFractalAnalyzer",
    "FractalResult",
    "SegmentArray",
    "VectorizedBoxCounter",
]