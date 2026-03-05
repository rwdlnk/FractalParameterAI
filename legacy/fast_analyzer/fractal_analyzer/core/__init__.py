"""
FastFractalAnalyzer v3 core module.

High-performance fractal analysis components.
"""

from .fractal_analyzer import FastFractalAnalyzer
from .data_types import SegmentArray, FractalResult, BoundingBox
from .box_counting import VectorizedBoxCounter

__all__ = [
    "FastFractalAnalyzer",
    "SegmentArray",
    "FractalResult",
    "BoundingBox",
    "VectorizedBoxCounter",
]