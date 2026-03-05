"""
I/O module for FastFractalAnalyzer v3.

Handles reading of various file formats for fractal analysis.
"""

from .vtk_reader import VTKReader
from .conrec_optimizer import OptimizedCONREC, optimize_conrec_for_vtk

__all__ = [
    "VTKReader",
    "OptimizedCONREC",
    "optimize_conrec_for_vtk",
]