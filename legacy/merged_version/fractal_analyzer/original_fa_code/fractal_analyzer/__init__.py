# __init__.py
from .main import FractalAnalyzer
from .core import FractalBase
from .analysis import BoxCounter
from .visualization import FractalVisualizer

__version__ = "0.24.0"
__all__ = ["FractalAnalyzer", "FractalBase", "BoxCounter", "FractalVisualizer"]
