# main.py
import numpy as np
from typing import Tuple, List, Dict, Optional
import importlib

from .core import FractalBase
from .analysis import BoxCounter
from .visualization import FractalVisualizer

class FractalAnalyzer:
    """Universal fractal dimension analysis tool."""
    
    def __init__(self, fractal_type: Optional[str] = None):
        """
        Initialize the fractal analyzer.
        
        Args:
            fractal_type: Type of fractal if known (koch, sierpinski, etc.)
        """
        self.base = FractalBase(fractal_type)
        self.box_counter = BoxCounter(self.base)
        self.visualizer = FractalVisualizer(fractal_type, self.base)  # Pass base reference
        self.fractal_type = fractal_type
    
    def generate_fractal(self, type_: str, level: int) -> Tuple[List, List]:
        """
        Generate points/segments based on fractal type.
        
        Args:
            type_: Fractal type (koch, sierpinski, etc.)
            level: Iteration level
            
        Returns:
            Tuple of (points, segments)
        """
        supported_types = ['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon']
        
        if type_ not in supported_types:
            raise ValueError(f"Unknown fractal type: {type_}")
        
        # Dynamically import the appropriate generator module
        try:
            generator_module = importlib.import_module(f'.generators.{type_}', package='fractal_analyzer')
            generate_function = getattr(generator_module, f'generate_{type_}')
            return generate_function(level)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load generator for {type_}: {str(e)}")
    
    def read_line_segments(self, filename: str) -> List:
        """Read line segments from a file."""
        return self.base.read_line_segments(filename)
    
    def write_segments_to_file(self, segments: List, filename: str):
        """Write line segments to a file."""
        return self.base.write_segments_to_file(segments, filename)
    
    def calculate_fractal_dimension(self, segments, min_box_size=0.001, max_box_size=None, 
                                  box_size_factor=2.0):
        """Calculate fractal dimension for a set of segments."""
        # Auto-determine max box size if not provided
        if max_box_size is None:
            min_x = min(min(s[0][0], s[1][0]) for s in segments)
            max_x = max(max(s[0][0], s[1][0]) for s in segments)
            min_y = min(min(s[0][1], s[1][1]) for s in segments)
            max_y = max(max(s[0][1], s[1][1]) for s in segments)
            
            extent = max(max_x - min_x, max_y - min_y)
            max_box_size = extent / 2
            print(f"Auto-determined max box size: {max_box_size}")
        
        # Perform box counting
        box_sizes, box_counts, bounding_box = self.box_counter.box_counting_optimized(
            segments, min_box_size, max_box_size, box_size_factor)
        
        # Calculate dimension
        fractal_dimension, error, intercept = self.box_counter.calculate_fractal_dimension(
            box_sizes, box_counts)
        
        return fractal_dimension, error, box_sizes, box_counts, bounding_box, intercept
    
    def plot_results(self, segments, box_sizes, box_counts, fractal_dimension, error, 
                    bounding_box, plot_boxes=False, level=None, custom_filename=None):
        """Plot the fractal and its dimension analysis."""
        return self.visualizer.plot_fractal_curve(
            segments, bounding_box, plot_boxes, box_sizes, box_counts, 
            custom_filename, level)
