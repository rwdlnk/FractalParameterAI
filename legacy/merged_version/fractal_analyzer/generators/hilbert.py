# generators/hilbert.py
import numpy as np
from typing import Tuple, List

def generate_hilbert(level: int) -> Tuple[List, List]:
    """Generate Hilbert curve at specified level."""
    
    points = []
    
    def hilbert_recursive(x0, y0, xi, xj, yi, yj, n):
        """Recursively generate points for the Hilbert curve."""
        if n <= 0:
            x = x0 + (xi + yi) / 2
            y = y0 + (xj + yj) / 2
            points.append((x, y))
        else:
            hilbert_recursive(x0, y0, yi / 2, yj / 2, xi / 2, xj / 2, n - 1)
            hilbert_recursive(x0 + xi / 2, y0 + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2, n - 1)
            hilbert_recursive(x0 + xi / 2 + yi / 2, y0 + xj / 2 + yj / 2, xi / 2, xj / 2, yi / 2, yj / 2, n - 1)
            hilbert_recursive(x0 + xi / 2 + yi, y0 + xj / 2 + yj, -yi / 2, -yj / 2, -xi / 2, -xj / 2, n - 1)
    
    # Generate the points
    hilbert_recursive(0, 0, 1, 0, 0, 1, level)
    
    # Create segments from points
    segments = []
    for i in range(len(points) - 1):
        segments.append(((points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1])))
    
    return points, segments
