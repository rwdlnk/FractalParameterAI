# generators/sierpinski.py
import numpy as np
from typing import Tuple, List

def generate_sierpinski(level: int) -> Tuple[List, List]:
    """Generate Sierpinski triangle at specified level."""
    
    def generate_triangles(p1, p2, p3, level):
        """Recursively generate Sierpinski triangles."""
        if level == 0:
            # Return line segments of the triangle
            return [(p1, p2), (p2, p3), (p3, p1)]
        else:
            # Calculate midpoints
            mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            mid2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
            mid3 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
            
            # Recursively generate three smaller triangles
            segments = []
            segments.extend(generate_triangles(p1, mid1, mid3, level - 1))
            segments.extend(generate_triangles(mid1, p2, mid2, level - 1))
            segments.extend(generate_triangles(mid3, mid2, p3, level - 1))
            
            return segments
    
    # Initial equilateral triangle
    p1 = (0, 0)
    p2 = (1, 0)
    p3 = (0.5, np.sqrt(3) / 2)
    
    segments = generate_triangles(p1, p2, p3, level)
    
    # Extract unique points
    points = []
    for seg in segments:
        if seg[0] not in points:
            points.append(seg[0])
        if seg[1] not in points:
            points.append(seg[1])
    
    return points, segments
