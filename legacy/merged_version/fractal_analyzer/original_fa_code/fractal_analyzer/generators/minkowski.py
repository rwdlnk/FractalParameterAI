# generators/minkowski.py
import numpy as np
from typing import Tuple, List

def generate_minkowski(level: int) -> Tuple[List, List]:
    """Generate Minkowski sausage/coastline at specified level."""
    
    def minkowski_recursive(x1, y1, x2, y2, level):
        """Recursively generate Minkowski curve."""
        if level == 0:
            return [(x1, y1), (x2, y2)]
        else:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            unit_x = dx / length
            unit_y = dy / length
            
            # Define the 8 segments of the Minkowski curve
            factor = 1 / 4  # Each segment is 1/4 of the original
            
            # Calculate points
            p0 = (x1, y1)
            p1 = (x1 + unit_x * length * factor, y1 + unit_y * length * factor)
            p2 = (p1[0] - unit_y * length * factor, p1[1] + unit_x * length * factor)
            p3 = (p2[0] + unit_x * length * factor, p2[1] + unit_y * length * factor)
            p4 = (p3[0] + unit_y * length * factor, p3[1] - unit_x * length * factor)
            p5 = (p4[0] + unit_x * length * factor, p4[1] + unit_y * length * factor)
            p6 = (p5[0] - unit_y * length * factor, p5[1] + unit_x * length * factor)
            p7 = (p6[0] + unit_x * length * factor, p6[1] + unit_y * length * factor)
            p8 = (x2, y2)
            
            # Recursively generate segments
            points = []
            points.extend(minkowski_recursive(p0[0], p0[1], p1[0], p1[1], level - 1)[:-1])
            points.extend(minkowski_recursive(p1[0], p1[1], p2[0], p2[1], level - 1)[:-1])
            points.extend(minkowski_recursive(p2[0], p2[1], p3[0], p3[1], level - 1)[:-1])
            points.extend(minkowski_recursive(p3[0], p3[1], p4[0], p4[1], level - 1)[:-1])
            points.extend(minkowski_recursive(p4[0], p4[1], p5[0], p5[1], level - 1)[:-1])
            points.extend(minkowski_recursive(p5[0], p5[1], p6[0], p6[1], level - 1)[:-1])
            points.extend(minkowski_recursive(p6[0], p6[1], p7[0], p7[1], level - 1)[:-1])
            points.extend(minkowski_recursive(p7[0], p7[1], p8[0], p8[1], level - 1))
        
            return points
    
    points = minkowski_recursive(0, 0, 1, 0, level)
    
    segments = []
    for i in range(len(points) - 1):
        segments.append(((points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1])))
    
    return points, segments
