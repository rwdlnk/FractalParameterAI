# generators/koch.py
import numpy as np
import math
from typing import Tuple, List
from numba import jit

def generate_koch(level: int) -> Tuple[List, List]:
    """Generate Koch curve at specified level."""
    
    @jit(nopython=True)
    def koch_points_jit(x1, y1, x2, y2, level, points_array, idx):
        """JIT-compiled Koch curve generator."""
        if level == 0:
            points_array[idx] = (x1, y1)
            return idx + 1
        else:
            angle = math.pi / 3
            x3 = x1 + (x2 - x1) / 3
            y3 = y1 + (y2 - y1) / 3
            x4 = (x1 + x2) / 2 + (y2 - y1) * math.sin(angle) / 3
            y4 = (y1 + y2) / 2 - (x2 - x1) * math.sin(angle) / 3
            x5 = x1 + 2 * (x2 - x1) / 3
            y5 = y1 + 2 * (y2 - y1) / 3
            
            idx = koch_points_jit(x1, y1, x3, y3, level - 1, points_array, idx)
            idx = koch_points_jit(x3, y3, x4, y4, level - 1, points_array, idx)
            idx = koch_points_jit(x4, y4, x5, y5, level - 1, points_array, idx)
            idx = koch_points_jit(x5, y5, x2, y2, level - 1, points_array, idx)
            
            return idx
    
    num_points = 4**level + 1
    points_array = np.zeros((num_points, 2), dtype=np.float64)
    
    final_idx = koch_points_jit(0, 0, 1, 0, level, points_array, 0)
    points_array[final_idx] = (1, 0)
    points = points_array[:final_idx+1]
    
    segments = []
    for i in range(len(points) - 1):
        segments.append(((points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1])))
    
    return points, segments
