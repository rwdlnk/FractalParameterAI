# core.py
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from collections import defaultdict

class FractalBase:
    """Base class for fractal analysis."""
    
    THEORETICAL_DIMENSIONS = {
        'koch': np.log(4) / np.log(3),      # ≈ 1.2619
        'sierpinski': np.log(3) / np.log(2), # ≈ 1.5850
        'minkowski': 1.5,                   # Exact value
        'hilbert': 2.0,                     # Space-filling
        'dragon': 1.5236                    # Approximate
    }
    
    def __init__(self, fractal_type: Optional[str] = None):
        """Initialize the fractal base class."""
        self.fractal_type = fractal_type
    
    def read_line_segments(self, filename: str) -> List:
        """Read line segments from a file."""
        start_time = time.time()
        print(f"Reading segments from {filename}...")
        
        segments = []
        with open(filename, 'r') as file:
            for line in file:
                if not line.strip() or line.strip().startswith('#'):
                    continue
                
                try:
                    coords = [float(x) for x in line.replace(',', ' ').split()]
                    if len(coords) == 4:
                        segments.append(((coords[0], coords[1]), (coords[2], coords[3])))
                except ValueError:
                    print(f"Warning: Could not parse line: {line}")
        
        print(f"Read {len(segments)} segments in {time.time() - start_time:.2f} seconds")
        return segments
    
    def write_segments_to_file(self, segments: List, filename: str):
        """Write line segments to a file."""
        start_time = time.time()
        print(f"Writing {len(segments)} segments to file {filename}...")
        
        with open(filename, 'w') as file:
            for (x1, y1), (x2, y2) in segments:
                file.write(f"{x1} {y1} {x2} {y2}\n")
        
        print(f"File writing completed in {time.time() - start_time:.2f} seconds")
    
    def liang_barsky_line_box_intersection(self, x1, y1, x2, y2, xmin, ymin, xmax, ymax):
        """Determine if a line segment intersects with a box using Liang-Barsky algorithm."""
        dx = x2 - x1
        dy = y2 - y1
        
        p = [-dx, dx, -dy, dy]
        q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
        
        if dx == 0 and (x1 < xmin or x1 > xmax):
            return False
        if dy == 0 and (y1 < ymin or y1 > ymax):
            return False
        
        if dx == 0 and dy == 0:
            return xmin <= x1 <= xmax and ymin <= y1 <= ymax
        
        t_min = 0.0
        t_max = 1.0
        
        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return False
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    t_min = max(t_min, t)
                else:
                    t_max = min(t_max, t)
        
        return t_min <= t_max
