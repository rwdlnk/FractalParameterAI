# generators/dragon.py
import numpy as np
from typing import Tuple, List

def generate_dragon(level: int) -> Tuple[List, List]:
    """Generate Dragon curve at specified level."""
    
    def dragon_sequence(n):
        """Generate the sequence for the dragon curve."""
        if n == 0:
            return [1]
        
        prev_seq = dragon_sequence(n - 1)
        new_seq = prev_seq.copy()
        new_seq.append(1)
        
        # This is the correct pattern for dragon curve generation
        for i in range(len(prev_seq) - 1, -1, -1):
            new_seq.append(0 if prev_seq[i] == 1 else 1)
        
        return new_seq
    
    # Generate sequence
    sequence = dragon_sequence(level)
    
    # Convert to points
    points = [(0, 0)]
    x, y = 0, 0
    direction = 0  # 0: right, 1: up, 2: left, 3: down
    
    for turn in sequence:
        if turn == 1:  # Right turn
            direction = (direction + 1) % 4
        else:  # Left turn
            direction = (direction - 1) % 4
        
        # Move in current direction
        if direction == 0:
            x += 1
        elif direction == 1:
            y += 1
        elif direction == 2:
            x -= 1
        else:
            y -= 1
        
        points.append((x, y))
    
    # Normalize to unit square
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    scale = max(max_x - min_x, max_y - min_y)
    if scale > 0:
        points = [((p[0] - min_x) / scale, (p[1] - min_y) / scale) for p in points]
    
    segments = []
    for i in range(len(points) - 1):
        segments.append(((points[i][0], points[i][1]), 
                        (points[i+1][0], points[i+1][1])))

    return points, segments
