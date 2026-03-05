# analysis.py
import numpy as np
import time
from scipy import stats
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
from .core import FractalBase

class BoxCounter:
    """Box counting implementation for fractal dimension analysis."""
    
    def __init__(self, fractal_base: FractalBase):
        """Initialize with reference to base class."""
        self.base = fractal_base
    
    def create_spatial_index(self, segments, min_x, min_y, max_x, max_y, cell_size):
        """Create a spatial index to speed up intersection checks."""
        start_time = time.time()
        print("Creating spatial index...")
        
        # Calculate grid dimensions
        grid_width = max(1, int(np.ceil((max_x - min_x) / cell_size)))
        grid_height = max(1, int(np.ceil((max_y - min_y) / cell_size)))
        
        # Debug information
        print(f"  Grid dimensions: {grid_width} x {grid_height}")
        print(f"  Total cells: {grid_width * grid_height}")
        print(f"  Cell size: {cell_size}")
        print(f"  Bounds: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
        # Check for extremely large grids
        if grid_width * grid_height > 1000000:
            print(f"WARNING: Grid is very large ({grid_width * grid_height} cells). This may take a while...")
        
        # Create the spatial index
        segment_grid = defaultdict(list)
        
        # Add progress reporting for large datasets
        segment_count = len(segments)
        report_interval = max(1, segment_count // 10)  # Report every 10%
        
        for i, ((x1, y1), (x2, y2)) in enumerate(segments):
            if i % report_interval == 0:
                print(f"  Progress: {i}/{segment_count} segments processed ({i*100//segment_count}%)")
            
            # Determine which grid cells this segment might intersect
            min_cell_x = max(0, int((min(x1, x2) - min_x) / cell_size))
            max_cell_x = min(grid_width - 1, int((max(x1, x2) - min_x) / cell_size))
            min_cell_y = max(0, int((min(y1, y2) - min_y) / cell_size))
            max_cell_y = min(grid_height - 1, int((max(y1, y2) - min_y) / cell_size))
            
            # Add segment to all relevant grid cells
            for cell_x in range(min_cell_x, max_cell_x + 1):
                for cell_y in range(min_cell_y, max_cell_y + 1):
                    segment_grid[(cell_x, cell_y)].append(i)
        
        print(f"Spatial index created in {time.time() - start_time:.2f} seconds")
        print(f"Total grid cells with segments: {len(segment_grid)}")
        
        return segment_grid, grid_width, grid_height
    
    def box_counting_optimized(self, segments, min_box_size, max_box_size, box_size_factor=2.0):
        """Optimized box counting using spatial indexing."""
        total_start_time = time.time()
        
        # Find the bounding box of all segments
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        # Add a small margin
        margin = max(max_x - min_x, max_y - min_y) * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        
        box_sizes = []
        box_counts = []
        
        current_box_size = max_box_size
        
        print("Box counting debug info:")
        print("  Box size  |  Box count  |  Time (s)")
        print("------------------------------------------")
        
        # Use the same cell size as original fd-all.py
        spatial_cell_size = min_box_size * 2  # Conservative cell size
        segment_grid, grid_width, grid_height = self.create_spatial_index(
            segments, min_x, min_y, max_x, max_y, spatial_cell_size)
        
        while current_box_size >= min_box_size:
            box_start_time = time.time()
            
            num_boxes_x = int(np.ceil((max_x - min_x) / current_box_size))
            num_boxes_y = int(np.ceil((max_y - min_y) / current_box_size))
            
            occupied_boxes = set()
            
            for i in range(num_boxes_x):
                for j in range(num_boxes_y):
                    box_xmin = min_x + i * current_box_size
                    box_ymin = min_y + j * current_box_size
                    box_xmax = box_xmin + current_box_size
                    box_ymax = box_ymin + current_box_size
                    
                    min_cell_x = max(0, int((box_xmin - min_x) / spatial_cell_size))
                    max_cell_x = min(grid_width - 1, int((box_xmax - min_x) / spatial_cell_size))
                    min_cell_y = max(0, int((box_ymin - min_y) / spatial_cell_size))
                    max_cell_y = min(grid_height - 1, int((box_ymax - min_y) / spatial_cell_size))
                    
                    segments_to_check = set()
                    for cell_x in range(min_cell_x, max_cell_x + 1):
                        for cell_y in range(min_cell_y, max_cell_y + 1):
                            segments_to_check.update(segment_grid.get((cell_x, cell_y), []))
                    
                    for seg_idx in segments_to_check:
                        (x1, y1), (x2, y2) = segments[seg_idx]
                        if self.base.liang_barsky_line_box_intersection(x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                            occupied_boxes.add((i, j))
                            break
            
            count = len(occupied_boxes)
            elapsed = time.time() - box_start_time
            print(f"  {current_box_size:.6f}  |  {count:8d}  |  {elapsed:.2f}")
            
            if count > 0:
                box_sizes.append(current_box_size)
                box_counts.append(count)
            else:
                print(f"  Warning: No boxes occupied at box size {current_box_size}. Skipping this size.")
            
            current_box_size /= box_size_factor
    
        if len(box_sizes) < 2:
            raise ValueError("Not enough valid box sizes for fractal dimension calculation.")
        
        print(f"\nTotal box counting time: {time.time() - total_start_time:.2f} seconds")
        
        return np.array(box_sizes), np.array(box_counts), (min_x, min_y, max_x, max_y)
    
    def calculate_fractal_dimension(self, box_sizes, box_counts):
        """Calculate the fractal dimension using box-counting method."""
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
    
        if np.any(np.isnan(log_sizes)) or np.any(np.isnan(log_counts)):
            valid = ~(np.isnan(log_sizes) | np.isnan(log_counts))
            log_sizes = log_sizes[valid]
            log_counts = log_counts[valid]
            print(f"Warning: Removed {np.sum(~valid)} invalid ln values")
    
        if len(log_sizes) < 2:
            print("Error: Not enough valid data points for regression!")
            return float('nan'), float('nan'), float('nan')
    
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
    
        fractal_dimension = -slope
    
        print(f"R-squared value: {r_value**2:.4f}")
    
        return fractal_dimension, std_err, intercept
