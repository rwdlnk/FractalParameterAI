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
        # Import and initialize the cache
        from fractal_analyzer.box_counting_cache import BoxCountingCache
        self.cache = BoxCountingCache()

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

    def box_counting_optimized(self, segments, min_box_size=0.001, max_box_size=None, box_size_factor=2.0):
        """Optimized box counting using spatial indexing."""
        # Check cache first
        cached_result = self.cache.get(segments, min_box_size, max_box_size, box_size_factor)
        if cached_result is not None:
            return cached_result
    
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
    
        # Ensure max_box_size is not None
        if max_box_size is None:
            extent = max(max_x - min_x, max_y - min_y)
            max_box_size = extent / 2
            print(f"Auto-determined max box size: {max_box_size}")
    
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
    
        # Cache results before returning
        result = (np.array(box_sizes), np.array(box_counts), (min_x, min_y, max_x, max_y))
        return self.cache.store(segments, min_box_size, max_box_size, box_size_factor, result)

    # Complete updated BoxCounter.calculate_fractal_dimension method with optimal scaling

    def calculate_fractal_dimension(self, box_sizes, box_counts, use_optimal_scaling=True, 
                                  window_width=5, sigma_thresh=0.05, min_region_length=5, debug=False):
        """
        Calculate the fractal dimension using box-counting method with optimal scaling region selection.
    
        Args:
            box_sizes (array-like): Array of box sizes
            box_counts (array-like): Array of box counts
            use_optimal_scaling (bool): Whether to use optimal scaling region selection
            window_width (int): Width of the window for local slope analysis
            sigma_thresh (float): Threshold for standard deviation to identify stable regions
            min_region_length (int): Minimum length of candidate scaling regions
            debug (bool): Whether to print debug information
        
        Returns:
            tuple: (fractal_dimension, std_error, intercept, r_squared, optimal_region, local_slopes)
        """
        # Convert to numpy arrays
        box_sizes = np.array(box_sizes)
        box_counts = np.array(box_counts)
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
    
        if np.any(np.isnan(log_sizes)) or np.any(np.isnan(log_counts)):
            valid = ~(np.isnan(log_sizes) | np.isnan(log_counts))
            log_sizes = log_sizes[valid]
            log_counts = log_counts[valid]
            print(f"Warning: Removed {np.sum(~valid)} invalid ln values")
    
        if len(log_sizes) < 2:
            print("Error: Not enough valid data points for regression!")
            return float('nan'), float('nan'), float('nan'), float('nan'), None, None
    
        # Use optimal scaling region selection if requested
        if use_optimal_scaling:
            # Initialize the scaling selector if not already available
            if not hasattr(self, 'scaling_selector'):
                from .optimal_scaling import OptimalScalingRegionSelector
                self.scaling_selector = OptimalScalingRegionSelector(
                    window_width=window_width, sigma_thresh=sigma_thresh, min_region_length=min_region_length
                )
        
            # Select optimal scaling region
            optimal_region, dimension, std_error, r_squared, local_slopes = (
                self.scaling_selector.select_optimal_region(box_sizes, box_counts, debug=debug)
            )
            
            # If optimal region found, calculate intercept
            if optimal_region:
                start, end = optimal_region
                log_sizes_opt = np.log(1/box_sizes[start:end+1])
                log_counts_opt = np.log(box_counts[start:end+1])
                _, intercept, _, _, _ = stats.linregress(log_sizes_opt, log_counts_opt)
            
                print(f"Optimal scaling region: indices {start} to {end}")
                print(f"Box size range: {box_sizes[start]:.8f} to {box_sizes[end]:.8f}")
            else:
                # If no optimal region found, use the entire range
                log_sizes_inv = np.log(1/box_sizes)
                slope, intercept, r_value, p_value, std_error = stats.linregress(
                    log_sizes_inv, log_counts)
                dimension = slope
                r_squared = r_value ** 2
            
                # Calculate local slopes manually for consistency
                local_slopes = np.zeros(len(log_sizes_inv) - 1)
                for i in range(len(log_sizes_inv) - 1):
                    local_slopes[i] = ((log_counts[i+1] - log_counts[i]) / 
                                      (log_sizes_inv[i+1] - log_sizes_inv[i]))
            
                print("No optimal scaling region found, using entire range")
        else:
            # Traditional approach: use the entire range
            log_sizes_inv = np.log(1/box_sizes)
            slope, intercept, r_value, p_value, std_error = stats.linregress(
                log_sizes_inv, log_counts)
            dimension = slope
            r_squared = r_value ** 2
            optimal_region = None
        
            # Calculate local slopes manually for consistency
            local_slopes = np.zeros(len(log_sizes_inv) - 1)
            for i in range(len(log_sizes_inv) - 1):
                local_slopes[i] = ((log_counts[i+1] - log_counts[i]) / 
                                  (log_sizes_inv[i+1] - log_sizes_inv[i]))
    
        print(f"R-squared value: {r_squared:.4f}")
    
        return dimension, std_error, intercept, r_squared, optimal_region, local_slopes

    # Helper method to calculate local slopes - add this to BoxCounter class
    def _calculate_local_slopes(self, log_box_sizes, log_box_counts):
        """Calculate local slopes between consecutive points in the log-log plot."""
        local_slopes = np.zeros(len(log_box_sizes) - 1)
        for i in range(len(log_box_sizes) - 1):
            local_slopes[i] = ((log_box_counts[i+1] - log_box_counts[i]) / 
                              (log_box_sizes[i+1] - log_box_sizes[i]))
        return local_slopes

