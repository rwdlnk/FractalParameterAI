# analysis_tools.py
import numpy as np
from scipy import stats
import time
from typing import Tuple, List, Dict, Optional

class FractalAnalysisTools:
    """Advanced analysis tools for fractal dimensions."""
    
    def __init__(self, fractal_analyzer):
        """Initialize with reference to the main analyzer."""
        self.fractal_analyzer = fractal_analyzer
    
    def trim_boundary_box_counts(self, box_sizes, box_counts, trim_count):
        """Trim specified number of box counts from each end of the data."""
        if trim_count == 0 or len(box_sizes) <= 2*trim_count:
            return box_sizes, box_counts
        
        return box_sizes[trim_count:-trim_count], box_counts[trim_count:-trim_count]
    
    def analyze_linear_region(self, segments, fractal_type=None, plot_results=True, 
                             plot_boxes=True, trim_boundary=0):
        """
        Analyze how the choice of linear region affects the calculated dimension.
        Uses a sliding window approach to identify the optimal scaling region.
        """
        print("\n==== ANALYZING LINEAR REGION SELECTION ====\n")
        
        # Use provided type or instance type
        type_used = fractal_type or self.fractal_analyzer.fractal_type
        
        if type_used in self.fractal_analyzer.base.THEORETICAL_DIMENSIONS:
            theoretical_dimension = self.fractal_analyzer.base.THEORETICAL_DIMENSIONS[type_used]
            print(f"Theoretical {type_used} dimension: {theoretical_dimension:.6f}")
        else:
            theoretical_dimension = None
            print("No theoretical dimension available for comparison")
        
        # Calculate extent to determine box sizes
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        extent = max(max_x - min_x, max_y - min_y)
        
        # Use same box sizes as original fd-all.py
        min_box_size = 0.001
        max_box_size = extent / 2
        box_size_factor = 1.5
        
        print(f"Using box size range: {min_box_size:.8f} to {max_box_size:.8f}")
        print(f"Box size reduction factor: {box_size_factor}")
        
        # Calculate fractal dimension with many data points
        box_sizes, box_counts, bounding_box = self.fractal_analyzer.box_counter.box_counting_optimized(
            segments, min_box_size, max_box_size, box_size_factor=box_size_factor)
        
        # Trim boundary box counts if requested
        if trim_boundary > 0:
            print(f"Trimming {trim_boundary} box counts from each end")
            box_sizes, box_counts = self.trim_boundary_box_counts(box_sizes, box_counts, trim_boundary)
            print(f"Box counts after trimming: {len(box_counts)}")
        
        # Convert to ln scale for analysis
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        
        # Analyze different window sizes for linear region selection
        min_window = 3  # Minimum points for regression
        max_window = len(log_sizes)
        
        windows = range(min_window, max_window + 1)
        dimensions = []
        errors = []
        r_squared = []
        start_indices = []
        end_indices = []
        
        print("Window size | Start idx | End idx | Dimension | Error | R²")
        print("-" * 65)
        
        # Try all possible window sizes
        for window_size in windows:
            best_r2 = -1
            best_dimension = None
            best_error = None
            best_start = None
            best_end = None
            
            # Try all possible starting points for this window size
            for start_idx in range(len(log_sizes) - window_size + 1):
                end_idx = start_idx + window_size
                
                # Perform regression on this window
                window_log_sizes = log_sizes[start_idx:end_idx]
                window_log_counts = log_counts[start_idx:end_idx]
                
                slope, _, r_value, _, std_err = stats.linregress(window_log_sizes, window_log_counts)
                dimension = -slope
            
                # Store if this is the best fit for this window size
                if r_value**2 > best_r2:
                    best_r2 = r_value**2
                    best_dimension = dimension
                    best_error = std_err
                    best_start = start_idx
                    best_end = end_idx
            
            # Store the best results for this window size
            dimensions.append(best_dimension)
            errors.append(best_error)
            r_squared.append(best_r2)
            start_indices.append(best_start)
            end_indices.append(best_end)
            
            print(f"{window_size:11d} | {best_start:9d} | {best_end:7d} | {best_dimension:9.6f} | {best_error:5.6f} | {best_r2:.6f}")
        
        # Find the window with dimension closest to theoretical or best R²
        if theoretical_dimension is not None:
            closest_idx = np.argmin(np.abs(np.array(dimensions) - theoretical_dimension))
        else:
            closest_idx = np.argmax(r_squared)
        
        optimal_window = windows[closest_idx]
        optimal_dimension = dimensions[closest_idx]
        optimal_start = start_indices[closest_idx]
        optimal_end = end_indices[closest_idx]
        
        print("\nResults:")
        if theoretical_dimension is not None:
            print(f"Theoretical dimension: {theoretical_dimension:.6f}")
            print(f"Closest dimension: {optimal_dimension:.6f} (window size: {optimal_window})")
        else:
            print(f"Best dimension (highest R²): {optimal_dimension:.6f} (window size: {optimal_window})")
        print(f"Optimal scaling region: points {optimal_start} to {optimal_end}")
        print(f"Box size range: {box_sizes[optimal_start]:.8f} to {box_sizes[optimal_end-1]:.8f}")
        
        return windows, dimensions, errors, r_squared, optimal_window, optimal_dimension

    # analysis_tools.py method with proper indentation
    def analyze_iterations(self, min_level=1, max_level=8, fractal_type=None, 
                          box_ratio=0.3, no_plots=False, no_box_plot=False):
        """
        Analyze how fractal dimension varies with iteration depth.
        Generates curves at different levels and calculates their dimensions.
        """
        print("\n==== ANALYZING DIMENSION VS ITERATION LEVEL ====\n")
            
        # Use provided type or instance type
        type_used = fractal_type or self.fractal_analyzer.fractal_type
        
        if type_used is None:
            raise ValueError("Fractal type must be specified either in constructor or as argument")
        
        theoretical_dimension = self.fractal_analyzer.base.THEORETICAL_DIMENSIONS.get(type_used)
        if theoretical_dimension:
            print(f"Theoretical {type_used} dimension: {theoretical_dimension:.6f}")
        
        # Initialize results storage
        levels = list(range(min_level, max_level + 1))
        dimensions = []
        errors = []
        r_squared = []
        
        # For each level, generate a curve and calculate its dimension
        for level in levels:
            print(f"\n--- Processing {type_used} curve at level {level} ---")
            
            # Generate the curve
            _, segments = self.fractal_analyzer.generate_fractal(type_used, level)
            
            # Calculate extent to determine box sizes
            min_x = min(min(s[0][0], s[1][0]) for s in segments)
            max_x = max(max(s[0][0], s[1][0]) for s in segments)
            min_y = min(min(s[0][1], s[1][1]) for s in segments)
            max_y = max(max(s[0][1], s[1][1]) for s in segments)
            extent = max(max_x - min_x, max_y - min_y)
            
            # Use appropriate box sizes
            min_box_size = 0.001
            max_box_size = extent / 2
            box_size_factor = 1.5
            
            # Perform box counting
            box_sizes, box_counts, bounding_box = self.fractal_analyzer.box_counter.box_counting_optimized(
                segments, min_box_size, max_box_size, box_size_factor=box_size_factor)
            
            # Calculate dimension
            fractal_dimension, error, intercept = self.fractal_analyzer.box_counter.calculate_fractal_dimension(
                box_sizes, box_counts)
            
            # Calculate R-squared value
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)
            _, _, r_value, _, _ = stats.linregress(log_sizes, log_counts)
            r_squared_value = r_value**2
            
            # Store results
            dimensions.append(fractal_dimension)
            errors.append(error)
            r_squared.append(r_squared_value)
            
            print(f"Level {level} - Fractal Dimension: {fractal_dimension:.6f} ± {error:.6f}")
            if theoretical_dimension:
                print(f"Difference from theoretical: {abs(fractal_dimension - theoretical_dimension):.6f}")
            print(f"R-squared: {r_squared_value:.6f}")
            
            # Plot results if requested
            if not no_plots:
                # Create separate filenames for curve and dimension plots
                curve_file = f"{type_used}_level_{level}_curve.png"
                dimension_file = f"{type_used}_level_{level}_dimension.png"
                
                print(f"Plotting fractal curve to {curve_file}")
                
                # Respect the no_box_plot parameter
                plot_boxes = (level <= 6) and not no_box_plot
                
                # Plot the fractal curve
                self.fractal_analyzer.visualizer.plot_fractal_curve(
                    segments, bounding_box, plot_boxes, box_sizes, box_counts, 
                    custom_filename=curve_file, level=level)
                
                # Plot the dimension analysis (log-log plot)
                # This would need to be implemented in the visualizer
                if hasattr(self.fractal_analyzer.visualizer, 'plot_loglog'):
                    self.fractal_analyzer.visualizer.plot_loglog(
                        box_sizes, box_counts, fractal_dimension, error, 
                        custom_filename=dimension_file)
        
        # Plot the dimension vs. level results if not disabled
        if not no_plots and hasattr(self.fractal_analyzer.visualizer, 'plot_dimension_vs_level'):
            self.fractal_analyzer.visualizer.plot_dimension_vs_level(
                levels, dimensions, errors, r_squared, theoretical_dimension, type_used)
        
        return levels, dimensions, errors, r_squared
