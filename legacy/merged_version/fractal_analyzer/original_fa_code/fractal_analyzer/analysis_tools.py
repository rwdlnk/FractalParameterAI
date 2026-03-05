# analysis_tools.py
import numpy as np
from scipy import stats
import time
from typing import Tuple, List, Dict, Optional
import os
import matplotlib.pyplot as plt

class FractalAnalysisTools:
    """Advanced analysis tools for fractal dimensions."""
    def __init__(self, analyzer):
        """Initialize with reference to the main analyzer."""
        self.analyzer = analyzer
        # Add attributes to store figure references
        self.dimension_fig = None
        self.box_fig = None

    def trim_boundary_box_counts(self, box_sizes, box_counts, trim_count):
        """Trim specified number of box counts from each end of the data."""
        if trim_count == 0 or len(box_sizes) <= 2*trim_count:
            return box_sizes, box_counts
        return box_sizes[trim_count:-trim_count], box_counts[trim_count:-trim_count]

    def analyze_linear_region(self, segments, fractal_type=None, plot_results=True,
                             plot_boxes=True, trim_boundary=0, save_plots=False,
                             output_dir=None, custom_prefix=None):
        """
        Analyze how the choice of linear region affects the calculated dimension.
        Uses a sliding window approach to identify the optimal scaling region.
        Parameters:
        -----------
        segments : list
            List of segments to analyze
        fractal_type : str, optional
            Type of fractal
        plot_results : bool, default=True
            Whether to display dimension analysis plot
        plot_boxes : bool, default=True
            Whether to display box counting visualization
        trim_boundary : int, default=0
            Number of box counts to trim from each end
        save_plots : bool, default=False
            Whether to save the generated plots
        output_dir : str, optional
            Directory to save plots (if save_plots is True)
        custom_prefix : str, optional
            Custom prefix for plot filenames
        Returns:
        --------
        tuple
            (windows, dimensions, errors, r_squared, optimal_window, optimal_dimension)
        """
        print("\n==== ANALYZING LINEAR REGION SELECTION ====\n")
        # Create output directory if saving plots
        if save_plots:
            if output_dir is None:
                output_dir = "fractal_analysis_output"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Plots will be saved to: {output_dir}")
        
        # Use provided type or instance type
        type_used = fractal_type or self.analyzer.fractal_type
        if type_used in self.analyzer.base.THEORETICAL_DIMENSIONS:
            theoretical_dimension = self.analyzer.base.THEORETICAL_DIMENSIONS[type_used]
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
        box_sizes, box_counts, bounding_box = self.analyzer.box_counter.box_counting_optimized(
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
        min_window = 3 # Minimum points for regression
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
        
        # Plot box counting if requested
        if plot_boxes:
            # Create and store the box counting figure
            self.box_fig = plt.figure(figsize=(10, 8))
            ax = self.box_fig.add_subplot(111)
            
            # Plot the curve with boxes
            self.analyzer.visualizer.plot_box_overlay(
                segments, box_sizes[-1], min_x, min_y, max_x, max_y, bounding_box)
            
            # Set box plot title and labels
            ax.set_title(f'Box Counting Visualization - Box Size: {box_sizes[-1]:.6f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save the box counting plot if requested
            if save_plots:
                prefix = custom_prefix or type_used or "fractal"
                box_filename = os.path.join(output_dir, f"{prefix}_box_counting.png")
                self.box_fig.savefig(box_filename, dpi=300)
                print(f"Saved box counting plot to: {box_filename}")
            
            # Show the box figure if interactive display is enabled
            if not save_plots:
                plt.show()
            else:
                plt.close(self.box_fig)
        
        # Plot dimension analysis if requested
        if plot_results:
            # Create and store the dimension analysis figure
            self.dimension_fig = plt.figure(figsize=(10, 8))
            ax = self.dimension_fig.add_subplot(111)
            
            # Plot all window sizes with error bars
            ax.errorbar(windows, dimensions, yerr=errors, fmt='o-', capsize=4,
                       color='blue', alpha=0.7, label='All window sizes')
            
            # Highlight the optimal window
            ax.axvline(x=optimal_window, color='red', linestyle='--', alpha=0.5,
                      label=f'Optimal window: {optimal_window}')
            ax.plot(optimal_window, optimal_dimension, 'ro', markersize=10)
            
            # Plot theoretical dimension if available
            if theoretical_dimension is not None:
                ax.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                          label=f'Theoretical: {theoretical_dimension:.4f}')
            
            # Add text box with dimension information
            textstr = f'Optimal Dimension: {optimal_dimension:.4f}\nR²: {r_squared[closest_idx]:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
            
            # Add a secondary y-axis for R-squared values
            ax2 = ax.twinx()
            ax2.plot(windows, r_squared, 'g--', marker='.', alpha=0.5, label='R-squared')
            ax2.set_ylabel('R-squared', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.legend(loc='lower right')
            
            # Set dimension plot title and labels
            ax.set_title(f'Fractal Dimension vs. Window Size - {type_used.capitalize() if type_used else "Unknown"}')
            ax.set_xlabel('Window Size')
            ax.set_ylabel('Fractal Dimension')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save the dimension analysis plot if requested
            if save_plots:
                prefix = custom_prefix or type_used or "fractal"
                dim_filename = os.path.join(output_dir, f"{prefix}_dimension_analysis.png")
                self.dimension_fig.savefig(dim_filename, dpi=300)
                print(f"Saved dimension analysis plot to: {dim_filename}")
            
            # Show the dimension figure if interactive display is enabled
            if not save_plots:
                plt.show()
            else:
                plt.close(self.dimension_fig)
        
        # Create a loglog plot of box counting data
        if save_plots:
            # Calculate fractal dimension using all data points
            fd, error, intercept = self.analyzer.box_counter.calculate_fractal_dimension(
                box_sizes, box_counts)
            
            # Create loglog plot using the visualizer
            prefix = custom_prefix or type_used or "fractal"
            loglog_filename = os.path.join(output_dir, f"{prefix}_loglog_plot.png")
            self.analyzer.visualizer.plot_loglog(
                box_sizes, box_counts, fd, error, intercept,
                custom_filename=loglog_filename)
            print(f"Saved loglog plot to: {loglog_filename}")
            
            # Create and save the fractal curve without boxes
            curve_filename = os.path.join(output_dir, f"{prefix}_curve.png")
            self.analyzer.visualizer.plot_fractal_curve(
                segments, bounding_box, plot_boxes=False,
                custom_filename=curve_filename)
            print(f"Saved fractal curve to: {curve_filename}")
            
            # Create combined plot with all visualizations
            self.create_combined_plot(
                segments, box_sizes, box_counts, windows, dimensions, errors, r_squared,
                optimal_window, optimal_dimension, theoretical_dimension, bounding_box,
                output_dir, prefix)
        
        return windows, dimensions, errors, r_squared, optimal_window, optimal_dimension

    def analyze_iterations(self, min_level=1, max_level=8, fractal_type=None,
                         box_ratio=0.3, no_plots=False, no_box_plot=False,
                         use_linear_region=False, trim_boundary=0,
                         save_plots=False, output_dir=None):
        """
        Analyze how fractal dimension varies with iteration depth.
        Generates curves at different levels and calculates their dimensions.
        
        Parameters:
        -----------
        min_level : int, default=1
            Minimum iteration level to analyze
        max_level : int, default=8
            Maximum iteration level to analyze
        fractal_type : str, optional
            Type of fractal
        box_ratio : float, default=0.3
            Ratio of smallest to largest box size
        no_plots : bool, default=False
            Whether to disable plot generation
        no_box_plot : bool, default=False
            Whether to disable box counting visualization
        use_linear_region : bool, default=False
            Whether to use analyze_linear_region for dimension calculation
        trim_boundary : int, default=0
            Number of box counts to trim when using linear region analysis
        save_plots : bool, default=False
            Whether to save plots
        output_dir : str, optional
            Directory to save plots (if save_plots is True)
        
        Returns:
        --------
        tuple
            If use_linear_region=False: (levels, dimensions, errors, r_squared)
            If use_linear_region=True: (levels, dimensions, errors, r_squared, optimal_windows, all_windows_data)
        """
        print("\n==== ANALYZING DIMENSION VS ITERATION LEVEL ====\n")
        
        # Create output directory if saving plots
        if save_plots:
            if output_dir is None:
                output_dir = "fractal_analysis_output"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Plots will be saved to: {output_dir}")
        
        # Use provided type or instance type
        type_used = fractal_type or self.analyzer.fractal_type
        if type_used is None:
            raise ValueError("Fractal type must be specified either in constructor or as argument")
        
        theoretical_dimension = self.analyzer.base.THEORETICAL_DIMENSIONS.get(type_used)
        if theoretical_dimension:
            print(f"Theoretical {type_used} dimension: {theoretical_dimension:.6f}")
        
        # Initialize results storage
        levels = list(range(min_level, max_level + 1))
        dimensions = []
        errors = []
        r_squared = []
        
        # For linear region analysis (if enabled)
        optimal_windows = []
        all_windows_data = []
        
        # For each level, generate a curve and calculate its dimension
        for level in levels:
            print(f"\n--- Processing {type_used} curve at level {level} ---")
            
            # Generate the curve
            _, segments = self.analyzer.generate_fractal(type_used, level)
            
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
            
            if use_linear_region:
                # Use linear region analysis (more sophisticated)
                windows, dims, errs, rsq, optimal_window, optimal_dimension = self.analyze_linear_region(
                    segments,
                    fractal_type=type_used,
                    plot_results=not no_plots and not save_plots,
                    plot_boxes=not no_box_plot and not no_plots and not save_plots,
                    trim_boundary=1,
                    save_plots=save_plots,
                    output_dir=output_dir,
                    custom_prefix=f"{type_used}_level{level}"
                )
                
                # Store the results
                dimensions.append(optimal_dimension)
                errors.append(errs[windows.index(optimal_window)])
                r_squared.append(rsq[windows.index(optimal_window)])
                optimal_windows.append(optimal_window)
                all_windows_data.append((windows, dims, errs, rsq))
                
                print(f"Level {level} - Fractal Dimension: {optimal_dimension:.6f} ± {errs[windows.index(optimal_window)]:.6f}")
                print(f"Optimal window: {optimal_window}")
                
            else:
                # Use direct box counting (original method)
                box_sizes, box_counts, bounding_box = self.analyzer.box_counter.box_counting_optimized(
                    segments, min_box_size, max_box_size, box_size_factor=box_size_factor)
                
                # Calculate dimension
                fractal_dimension, error, intercept = self.analyzer.box_counter.calculate_fractal_dimension(
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
                
                # Plot results if requested and not using linear region
                if not no_plots and not save_plots:
                    # Create separate filenames for curve and dimension plots
                    curve_file = f"{type_used}_level_{level}_curve.png"
                    dimension_file = f"{type_used}_level_{level}_dimension.png"
                    
                    # Respect the no_box_plot parameter
                    plot_boxes = (level <= 6) and not no_box_plot
                    
                    # Plot the fractal curve
                    self.analyzer.visualizer.plot_fractal_curve(
                        segments, bounding_box, plot_boxes, box_sizes, box_counts,
                        custom_filename=curve_file, level=level)
                    
                    # Plot the dimension analysis (log-log plot)
                    if hasattr(self.analyzer.visualizer, 'plot_loglog'):
                        self.analyzer.visualizer.plot_loglog(
                            box_sizes, box_counts, fractal_dimension, error,
                            custom_filename=dimension_file)
            
            if theoretical_dimension:
                print(f"Difference from theoretical: {abs(dimensions[-1] - theoretical_dimension):.6f}")
            
            print(f"R-squared: {r_squared[-1]:.6f}")
        
        # Plot the dimension vs. level results
        if not no_plots and not save_plots and hasattr(self.analyzer.visualizer, 'plot_dimension_vs_level'):
            self.analyzer.visualizer.plot_dimension_vs_level(
                levels, dimensions, errors, r_squared, theoretical_dimension, type_used)
        
        if use_linear_region:
            return levels, dimensions, errors, r_squared, optimal_windows, all_windows_data
        else:
            return levels, dimensions, errors, r_squared

    def create_combined_plot(self, segments, box_sizes, box_counts, windows, dimensions,
                          errors, r_squared, optimal_window, optimal_dimension,
                          theoretical_dimension, bounding_box, output_dir, prefix):
        """Create a combined plot with all visualizations."""
        # Create a combined figure
        combined_fig = plt.figure(figsize=(15, 10))
        
        # 1. Fractal curve (top left)
        ax1 = combined_fig.add_subplot(2, 2, 1)
        
        # Extract and plot points
        x_points = []
        y_points = []
        for (x1, y1), (x2, y2) in segments[:10000]: # Limit for visualization
            x_points.extend([x1, x2, None])
            y_points.extend([y1, y2, None])
        
        if x_points:
            x_points = x_points[:-1]
            y_points = y_points[:-1]
        
        ax1.plot(x_points, y_points, 'k-', linewidth=0.8)
        
        # Set curve plot properties
        min_x, min_y, max_x, max_y = bounding_box
        view_margin = max(max_x - min_x, max_y - min_y) * 0.05
        ax1.set_xlim(min_x - view_margin, max_x + view_margin)
        ax1.set_ylim(min_y - view_margin, max_y + view_margin)
        ax1.set_title('Fractal Curve')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Box counting visualization (top right)
        ax2 = combined_fig.add_subplot(2, 2, 2)
        
        # Redraw box overlay on this axis
        box_size = box_sizes[-1] # Use smallest box size
        num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
        
        # Get bounding box and plot properties
        ax2.set_xlim(min_x - view_margin, max_x + view_margin)
        ax2.set_ylim(min_y - view_margin, max_y + view_margin)
        ax2.set_title(f'Box Counting - Size: {box_size:.6f}')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Plot the curve
        if x_points:
            ax2.plot(x_points, y_points, 'k-', linewidth=0.8)
        
        # Add sample boxes for visualization (just a few for the combined plot)
        from matplotlib.patches import Rectangle
        rectangles = []
        
        # Simplified approach: just draw some boxes around the curve
        num_sample_boxes = min(500, num_boxes_x * num_boxes_y) # Limit for visibility
        for i in range(0, num_boxes_x, max(1, num_boxes_x // 20)):
            for j in range(0, num_boxes_y, max(1, num_boxes_y // 20)):
                box_xmin = min_x + i * box_size
                box_ymin = min_y + j * box_size
                rectangles.append(Rectangle((box_xmin, box_ymin), box_size, box_size,
                                         facecolor='none', edgecolor='r', linewidth=0.5))
        
        from matplotlib.collections import PatchCollection
        pc = PatchCollection(rectangles, facecolor='none', edgecolor='r', linewidth=0.5, alpha=0.8)
        ax2.add_collection(pc)
        
        # 3. Log-log plot (bottom left)
        ax3 = combined_fig.add_subplot(2, 2, 3)
        
        # Calculate fractal dimension for the plot
        fd, error, intercept = self.analyzer.box_counter.calculate_fractal_dimension(box_sizes, box_counts)
        
        # Plot the data points
        ax3.loglog(box_sizes, box_counts, 'bo-', label='Data points', markersize=4)
        
        # Plot the linear regression line
        log_sizes = np.log(box_sizes)
        fit_counts = np.exp(intercept + (-fd) * log_sizes)
        ax3.loglog(box_sizes, fit_counts, 'r-',
                 label=f'Fit: D = {fd:.4f} ± {error:.4f}')
        
        # Set loglog plot properties
        ax3.set_title('Box Counting: ln(N) vs ln(1/r)')
        ax3.set_xlabel('Box Size (r)')
        ax3.set_ylabel('Number of Boxes (N)')
        ax3.legend()
        ax3.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # 4. Dimension analysis (bottom right)
        ax4 = combined_fig.add_subplot(2, 2, 4)
        
        # Plot all window sizes
        ax4.errorbar(windows, dimensions, yerr=errors, fmt='o-', capsize=4,
                   color='blue', alpha=0.7, label='All windows')
        
        # Highlight the optimal window
        ax4.axvline(x=optimal_window, color='red', linestyle='--', alpha=0.5,
                  label=f'Optimal: {optimal_window}')
        ax4.plot(optimal_window, optimal_dimension, 'ro', markersize=10)
        
        # Plot theoretical dimension if available
        if theoretical_dimension is not None:
            ax4.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                      label=f'Theoretical: {theoretical_dimension:.4f}')
        
        # Add text box with dimension information
        textstr = f'Dimension: {optimal_dimension:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        # Set dimension plot properties
        ax4.set_title('Dimension vs. Window Size')
        ax4.set_xlabel('Window Size')
        ax4.set_ylabel('Fractal Dimension')
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.legend(loc='upper right')
        
        # Add a secondary y-axis for R-squared values
        ax4b = ax4.twinx()
        ax4b.plot(windows, r_squared, 'g--', marker='.', alpha=0.5, label='R-squared')
        ax4b.set_ylabel('R-squared', color='g')
        ax4b.tick_params(axis='y', labelcolor='g')
        
        # Set overall title and adjust layout
        plt.suptitle(f'{prefix.capitalize()} Fractal Analysis - Dimension: {optimal_dimension:.6f}',
                   fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        # Save the combined plot
        combined_filename = os.path.join(output_dir, f"{prefix}_combined_analysis.png")
        combined_fig.savefig(combined_filename, dpi=300)
        plt.close(combined_fig)
        print(f"Saved combined analysis plot to: {combined_filename}")
        
        return combined_filename
