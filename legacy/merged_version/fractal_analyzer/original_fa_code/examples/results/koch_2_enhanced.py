#!/usr/bin/env python3
# examples/koch_2_enhanced.py
"""
Enhanced Koch curve fractal dimension analysis example with plot saving.
This example demonstrates how to generate and analyze a Koch curve and save the plots.
"""

import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to allow imports from the fractal_analyzer package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

def main():
    """Main function to demonstrate Koch curve analysis with plot saving."""
    # Create a timestamp-based output directory within the examples directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), f"koch_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the fractal analyzer with Koch fractal type
    fractal = 'koch'
    analyzer = FractalAnalyzer(fractal)
    analysis = FractalAnalysisTools(analyzer)
    
    # Set the iteration level
    iter_level = 5
    
    # Generate fractal
    print(f"\nGenerating {fractal} fractal at iteration level {iter_level}...")
    curve, segments = analyzer.generate_fractal(fractal, iter_level)
    print(f"Generated {len(segments)} line segments")
    
    # Calculate fractal dimension
    print(f"\nCalculating fractal dimension...")
    fd, error, box_sizes, box_counts, bounding_box, intercept = analyzer.calculate_fractal_dimension(segments)
    print(f"Fractal dimension: {fd:.6f} ± {error:.6f}")
    
    # Plot using the built-in visualizer
    print(f"\nPlotting fractal curve...")
    curve_filename = analyzer.plot_results(
        segments, box_sizes, box_counts, fd, error, bounding_box,
        plot_boxes=False,  # Set to False to avoid box overlay
        level=iter_level,
        custom_filename=os.path.join(output_dir, f"{fractal}_curve_iter{iter_level}.png")
    )
    print(f"Saved fractal curve to: {curve_filename}")
    
    # Plot log-log analysis
    print(f"\nPlotting log-log analysis...")
    loglog_filename = os.path.join(output_dir, f"{fractal}_loglog_iter{iter_level}.png")
    analyzer.visualizer.plot_loglog(
        box_sizes, box_counts, fd, error, intercept,
        custom_filename=loglog_filename
    )
    print(f"Saved log-log plot to: {loglog_filename}")
    
    # Analyze linear region to find optimal window
    print(f"\nAnalyzing optimal linear region...")
    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis.analyze_linear_region(
        segments,
        fractal_type=fractal,
        plot_results=True,
        plot_boxes=False  # Set to False to avoid the box overlay error
    )
    
    # Create dimension analysis plot manually
    print(f"\nPlotting dimension analysis...")
    plt.figure(figsize=(10, 8))
    plt.errorbar(windows, dimensions, yerr=errors, fmt='o-', capsize=4, 
                color='blue', alpha=0.7, label='All window sizes')
    
    # Highlight the optimal window
    plt.axvline(x=optimal_window, color='red', linestyle='--', alpha=0.5, 
               label=f'Optimal window: {optimal_window}')
    plt.plot(optimal_window, optimal_dimension, 'ro', markersize=10)
    
    # Get theoretical dimension if available
    if hasattr(analyzer.base, 'THEORETICAL_DIMENSIONS'):
        theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal)
        if theoretical_dimension is not None:
            plt.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                     label=f'Theoretical: {theoretical_dimension:.4f}')
    
    # Add text box with dimension information
    r_squared_value = r_squared[windows.index(optimal_window)]
    textstr = f'Optimal Dimension: {optimal_dimension:.4f}\nR²: {r_squared_value:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
           verticalalignment='top', bbox=props)
    
    # Set dimension plot title and labels
    plt.title(f'Fractal Dimension vs. Window Size - {fractal.capitalize()}')
    plt.xlabel('Window Size')
    plt.ylabel('Fractal Dimension')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the dimension analysis plot
    dimension_filename = os.path.join(output_dir, f"{fractal}_dimension_analysis_iter{iter_level}.png")
    plt.savefig(dimension_filename, dpi=300)
    plt.close()
    print(f"Saved dimension analysis plot to: {dimension_filename}")
    
    # Create combined visualization
    print(f"\nCreating combined visualization...")
    create_combined_visualization(
        curve, segments, box_sizes, box_counts, windows, dimensions, 
        errors, r_squared, optimal_window, optimal_dimension, fd, error,
        fractal, iter_level, output_dir
    )
    
    print(f"\nOptimal dimension ({fractal}: {iter_level} iterations): {optimal_dimension:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return 0

def create_combined_visualization(curve, segments, box_sizes, box_counts, windows, dimensions, 
                                 errors, r_squared, optimal_window, optimal_dimension,
                                 fd, error, fractal_type, iter_level, output_dir):
    """Create a combined visualization with all analysis elements."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Calculate extent for curve plotting
    min_x = min(min(s[0][0], s[1][0]) for s in segments)
    max_x = max(max(s[0][0], s[1][0]) for s in segments)
    min_y = min(min(s[0][1], s[1][1]) for s in segments)
    max_y = max(max(s[0][1], s[1][1]) for s in segments)
    view_margin = max(max_x - min_x, max_y - min_y) * 0.05
    
    # 1. Fractal curve (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Extract and plot points
    x, y = zip(*curve)
    ax1.plot(x, y, 'k-', linewidth=0.8)
    
    # Set curve plot properties
    ax1.set_xlim(min_x - view_margin, max_x + view_margin)
    ax1.set_ylim(min_y - view_margin, max_y + view_margin)
    ax1.set_title(f'{fractal_type.capitalize()} Curve (Iteration {iter_level})')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Log-log plot (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Plot the data points
    ax2.loglog(box_sizes, box_counts, 'bo-', label='Data points', markersize=4)
    
    # Plot the linear regression line
    log_sizes = np.log(box_sizes)
    fit_counts = np.exp(intercept + (-fd) * log_sizes)
    ax2.loglog(box_sizes, fit_counts, 'r-', 
             label=f'Fit: D = {fd:.4f} ± {error:.4f}')
    
    # Set loglog plot properties
    ax2.set_title('Box Counting: ln(N) vs ln(1/r)')
    ax2.set_xlabel('Box Size (r)')
    ax2.set_ylabel('Number of Boxes (N)')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 3. Dimension analysis (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Plot all window sizes
    ax3.errorbar(windows, dimensions, yerr=errors, fmt='o-', capsize=4, 
               color='blue', alpha=0.7, label='All windows')
    
    # Highlight the optimal window
    ax3.axvline(x=optimal_window, color='red', linestyle='--', alpha=0.5, 
              label=f'Optimal: {optimal_window}')
    ax3.plot(optimal_window, optimal_dimension, 'ro', markersize=10)
    
    # Get theoretical dimension if available
    if hasattr(FractalAnalyzer(fractal_type).base, 'THEORETICAL_DIMENSIONS'):
        theoretical_dimension = FractalAnalyzer(fractal_type).base.THEORETICAL_DIMENSIONS.get(fractal_type)
        if theoretical_dimension is not None:
            ax3.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                      label=f'Theoretical: {theoretical_dimension:.4f}')
    
    # Set dimension plot properties
    ax3.set_title('Dimension vs. Window Size')
    ax3.set_xlabel('Window Size')
    ax3.set_ylabel('Fractal Dimension')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper right')
    
    # Add a secondary y-axis for R-squared values
    ax3b = ax3.twinx()
    ax3b.plot(windows, r_squared, 'g--', marker='.', alpha=0.5, label='R-squared')
    ax3b.set_ylabel('R-squared', color='g')
    ax3b.tick_params(axis='y', labelcolor='g')
    ax3b.set_ylim(0.9, 1.01)
    
    # 4. Results summary (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')  # Hide axes
    
    # Create a text summary
    summary_text = f"""
    Fractal Analysis Summary
    
    Fractal Type: {fractal_type.capitalize()}
    Iteration Level: {iter_level}
    
    Box Counting Results:
    - Fractal Dimension: {fd:.6f} ± {error:.6f}
    
    Optimal Window Analysis:
    - Optimal Window Size: {optimal_window}
    - Optimal Dimension: {optimal_dimension:.6f}
    - R-squared: {r_squared[windows.index(optimal_window)]:.6f}
    """
    
    # Get theoretical dimension if available
    if hasattr(FractalAnalyzer(fractal_type).base, 'THEORETICAL_DIMENSIONS'):
        theoretical_dimension = FractalAnalyzer(fractal_type).base.THEORETICAL_DIMENSIONS.get(fractal_type)
