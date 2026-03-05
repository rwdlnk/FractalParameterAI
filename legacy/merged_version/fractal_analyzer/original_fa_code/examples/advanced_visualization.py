#!/usr/bin/env python3
# examples/advanced_visualization.py
"""
Advanced fractal visualization example.
This script provides a command-line interface for generating and analyzing
various types of fractals with customizable parameters and plot saving options.
"""

import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to allow imports from the fractal_analyzer package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools
from fractal_analyzer.core import FractalBase

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Fractal Visualization")
    parser.add_argument("-t", "--type", default="koch", 
                        choices=["koch", "sierpinski", "minkowski", "hilbert", "dragon"],
                        help="Type of fractal to analyze")
    parser.add_argument("-i", "--iterations", type=int, default=5,
                        help="Number of iterations for fractal generation")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory for saved plots")
    parser.add_argument("--no-boxes", action="store_true",
                        help="Disable box counting visualization")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display plots (only save)")
    return parser.parse_args()

def main():
    """Main function for advanced fractal visualization."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory based on arguments or timestamp
    if args.output:
        # If a relative path is provided, make it relative to the examples directory
        if not os.path.isabs(args.output):
            output_dir = os.path.join(os.path.dirname(__file__), args.output)
        else:
            output_dir = args.output
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 
                                f"{args.type}_analysis_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the fractal analyzer with appropriate type
    analyzer = FractalAnalyzer(args.type)
    analysis = FractalAnalysisTools(analyzer)
    
    # Ensure the visualizer has a reference to the base object
    if not hasattr(analyzer.visualizer, 'base') or analyzer.visualizer.base is None:
        analyzer.visualizer.base = analyzer.base
    
    print(f"\n==== ADVANCED VISUALIZATION: {args.type.upper()} FRACTAL ====")
    print(f"Iteration level: {args.iterations}")
    print(f"Output directory: {output_dir}")
    
    # Generate fractal
    print(f"\nGenerating {args.type} fractal at iteration level {args.iterations}...")
    curve, segments = analyzer.generate_fractal(args.type, args.iterations)
    print(f"Generated {len(segments)} line segments")
    
    # Calculate fractal dimension
    print(f"\nCalculating fractal dimension...")
    fd, error, box_sizes, box_counts, bounding_box, intercept = analyzer.calculate_fractal_dimension(segments)
    print(f"Fractal dimension: {fd:.6f} ± {error:.6f}")
    
    # Plot the fractal curve
    print(f"\nPlotting fractal curve...")
    curve_filename = analyzer.plot_results(
        segments, box_sizes, box_counts, fd, error, bounding_box,
        plot_boxes=not args.no_boxes,
        level=args.iterations,
        custom_filename=os.path.join(output_dir, f"{args.type}_curve_iter{args.iterations}.png")
    )
    print(f"Saved fractal curve to: {curve_filename}")
    
    # Plot log-log analysis
    print(f"\nPlotting log-log analysis...")
    loglog_filename = os.path.join(output_dir, f"{args.type}_loglog_iter{args.iterations}.png")
    analyzer.visualizer.plot_loglog(
        box_sizes, box_counts, fd, error, intercept,
        custom_filename=loglog_filename
    )
    print(f"Saved log-log plot to: {loglog_filename}")
    
    # Analyze linear region to find optimal window
    print(f"\nAnalyzing optimal linear region...")
    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis.analyze_linear_region(
        segments,
        fractal_type=args.type,
        plot_results=not args.no_show,
        plot_boxes=not args.no_boxes and not args.no_show
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
    theoretical_dimension = FractalBase.THEORETICAL_DIMENSIONS.get(args.type)
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
    plt.title(f'Fractal Dimension vs. Window Size - {args.type.capitalize()}')
    plt.xlabel('Window Size')
    plt.ylabel('Fractal Dimension')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the dimension analysis plot
    dimension_filename = os.path.join(output_dir, f"{args.type}_dimension_analysis_iter{args.iterations}.png")
    plt.savefig(dimension_filename, dpi=300)
    plt.close()
    print(f"Saved dimension analysis plot to: {dimension_filename}")
    
    # Create combined visualization
    print(f"\nCreating combined visualization...")
    create_combined_visualization(
        curve, segments, box_sizes, box_counts, windows, dimensions, 
        errors, r_squared, optimal_window, optimal_dimension, fd, error,
        args.type, args.iterations, output_dir, intercept
    )
    
    # Generate a summary report
    generate_report(
        args.type, args.iterations, segments, optimal_dimension, 
        r_squared[windows.index(optimal_window)], fd, error, output_dir
    )
    
    print(f"\nOptimal dimension ({args.type}: {args.iterations} iterations): {optimal_dimension:.6f}")
    print(f"All visualizations and analysis saved to: {output_dir}")
    
    return 0

def create_combined_visualization(curve, segments, box_sizes, box_counts, windows, dimensions, 
                                 errors, r_squared, optimal_window, optimal_dimension,
                                 fd, error, fractal_type, iter_level, output_dir, intercept):
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
    theoretical_dimension = FractalBase.THEORETICAL_DIMENSIONS.get(fractal_type)
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
    theoretical_dimension = FractalBase.THEORETICAL_DIMENSIONS.get(fractal_type)
    if theoretical_dimension is not None:
        summary_text += f"""
    Theoretical Dimension: {theoretical_dimension:.6f}
    Difference: {abs(optimal_dimension - theoretical_dimension):.6f}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
           verticalalignment='top', family='monospace')
    
    # Set overall title and adjust layout
    plt.suptitle(f'{fractal_type.capitalize()} Fractal Analysis', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the combined visualization
    combined_filename = os.path.join(output_dir, f"{fractal_type}_combined_analysis_iter{iter_level}.png")
    plt.savefig(combined_filename, dpi=300)
    plt.close()
    
    print(f"Saved combined visualization to: {combined_filename}")
    return combined_filename

def generate_report(fractal_type, iterations, segments, dimension, r_squared, 
                   fd, error, output_dir):
    """Generate a summary report of the fractal analysis."""
    report_file = os.path.join(output_dir, f"{fractal_type}_iter{iterations}_report.txt")
    
    # Get theoretical dimension if available
    theoretical_dimension = FractalBase.THEORETICAL_DIMENSIONS.get(fractal_type)
    
    with open(report_file, 'w') as f:
        f.write(f"FRACTAL ANALYSIS REPORT\n")
        f.write(f"======================\n\n")
        f.write(f"Fractal Type: {fractal_type.capitalize()}\n")
        f.write(f"Iteration Level: {iterations}\n")
        f.write(f"Segment Count: {len(segments)}\n\n")
        
        f.write(f"BOX COUNTING ANALYSIS\n")
        f.write(f"-------------------\n")
        f.write(f"Fractal Dimension: {fd:.6f} ± {error:.6f}\n\n")
        
        f.write(f"OPTIMAL WINDOW ANALYSIS\n")
        f.write(f"---------------------\n")
        f.write(f"Optimal Dimension: {dimension:.6f}\n")
        f.write(f"R-squared Value: {r_squared:.6f}\n")
        
        if theoretical_dimension is not None:
            f.write(f"\nTHEORETICAL COMPARISON\n")
            f.write(f"---------------------\n")
            f.write(f"Theoretical Dimension: {theoretical_dimension:.6f}\n")
            f.write(f"Difference from Box Counting: {abs(fd - theoretical_dimension):.6f}\n")
            f.write(f"Difference from Optimal Window: {abs(dimension - theoretical_dimension):.6f}\n")
        
        f.write(f"\nGENERATED FILES\n")
        f.write(f"--------------\n")
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                f.write(f"- {filename}\n")
    
    print(f"Generated summary report: {report_file}")

if __name__ == "__main__":
    main()
