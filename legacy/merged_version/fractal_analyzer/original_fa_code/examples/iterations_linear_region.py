#!/usr/bin/env python3
# examples/iterations_linear_region.py
"""
Example demonstrating fractal dimension analysis across different iteration levels
using the linear region analysis method for enhanced accuracy.
"""
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fractal Iteration Analysis with Linear Region")
    parser.add_argument("-t", "--type", default="koch",
                      choices=["koch", "sierpinski", "minkowski", "hilbert", "dragon"],
                      help="Type of fractal to analyze")
    parser.add_argument("--min-level", type=int, default=1,
                      help="Minimum iteration level to analyze")
    parser.add_argument("--max-level", type=int, default=5, 
                      help="Maximum iteration level to analyze")
    parser.add_argument("-o", "--output", default=None,
                      help="Output directory for saved plots")
    return parser.parse_args()

def main():
    """Main function for fractal iteration analysis using linear region method."""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory based on arguments or timestamp
    if args.output:
        if not os.path.isabs(args.output):
            output_dir = os.path.join(os.path.dirname(__file__), args.output)
        else:
            output_dir = args.output
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__),
                                f"{args.type}_linreg_iter_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the analyzer with appropriate type
    analyzer = FractalAnalyzer(args.type)
    analysis = FractalAnalysisTools(analyzer)
    
    print(f"\n==== LINEAR REGION ITERATION ANALYSIS: {args.type.upper()} FRACTAL ====")
    print(f"Analyzing levels: {args.min_level} to {args.max_level}")
    print(f"Output directory: {output_dir}")
    
    # Run the iteration analysis with linear region enabled
    levels, dimensions, errors, r_squared, optimal_windows, all_windows_data = analysis.analyze_iterations(
        min_level=args.min_level,
        max_level=args.max_level,
        fractal_type=args.type,
        no_plots=True,  # We'll create our own plots
        use_linear_region=True
    )
    
    # Create visualization of optimal window selection across iterations
    plt.figure(figsize=(12, 8))
    
    # Primary plot: Dimension vs Level with error bars
    plt.errorbar(levels, dimensions, yerr=errors, fmt='o-', capsize=4,
               color='blue', alpha=0.7, label='Optimal dimension')
    
    # Get theoretical dimension if available
    theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(args.type)
    if theoretical_dimension is not None:
        plt.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                  label=f'Theoretical: {theoretical_dimension:.4f}')
    
    # Add window size annotations
    for i, level in enumerate(levels):
        plt.annotate(f'w={optimal_windows[i]}', 
                   xy=(level, dimensions[i]), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8)
    
    # Set plot properties
    plt.title(f'Dimension vs. Iteration Level with Linear Region Analysis - {args.type.capitalize()}')
    plt.xlabel('Iteration Level')
    plt.ylabel('Fractal Dimension')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust axis limits to ensure all data fits within plot
    plt.xlim(min(levels) - 0.5, max(levels) + 0.5)
    y_min = min(dimensions) - max(errors) - 0.05
    y_max = max(dimensions) + max(errors) + 0.05
    if theoretical_dimension is not None:
        y_min = min(y_min, theoretical_dimension - 0.05)
        y_max = max(y_max, theoretical_dimension + 0.05)
    plt.ylim(y_min, y_max)
    
    # Save the main plot
    main_plot_filename = os.path.join(output_dir, f"{args.type}_linreg_dimensions.png")
    plt.savefig(main_plot_filename, dpi=300)
    plt.close()
    print(f"Saved main plot to: {main_plot_filename}")
    
    # Create window analysis heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    window_matrix = []
    for level_idx, (windows, dims, _, _) in enumerate(all_windows_data):
        level_data = []
        for window in range(min(w for w_data in all_windows_data for w in w_data[0]), 
                          max(w for w_data in all_windows_data for w in w_data[0]) + 1):
            if window in windows:
                window_idx = windows.index(window)
                level_data.append(dims[window_idx])
            else:
                level_data.append(np.nan)
        window_matrix.append(level_data)
    
    # Create the heatmap
    windows_range = range(min(w for w_data in all_windows_data for w in w_data[0]), 
                        max(w for w_data in all_windows_data for w in w_data[0]) + 1)
    heatmap = ax.imshow(window_matrix, cmap='viridis', aspect='auto', 
                       extent=[min(windows_range)-0.5, max(windows_range)+0.5, 
                              max(levels)+0.5, min(levels)-0.5])
    
    # Mark optimal windows
    for i, level in enumerate(levels):
        ax.plot(optimal_windows[i], level, 'ro', markersize=8)
    
    # Add colorbar and labels
    cbar = fig.colorbar(heatmap)
    cbar.set_label('Fractal Dimension')
    ax.set_title(f'Window Size Analysis Across Iteration Levels - {args.type.capitalize()}')
    ax.set_xlabel('Window Size')
    ax.set_ylabel('Iteration Level')
    
    # Add theoretical dimension reference
    if theoretical_dimension is not None:
        textstr = f'Theoretical dimension: {theoretical_dimension:.6f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
              verticalalignment='bottom', bbox=props)
    
    # Save the heatmap
    heatmap_filename = os.path.join(output_dir, f"{args.type}_window_heatmap.png")
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()
    print(f"Saved window analysis heatmap to: {heatmap_filename}")
    
    # Generate a detailed report
    report_filename = os.path.join(output_dir, f"{args.type}_linreg_iteration_report.txt")
    with open(report_filename, 'w') as f:
        f.write(f"LINEAR REGION ITERATION ANALYSIS REPORT\n")
        f.write(f"====================================\n\n")
        f.write(f"Fractal Type: {args.type.capitalize()}\n")
        f.write(f"Iteration Levels: {args.min_level} to {args.max_level}\n\n")
        
        if theoretical_dimension is not None:
            f.write(f"Theoretical Dimension: {theoretical_dimension:.6f}\n\n")
        
        f.write(f"OPTIMAL WINDOW RESULTS\n")
        f.write(f"---------------------\n")
        f.write(f"{'Level':<8} {'Window':<8} {'Dimension':<12} {'Error':<12} {'R-squared':<12}")
        if theoretical_dimension is not None:
            f.write(f" {'Diff from Theo.':<16}\n")
        else:
            f.write(f"\n")
        
        for i, level in enumerate(levels):
            f.write(f"{level:<8} {optimal_windows[i]:<8} {dimensions[i]:12.6f} {errors[i]:12.6f} {r_squared[i]:12.6f}")
            if theoretical_dimension is not None:
                diff = abs(dimensions[i] - theoretical_dimension)
                f.write(f" {diff:16.6f}\n")
            else:
                f.write(f"\n")
        
        f.write(f"\nDETAILED WINDOW ANALYSIS\n")
        f.write(f"------------------------\n")
        
        for i, level in enumerate(levels):
            f.write(f"\nLevel {level}:\n")
            f.write(f"{'Window':<8} {'Dimension':<12} {'Error':<12} {'R-squared':<12}\n")
            
            windows, dims, errs, rsq = all_windows_data[i]
            for j, window in enumerate(windows):
                marker = " *" if window == optimal_windows[i] else ""
                f.write(f"{window:<8} {dims[j]:12.6f} {errs[j]:12.6f} {rsq[j]:12.6f}{marker}\n")
    
    print(f"Saved detailed report to: {report_filename}")
    print(f"\nLinear region iteration analysis complete.")
    return 0

if __name__ == "__main__":
    main()
