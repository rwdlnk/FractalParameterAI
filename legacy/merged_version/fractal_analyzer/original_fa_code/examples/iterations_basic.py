#!/usr/bin/env python3
# examples/iterations_basic.py
"""
Basic example demonstrating fractal dimension analysis across different iteration levels.
This example shows how the calculated fractal dimension converges as iteration level increases.
"""
import os
import sys
import time
import matplotlib.pyplot as plt
import argparse

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic Fractal Iteration Analysis")
    parser.add_argument("-t", "--type", default="koch",
                      choices=["koch", "sierpinski", "minkowski", "hilbert", "dragon"],
                      help="Type of fractal to analyze")
    parser.add_argument("--min-level", type=int, default=1,
                      help="Minimum iteration level to analyze")
    parser.add_argument("--max-level", type=int, default=6,
                      help="Maximum iteration level to analyze")
    parser.add_argument("--use-linear-region", action="store_true",
                      help="Use linear region analysis for each level")
    parser.add_argument("-o", "--output", default=None,
                      help="Output directory for saved plots")
    return parser.parse_args()

def main():
    """Main function to demonstrate iteration analysis of Koch curve."""
    # Parse command line arguments
    args = parse_args()
    
    # Create a timestamp-based output directory
    if args.output:
        if not os.path.isabs(args.output):
            output_dir = os.path.join(os.path.dirname(__file__), args.output)
        else:
            output_dir = args.output
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 
                                f"iteration_analysis_{args.type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the analyzer with fractal type
    fractal = args.type
    analyzer = FractalAnalyzer(fractal)
    analysis = FractalAnalysisTools(analyzer)
    
    print(f"\nAnalyzing {fractal} fractal from level {args.min_level} to {args.max_level}...")
    
    # Run the iteration analysis
    if args.use_linear_region:
        print("Using linear region analysis method...")
        levels, dimensions, errors, r_squared, optimal_windows, _ = analysis.analyze_iterations(
            min_level=args.min_level,
            max_level=args.max_level,
            fractal_type=fractal,
            no_plots=True,  # We'll create our own plots
            use_linear_region=True
        )
    else:
        print("Using standard box counting method...")
        levels, dimensions, errors, r_squared = analysis.analyze_iterations(
            min_level=args.min_level,
            max_level=args.max_level,
            fractal_type=fractal,
            no_plots=True  # We'll create our own plots
        )
    
    # Create a plot of dimension vs. iteration level
    plt.figure(figsize=(10, 6))
    plt.errorbar(levels, dimensions, yerr=errors, fmt='o-', capsize=4, 
                 color='blue', alpha=0.7, label='Calculated dimension')
    
    # Add theoretical dimension if available
    if hasattr(analyzer.base, 'THEORETICAL_DIMENSIONS'):
        theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal)
        if theoretical_dimension is not None:
            plt.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                      label=f'Theoretical: {theoretical_dimension:.4f}')
    
    # Add text box with convergence information
    final_dimension = dimensions[-1]
    final_error = errors[-1]
    textstr = f'Final Dimension: {final_dimension:.6f}\nError: {final_error:.6f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Set plot title and labels
    plt.title(f'Fractal Dimension vs. Iteration Level - {fractal.capitalize()}')
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
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f"{fractal}_dimension_vs_iteration.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved dimension vs. iteration plot to: {plot_filename}")
    
    # Create and save a summary report
    report_filename = os.path.join(output_dir, f"{fractal}_iteration_analysis_report.txt")
    with open(report_filename, 'w') as f:
        f.write(f"FRACTAL ITERATION ANALYSIS REPORT\n")
        f.write(f"===============================\n\n")
        f.write(f"Fractal Type: {fractal.capitalize()}\n")
        f.write(f"Iteration Levels: {args.min_level} to {args.max_level}\n")
        f.write(f"Analysis Method: {'Linear Region' if args.use_linear_region else 'Standard Box Counting'}\n\n")
        
        f.write(f"DETAILED RESULTS\n")
        f.write(f"--------------\n")
        if args.use_linear_region:
            f.write(f"{'Level':<8} {'Window':<8} {'Dimension':<12} {'Error':<12} {'R-squared':<12}\n")
            for i, level in enumerate(levels):
                f.write(f"{level:<8} {optimal_windows[i]:<8} {dimensions[i]:12.6f} {errors[i]:12.6f} {r_squared[i]:12.6f}\n")
        else:
            f.write(f"{'Level':<8} {'Dimension':<12} {'Error':<12} {'R-squared':<12}\n")
            for i, level in enumerate(levels):
                f.write(f"{level:<8} {dimensions[i]:12.6f} {errors[i]:12.6f} {r_squared[i]:12.6f}\n")
        
        f.write(f"\nCONVERGENCE ANALYSIS\n")
        f.write(f"-------------------\n")
        f.write(f"Final dimension (level {args.max_level}): {dimensions[-1]:.6f} ± {errors[-1]:.6f}\n")
        
        if hasattr(analyzer.base, 'THEORETICAL_DIMENSIONS'):
            theoretical_dim = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal)
            if theoretical_dim is not None:
                f.write(f"Theoretical dimension: {theoretical_dim:.6f}\n")
                diff = abs(dimensions[-1] - theoretical_dim)
                f.write(f"Difference: {diff:.6f}\n")
                convergence_rate = abs(dimensions[-1] - theoretical_dim) / theoretical_dim * 100
                f.write(f"Convergence: {100 - convergence_rate:.2f}%\n")
    
    print(f"Saved analysis report to: {report_filename}")
    print(f"\nFinal dimension ({fractal} at level {args.max_level}): {dimensions[-1]:.6f} ± {errors[-1]:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    main()
