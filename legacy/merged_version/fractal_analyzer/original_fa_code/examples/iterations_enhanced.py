#!/usr/bin/env python3
# examples/iterations_enhanced.py
"""
Enhanced fractal dimension analysis across different iteration levels.
This script provides a command-line interface for analyzing how fractal dimension
converges with increasing iteration levels, with comprehensive visualizations.
"""
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Fractal Iteration Analysis")
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
    parser.add_argument("--no-show", action="store_true",
                      help="Do not display plots (only save)")
    return parser.parse_args()

def main():
    """Main function for enhanced fractal iteration analysis."""
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
                                f"{args.type}_iter_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the analyzer and analysis tools
    analyzer = FractalAnalyzer(args.type)
    analysis = FractalAnalysisTools(analyzer)
    
    print(f"\n==== ENHANCED ITERATION ANALYSIS: {args.type.upper()} FRACTAL ====")
    print(f"Iteration levels: {args.min_level} to {args.max_level}")
    print(f"Analysis method: {'Linear Region' if args.use_linear_region else 'Standard Box Counting'}")
    print(f"Output directory: {output_dir}")
    
    # Run the appropriate iteration analysis
    if args.use_linear_region:
        print("\nUsing linear region analysis for each level...")
        levels, dimensions, errors, r_squared, optimal_windows, all_windows_data = analysis.analyze_iterations(
            min_level=args.min_level,
            max_level=args.max_level,
            fractal_type=args.type,
            no_plots=True,  # We'll create our own plots
            use_linear_region=True
        )
    else:
        print("\nUsing standard box counting method...")
        levels, dimensions, errors, r_squared = analysis.analyze_iterations(
            min_level=args.min_level,
            max_level=args.max_level,
            fractal_type=args.type,
            no_plots=True  # We'll create our own plots
        )
    
    # Collect segments for each level
    all_segments = []
    all_curves = []
    all_box_sizes = []
    all_box_counts = []
    
    print("\nPerforming detailed analysis for each level...")
    for level in levels:
        print(f"  Processing level {level}...")
        curve, segments = analyzer.generate_fractal(args.type, level)
        all_curves.append(curve)
        all_segments.append(segments)
        
        # Calculate box counts for visualization
        fd, error, box_sizes, box_counts, bounding_box, intercept = analyzer.calculate_fractal_dimension(segments)
        all_box_sizes.append(box_sizes)
        all_box_counts.append(box_counts)
    
    # Create comprehensive visualization
    print("\nCreating comprehensive visualization...")
    create_comprehensive_visualization(
        args.type, levels, dimensions, errors, r_squared,
        all_curves, all_segments, all_box_sizes, all_box_counts,
        analyzer, output_dir, args.use_linear_region,
        optimal_windows if args.use_linear_region else None
    )
    
    # Generate a detailed report
    generate_iteration_report(
        args.type, levels, dimensions, errors, r_squared,
        analyzer, all_segments, output_dir, args.use_linear_region,
        optimal_windows if args.use_linear_region else None
    )
    
    # Plot convergence graph
    plot_convergence_graph(
        args.type, levels, dimensions, errors, analyzer, output_dir
    )
    
    print(f"\nFinal dimension ({args.type} at level {args.max_level}): {dimensions[-1]:.6f} ± {errors[-1]:.6f}")
    print(f"All visualizations and analysis saved to: {output_dir}")
    
    return 0

def create_comprehensive_visualization(fractal_type, levels, dimensions, errors, 
                                     r_squared, all_curves, all_segments, 
                                     all_box_sizes, all_box_counts, 
                                     analyzer, output_dir, use_linear_region=False,
                                     optimal_windows=None):
    """Create a comprehensive visualization of iteration analysis."""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Dimension convergence (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.errorbar(levels, dimensions, yerr=errors, fmt='o-', capsize=4,
               color='blue', alpha=0.7, label='Calculated dimension')
    
    # Add theoretical dimension if available
    theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal_type)
    if theoretical_dimension is not None:
        ax1.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                  label=f'Theoretical: {theoretical_dimension:.4f}')
    
    # Add window size annotations if using linear region
    if use_linear_region and optimal_windows:
        for i, level in enumerate(levels):
            ax1.annotate(f'w={optimal_windows[i]}', 
                       xy=(level, dimensions[i]), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=8)
    
    # Set dimension plot properties
    ax1.set_title('Dimension Convergence with Iteration')
    ax1.set_xlabel('Iteration Level')
    ax1.set_ylabel('Fractal Dimension')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Adjust axis limits
    ax1.set_xlim(min(levels) - 0.5, max(levels) + 0.5)
    y_min = min(dimensions) - max(errors) - 0.05
    y_max = max(dimensions) + max(errors) + 0.05
    if theoretical_dimension is not None:
        y_min = min(y_min, theoretical_dimension - 0.05)
        y_max = max(y_max, theoretical_dimension + 0.05)
    ax1.set_ylim(y_min, y_max)
    
    # Add a secondary y-axis for R-squared values
    ax1b = ax1.twinx()
    ax1b.plot(levels, r_squared, 'g--', marker='.', alpha=0.5, label='R-squared')
    ax1b.set_ylabel('R-squared', color='g')
    ax1b.tick_params(axis='y', labelcolor='g')
    ax1b.set_ylim(0.9, 1.01)
    
    # 2. Multi-level curve comparison (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Plot 3 selected levels for comparison (or fewer if not enough levels)
    if len(levels) >= 3:
        selected_levels = [levels[0], levels[len(levels)//2], levels[-1]]
    else:
        selected_levels = levels
    
    colors = ['blue', 'green', 'red']
    
    for i, level in enumerate(selected_levels):
        level_idx = levels.index(level)
        curve = all_curves[level_idx]
        x, y = zip(*curve)
        ax2.plot(x, y, color=colors[i % len(colors)], linewidth=0.8+i*0.4,
               label=f'Level {level} (D={dimensions[level_idx]:.4f})')
    
    # Calculate limits for curve plotting
    all_x = [point[0] for curve in all_curves for point in curve]
    all_y = [point[1] for curve in all_curves for point in curve]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    
    # Set curve plot properties
    ax2.set_xlim(x_min - margin, x_max + margin)
    ax2.set_ylim(y_min - margin, y_max + margin)
    ax2.set_title(f'{fractal_type.capitalize()} Curve at Different Iteration Levels')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Final level loglog plot (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Plot the loglog data for the final level
    final_box_sizes = all_box_sizes[-1]
    final_box_counts = all_box_counts[-1]
    
    # Calculate dimension for the final level
    fd, error, intercept = analyzer.box_counter.calculate_fractal_dimension(
        final_box_sizes, final_box_counts)
    
    # Plot data points
    ax3.loglog(final_box_sizes, final_box_counts, 'bo-', 
             label='Data points', markersize=4)
    
    # Plot linear regression line
    log_sizes = np.log(final_box_sizes)
    fit_counts = np.exp(intercept + (-fd) * log_sizes)
    ax3.loglog(final_box_sizes, fit_counts, 'r-',
             label=f'Fit: D = {fd:.4f} ± {error:.4f}')
    
    # Set loglog plot properties and adjust axis limits
    ax3.set_title(f'Box Counting (Level {levels[-1]})')
    ax3.set_xlabel('Box Size (r)')
    ax3.set_ylabel('Number of Boxes (N)')
    ax3.legend()
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    ax3.set_xlim(min(final_box_sizes) * 0.9, max(final_box_sizes) * 1.1)
    ax3.set_ylim(min(final_box_counts) * 0.9, max(final_box_counts) * 1.1)
    
    # 4. Dimension error analysis (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Calculate relative error compared to theoretical dimension
    if theoretical_dimension is not None:
        rel_errors = [abs(dim - theoretical_dimension) / theoretical_dimension * 100 
                    for dim in dimensions]
        ax4.semilogy(levels, rel_errors, 'ro-', markersize=6, label='Relative error')
        
        # Plot power law fit if enough points
        if len(levels) >= 3:
            # Fit a power law: error ~ level^(-alpha)
            valid_indices = [i for i, e in enumerate(rel_errors) if e > 0]
            if len(valid_indices) >= 3:
                valid_levels = [levels[i] for i in valid_indices]
                valid_errors = [rel_errors[i] for i in valid_indices]
                log_levels = np.log(valid_levels)
                log_errors = np.log(valid_errors)
                
                # Use numpy polyfit directly
                poly_results = np.polyfit(log_levels, log_errors, 1, full=False)
                slope, intercept = poly_results
                r_value = np.corrcoef(log_levels, log_errors)[0, 1]
                
                # Plot the fit line
                fit_levels = np.linspace(min(levels), max(levels), 100)
                fit_errors = np.exp(intercept) * fit_levels ** slope
                ax4.semilogy(fit_levels, fit_errors, 'b--', 
                           label=f'Fit: error ~ level^({slope:.2f})')
        
        ax4.set_title('Error Convergence')
        ax4.set_xlabel('Iteration Level')
        ax4.set_ylabel('Relative Error (%)')
        ax4.grid(True, which='both', linestyle='--', alpha=0.5)
        ax4.legend()
        
        # Adjust axis limits
        ax4.set_xlim(min(levels) - 0.5, max(levels) + 0.5)
        if min(rel_errors) > 0:
            ax4.set_ylim(min(rel_errors) * 0.5, max(rel_errors) * 2)
    else:
        # If no theoretical dimension, show general convergence
        # Calculate differences between consecutive levels
        if len(levels) >= 2:
            level_diffs = []
            for i in range(1, len(levels)):
                level_diffs.append(abs(dimensions[i] - dimensions[i-1]))
            
            # Plot the differences
            ax4.semilogy(levels[1:], level_diffs, 'ro-', markersize=6,
                       label='Change between levels')
            ax4.set_title('Convergence Between Levels')
            ax4.set_xlabel('Iteration Level')
            ax4.set_ylabel('Dimension Change')
            ax4.grid(True, which='both', linestyle='--', alpha=0.5)
            ax4.legend()
            
            # Adjust axis limits
            ax4.set_xlim(min(levels[1:]) - 0.5, max(levels[1:]) + 0.5)
            if min(level_diffs) > 0:
                ax4.set_ylim(min(level_diffs) * 0.5, max(level_diffs) * 2)
        else:
            # Not enough levels for comparison, show error bars
            ax4.errorbar(levels, dimensions, yerr=errors, fmt='ro-', 
                       capsize=4, label='Dimension with error')
            ax4.set_title('Dimension Error Analysis')
            ax4.set_xlabel('Iteration Level')
            ax4.set_ylabel('Fractal Dimension')
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()
    
    # Set overall title and adjust layout
    plt.suptitle(f'{fractal_type.capitalize()} Fractal Iteration Analysis', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save the combined visualization
    combined_filename = os.path.join(output_dir, f"{fractal_type}_comprehensive_analysis.png")
    plt.savefig(combined_filename, dpi=300)
    plt.close(fig)
    print(f"Saved comprehensive visualization to: {combined_filename}")
    
    return combined_filename

def plot_convergence_graph(fractal_type, levels, dimensions, errors, analyzer, output_dir):
    """Create a detailed convergence graph with fit curve."""
    plt.figure(figsize=(10, 6))
    
    # Plot the dimension vs. iteration level
    plt.errorbar(levels, dimensions, yerr=errors, fmt='o-', capsize=4,
               color='blue', alpha=0.7, label='Calculated dimension')
    
    # Get theoretical dimension if available
    theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal_type)
    if theoretical_dimension is not None:
        plt.axhline(y=theoretical_dimension, color='green', linestyle=':', alpha=0.7,
                  label=f'Theoretical: {theoretical_dimension:.4f}')
        
        # Try to fit a convergence curve
        # Model: D(n) = D_inf - C/n^alpha
        if len(levels) >= 4:
            # Use the difference from theoretical
            deltas = [theoretical_dimension - d for d in dimensions]
            
            # Take log of both sides to linearize
            log_levels = np.log(levels)
            log_deltas = [np.log(abs(d)) for d in deltas if abs(d) > 0]
            valid_levels = [levels[i] for i, d in enumerate(deltas) if abs(d) > 0]
            
            if len(valid_levels) >= 3:
                try:
                    # Fit a straight line to the log-log data
                    log_valid_levels = np.log(valid_levels)
                    
                    # Use numpy polyfit directly
                    poly_results = np.polyfit(log_valid_levels, log_deltas, 1, full=False)
                    slope, intercept = poly_results
                    
                    # Convert back to original form
                    alpha = -slope
                    C = np.exp(intercept)
                    
                    # Create a smooth curve for the fit
                    fine_levels = np.linspace(min(levels), max(levels) * 1.5, 100)
                    fit_dimensions = [theoretical_dimension - C * (n ** (-alpha)) for n in fine_levels]
                    
                    # Plot the fit
                    plt.plot(fine_levels, fit_dimensions, 'r--', 
                           label=f'Fit: D(n) = D∞ - C/n^{alpha:.2f}')
                    
                    # Add extrapolation annotation
                    plt.axvspan(max(levels), max(levels) * 1.5, alpha=0.1, color='gray')
                    plt.text(max(levels) * 1.2, theoretical_dimension - 0.05, 
                           'Extrapolation', rotation=0, ha='center', alpha=0.7)
                except Exception as e:
                    print(f"Could not fit convergence curve: {e}")
    
    # Set plot title and labels
    plt.title(f'Fractal Dimension Convergence - {fractal_type.capitalize()}')
    plt.xlabel('Iteration Level')
    plt.ylabel('Fractal Dimension')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust axis limits
    plt.xlim(min(levels) - 0.5, max(levels) * 1.2 if theoretical_dimension is not None else max(levels) + 0.5)
    y_min = min(dimensions) - max(errors) - 0.05
    y_max = max(dimensions) + max(errors) + 0.05
    if theoretical_dimension is not None:
        y_min = min(y_min, theoretical_dimension - 0.05)
        y_max = max(y_max, theoretical_dimension + 0.05)
    plt.ylim(y_min, y_max)
    
    # Add arrows pointing to next potential iterations if theoretical dimension exists
    if theoretical_dimension is not None and max(levels) < 10:
        next_level = max(levels) + 1
        if len(dimensions) >= 2:
            # Simple linear extrapolation for next iteration
            next_dim = dimensions[-1] + (dimensions[-1] - dimensions[-2])
            plt.annotate('', xy=(next_level, next_dim), xytext=(max(levels), dimensions[-1]),
                       arrowprops=dict(arrowstyle='->', color='purple'))
    
    # Save the plot
    convergence_filename = os.path.join(output_dir, f"{fractal_type}_dimension_convergence.png")
    plt.savefig(convergence_filename, dpi=300)
    plt.close()
    print(f"Saved convergence plot to: {convergence_filename}")
    
    return convergence_filename

def generate_iteration_report(fractal_type, levels, dimensions, errors, r_squared,
                            analyzer, all_segments, output_dir, use_linear_region=False,
                            optimal_windows=None):
    """Generate a detailed report of the iteration analysis."""
    report_file = os.path.join(output_dir, f"{fractal_type}_iteration_analysis_report.txt")
    theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal_type)
    
    with open(report_file, 'w') as f:
        f.write(f"FRACTAL ITERATION ANALYSIS REPORT\n")
        f.write(f"===============================\n\n")
        f.write(f"Fractal Type: {fractal_type.capitalize()}\n")
        f.write(f"Iteration Levels: {min(levels)} to {max(levels)}\n")
        f.write(f"Analysis Method: {'Linear Region' if use_linear_region else 'Standard Box Counting'}\n\n")
        
        f.write(f"THEORETICAL REFERENCE\n")
        f.write(f"--------------------\n")
        if theoretical_dimension is not None:
            f.write(f"Theoretical Dimension: {theoretical_dimension:.6f}\n\n")
        else:
            f.write(f"No theoretical dimension available for this fractal type.\n\n")
        
        f.write(f"DETAILED RESULTS BY LEVEL\n")
        f.write(f"------------------------\n")
        
        if use_linear_region and optimal_windows:
            f.write(f"{'Level':<8} {'Segments':<10} {'Window':<8} {'Dimension':<12} {'Error':<12} {'R-squared':<12}")
        else:
            f.write(f"{'Level':<8} {'Segments':<10} {'Dimension':<12} {'Error':<12} {'R-squared':<12}")
        
        if theoretical_dimension is not None:
            f.write(f" {'Diff from Theo.':<16} {'Rel. Error (%)':<16}\n")
        else:
            f.write(f"\n")
        
        for i, level in enumerate(levels):
            segments = all_segments[i]
            if use_linear_region and optimal_windows:
                f.write(f"{level:<8} {len(segments):<10} {optimal_windows[i]:<8} {dimensions[i]:12.6f} {errors[i]:12.6f} {r_squared[i]:12.6f}")
            else:
                f.write(f"{level:<8} {len(segments):<10} {dimensions[i]:12.6f} {errors[i]:12.6f} {r_squared[i]:12.6f}")
            
            if theoretical_dimension is not None:
                diff = abs(dimensions[i] - theoretical_dimension)
                rel_err = diff / theoretical_dimension * 100
                f.write(f" {diff:16.6f} {rel_err:16.4f}\n")
            else:
                f.write(f"\n")
        
        f.write(f"\nCONVERGENCE ANALYSIS\n")
        f.write(f"-------------------\n")
        f.write(f"Final dimension (level {max(levels)}): {dimensions[-1]:.6f} ± {errors[-1]:.6f}\n")
        
        if theoretical_dimension is not None:
            final_diff = abs(dimensions[-1] - theoretical_dimension)
            final_rel_err = final_diff / theoretical_dimension * 100
            f.write(f"Difference from theoretical: {final_diff:.6f}\n")
            f.write(f"Relative error: {final_rel_err:.4f}%\n")
            f.write(f"Convergence: {100 - final_rel_err:.2f}%\n\n")
            
            if len(levels) >= 4:
                # Calculate convergence rate for report
                try:
                    # Use the difference from theoretical
                    deltas = [theoretical_dimension - d for d in dimensions]
                    
                    # Take log of both sides to linearize
                    log_levels = np.log(levels)
                    log_deltas = [np.log(abs(d)) for d in deltas if abs(d) > 0]
                    valid_levels = [levels[i] for i, d in enumerate(deltas) if abs(d) > 0]
                    
                    if len(valid_levels) >= 3:
                        log_valid_levels = np.log(valid_levels)
                        
                        # Use numpy polyfit directly
                        poly_results = np.polyfit(log_valid_levels, log_deltas, 1, full=False)
                        slope, intercept = poly_results
                        r_value = np.corrcoef(log_valid_levels, log_deltas)[0, 1]
                        
                        f.write(f"CONVERGENCE RATE ANALYSIS\n")
                        f.write(f"------------------------\n")
                        f.write(f"Fitted convergence model: error ~ level^({slope:.4f})\n")
                        f.write(f"Convergence coefficient: {np.exp(intercept):.6f}\n")
                        f.write(f"R-squared of convergence fit: {r_value**2:.6f}\n\n")
                        
                        f.write(f"EXTRAPOLATION PREDICTIONS\n")
                        f.write(f"------------------------\n")
                        f.write(f"Predictions for higher iteration levels:\n")
                        f.write(f"{'Level':<8} {'Predicted Dimension':<20} {'Predicted Error (%)':<20}\n")
                        
                        alpha = -slope
                        C = np.exp(intercept)
                        
                        for level in range(max(levels) + 1, max(levels) + 6):
                            pred_diff = C * (level ** slope)
                            pred_dim = theoretical_dimension - pred_diff  # Assuming underestimation
                            pred_error = (pred_diff / theoretical_dimension) * 100
                            f.write(f"{level:<8} {pred_dim:20.6f} {pred_error:20.4f}\n")
                except Exception as e:
                    f.write(f"Could not analyze convergence rate: {str(e)}\n")
        
        f.write(f"\nGENERATED FILES\n")
        f.write(f"--------------\n")
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                f.write(f"- {filename}\n")
    
    print(f"Generated detailed analysis report: {report_file}")
    return report_file

if __name__ == "__main__":
    main()
