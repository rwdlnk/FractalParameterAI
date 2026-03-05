"""
Command-line interface for FastFractalAnalyzer v3.

Provides comprehensive CLI options matching the original FractalAnalyzer functionality.
"""

import argparse
import sys
import os
import csv
from datetime import datetime
import numpy as np
from typing import Optional
from scipy import stats

from .core.fractal_analyzer import FastFractalAnalyzer
from .core.data_types import SegmentArray, FractalResult
from .io.vtk_reader import VTKReader
from .plotting.fractal_plots import FractalPlotter, PlotConfig


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="FastFractalAnalyzer v3 - High-performance fractal dimension analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze segments from file
  fast-fractal --file data.txt --analyze_linear_region

  # Generate and analyze Koch curve
  fast-fractal --generate koch --level 5 --analyze_linear_region

  # Analyze across multiple iteration levels
  fast-fractal --generate koch --analyze_iterations --min_level 3 --max_level 6

  # Baseline analysis with no optimizations
  fast-fractal --generate koch --level 5 --base_no_opt

  # Generate publication-quality plots
  fast-fractal --generate sierpinski --level 4 --eps_plots --no_titles

  # Analyze dragon curve
  fast-fractal --generate dragon --level 6 --analyze_linear_region

  # Compare multiple fractals
  fast-fractal --generate hilbert --level 4 --analyze_iterations
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Load line segments from file (x1 y1 x2 y2 format)'
    )
    input_group.add_argument(
        '--vtk',
        type=str,
        help='Load interface from VTK file (RECTILINEAR_GRID format)'
    )
    input_group.add_argument(
        '--generate', '-g',
        type=str,
        choices=['koch', 'sierpinski', 'dragon', 'minkowski', 'hilbert'],
        help='Generate mathematical fractal (Koch, Sierpinski, Dragon, Minkowski, or Hilbert curve)'
    )

    # Generation parameters
    parser.add_argument(
        '--level', '-l',
        type=int,
        default=5,
        help='Iteration level for fractal generation (default: 5)'
    )

    # Analysis options
    analysis_group = parser.add_mutually_exclusive_group()
    analysis_group.add_argument(
        '--analyze_linear_region',
        action='store_true',
        help='Perform optimized linear scaling region analysis (default)'
    )
    analysis_group.add_argument(
        '--analyze_iterations',
        action='store_true',
        help='Analyze fractal across multiple iteration levels'
    )
    analysis_group.add_argument(
        '--base_no_opt',
        action='store_true',
        help='Baseline analysis: no optimizations, simple regression on all data'
    )

    # Iteration analysis parameters
    parser.add_argument(
        '--min_level',
        type=int,
        default=3,
        help='Minimum iteration level for --analyze_iterations (default: 3)'
    )
    parser.add_argument(
        '--max_level',
        type=int,
        default=6,
        help='Maximum iteration level for --analyze_iterations (default: 6)'
    )

    # Box counting parameters
    parser.add_argument(
        '--min_box_size',
        type=float,
        help='Minimum box size (auto-estimated if not specified)'
    )
    parser.add_argument(
        '--max_box_size',
        type=float,
        help='Maximum box size (auto-estimated if not specified)'
    )
    parser.add_argument(
        '--size_factor',
        type=float,
        default=1.5,
        help='Box size reduction factor (default: 1.5)'
    )

    # Quality parameters (for optimized analysis only)
    parser.add_argument(
        '--min_r_squared',
        type=float,
        default=0.99,
        help='Minimum R² threshold for valid results (default: 0.99)'
    )
    parser.add_argument(
        '--min_scaling_decades',
        type=float,
        default=1.5,
        help='Minimum log10 scaling range (default: 1.5)'
    )

    # Output and plotting options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--no_titles',
        action='store_true',
        help='Disable plot titles for publication'
    )
    parser.add_argument(
        '--eps_plots',
        action='store_true',
        help='Save plots in EPS format for publication'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plot generation'
    )

    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except results'
    )

    return parser


def setup_analyzer(args) -> FastFractalAnalyzer:
    """Create and configure the analyzer based on command-line arguments."""

    if args.base_no_opt:
        # Baseline: all optimizations disabled
        grid_optimization = False
        boundary_artifact_removal = False
    else:
        # Normal optimized analysis
        grid_optimization = True
        boundary_artifact_removal = True

    analyzer = FastFractalAnalyzer(
        min_r_squared=args.min_r_squared,
        min_scaling_decades=args.min_scaling_decades,
        grid_optimization=grid_optimization,
        boundary_artifact_removal=boundary_artifact_removal
    )

    return analyzer


def load_segments(args) -> SegmentArray:
    """Load segments based on input arguments."""
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)

        if not args.quiet:
            print(f"Loading segments from {args.file}...")

        segments = SegmentArray.from_file(args.file)

        if not args.quiet:
            print(f"Loaded {segments.n_segments} segments")
            print(f"Bounding box: {segments.bbox}")

        return segments

    elif args.vtk:
        if not os.path.exists(args.vtk):
            print(f"Error: VTK file '{args.vtk}' not found.")
            sys.exit(1)

        if not args.quiet:
            print(f"Loading interface from VTK file {args.vtk}...")

        reader = VTKReader()
        result = reader.analyze_vtk_file(args.vtk, interface_value=0.5)
        segments = result['interface_segments']

        if not args.quiet:
            print(f"Grid: {result['nx']} x {result['ny']}")
            print(f"Extracted {segments.n_segments} interface segments")
            if segments.n_segments > 0:
                print(f"Interface bounding box: {segments.bbox}")
                print(f"Total interface length: {segments.total_length:.6f}")

        return segments

    elif args.generate:
        if not args.quiet:
            print(f"Generating {args.generate} curve at level {args.level}...")

        analyzer = setup_analyzer(args)
        segments = analyzer._generate_fractal(args.generate, args.level)

        if not args.quiet:
            print(f"Generated {segments.n_segments} segments")
            expected_segments = 4**args.level if args.generate == 'koch' else "varies"
            print(f"Expected segments: {expected_segments}")
            print(f"Total length: {segments.total_length:.6f}")

        return segments

    else:
        print("Error: Must specify either --file, --vtk, or --generate")
        sys.exit(1)


def baseline_analysis(analyzer: FastFractalAnalyzer, segments: SegmentArray, args) -> FractalResult:
    """
    Perform baseline analysis with no optimizations.

    Uses all box counting data with simple linear regression on log-log plot.
    """
    if not args.quiet:
        print("\n" + "="*50)
        print("BASELINE ANALYSIS (NO OPTIMIZATIONS)")
        print("="*50)
        print("• No grid optimization")
        print("• No boundary artifact removal")
        print("• No sliding window selection")
        print("• Simple regression on all data points")

    # Estimate box size range if not provided
    if args.min_box_size is None or args.max_box_size is None:
        auto_min, auto_max = analyzer._estimate_box_size_range(segments)
        min_box_size = args.min_box_size or auto_min
        max_box_size = args.max_box_size or auto_max
    else:
        min_box_size = args.min_box_size
        max_box_size = args.max_box_size

    # Get raw box counting data (no optimizations applied)
    box_sizes, box_counts = analyzer.box_counter.compute_box_spectrum(
        segments=segments,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
        size_factor=args.size_factor,
        grid_optimization=False  # Force disable
    )

    # Simple linear regression on ALL data (no window selection)
    log_sizes = np.log10(box_sizes)
    log_counts = np.log10(box_counts)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)
    dimension = -slope  # Negative slope = positive dimension
    r_squared = r_value**2

    # Create result object
    result = FractalResult(
        dimension=dimension,
        r_squared=r_squared,
        box_sizes=box_sizes,
        box_counts=box_counts,
        slope=slope,
        intercept=intercept,
        scaling_range=(box_sizes[-1], box_sizes[0]),  # Full range
        n_points=len(box_sizes)
    )

    return result


def analyze_linear_region(analyzer: FastFractalAnalyzer, segments: SegmentArray, args) -> FractalResult:
    """Perform optimized linear scaling region analysis."""
    if not args.quiet:
        print("\n" + "="*50)
        print("OPTIMIZED LINEAR SCALING REGION ANALYSIS")
        print("="*50)
        print("• Grid optimization enabled")
        print("• Boundary artifact removal enabled")
        print("• Sliding window selection for optimal scaling region")

    # Perform optimized analysis
    result = analyzer.compute_fractal_dimension(
        segments,
        min_box_size=args.min_box_size,
        max_box_size=args.max_box_size,
        size_factor=args.size_factor
    )

    return result


def save_results_csv(result: FractalResult, theoretical_dim: Optional[float], segments: SegmentArray, args) -> None:
    """Save analysis results to CSV file in output directory."""
    if not args.output:
        return

    csv_path = os.path.join(args.output, 'results.csv')

    # Prepare data row
    fractal_type = args.generate if hasattr(args, 'generate') and args.generate else 'file_input'
    level = args.level if hasattr(args, 'level') and args.level else 'N/A'

    # Calculate error statistics if theoretical dimension available
    error_abs = abs(result.dimension - theoretical_dim) if theoretical_dim else None
    error_pct = (error_abs / theoretical_dim * 100) if theoretical_dim and theoretical_dim != 0 else None

    # Create row data
    row_data = {
        'fractal_type': fractal_type,
        'level': level,
        'segments': segments.n_segments,
        'dimension': f"{result.dimension:.6f}",
        'r_squared': f"{result.r_squared:.6f}",
        'theoretical_dim': f"{theoretical_dim:.6f}" if theoretical_dim else 'N/A',
        'error_abs': f"{error_abs:.6f}" if error_abs else 'N/A',
        'error_pct': f"{error_pct:.2f}" if error_pct else 'N/A',
        'scaling_range_min': f"{result.scaling_range[0]:.6f}",
        'scaling_range_max': f"{result.scaling_range[1]:.6f}",
        'n_points': result.n_points,
        'is_valid': result.is_valid,
        'timestamp': datetime.now().isoformat()
    }

    # Write header if file doesn't exist
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

    if not args.quiet:
        print(f"  Results saved to: {os.path.basename(csv_path)}")


def save_interface_segments(segments: SegmentArray, args) -> None:
    """Save interface segments to data file in output directory."""
    if not args.output:
        return

    # Create filename based on fractal type and level
    fractal_type = args.generate if hasattr(args, 'generate') and args.generate else 'fractal'
    level = args.level if hasattr(args, 'level') and args.level else 'unknown'

    filename = f"interface_{fractal_type}_{level}.dat"
    file_path = os.path.join(args.output, filename)

    # Write segments in standard format
    with open(file_path, 'w') as f:
        f.write(f"# Interface segments for {fractal_type} level {level}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total segments: {segments.n_segments}\n")
        f.write("# Format: x1 y1 x2 y2\n")

        # Write segment data
        seg_data = segments.segments
        for i in range(segments.n_segments):
            x1, y1 = seg_data[i, 0, :]
            x2, y2 = seg_data[i, 1, :]
            f.write(f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n")

    if not args.quiet:
        print(f"  Interface segments saved to: {os.path.basename(file_path)}")


def display_results(result: FractalResult, theoretical_dim: Optional[float], args) -> None:
    """Display analysis results."""
    print(f"\nFRACTAL DIMENSION RESULTS:")
    print(f"Computed dimension: {result.dimension:.6f}")
    print(f"R-squared: {result.r_squared:.6f}")
    print(f"Scaling range: {result.scaling_range[1]:.6f} to {result.scaling_range[0]:.6f}")
    print(f"Number of points: {result.n_points}")
    print(f"Valid result: {'✓' if result.is_valid else '✗'}")

    if theoretical_dim is not None:
        error = abs(result.dimension - theoretical_dim)
        error_pct = error / theoretical_dim * 100
        print(f"Theoretical dimension: {theoretical_dim:.6f}")
        print(f"Error: {error:.6f} ({error_pct:.1f}%)")

        if error < 0.05:
            print("Accuracy: Excellent (< 5% error)")
        elif error < 0.1:
            print("Accuracy: Good (< 10% error)")
        else:
            print("Accuracy: Needs improvement (> 10% error)")


def generate_plots(result: FractalResult, segments: SegmentArray, args) -> None:
    """Generate plots for analysis results."""
    if not args.quiet:
        print(f"\nGenerating plots in {args.output}...")

    # Configure plot settings
    plot_config = PlotConfig(
        save_eps=args.eps_plots,
        save_png=True,  # Always save PNG
        use_titles=not args.no_titles,
        dpi=300
    )

    plotter = FractalPlotter(plot_config)

    # Generate box counting log-log plot
    try:
        # Pass fractal type and level if this is a generated fractal
        fractal_type = args.generate if hasattr(args, 'generate') and args.generate else None
        iteration_level = args.level if hasattr(args, 'level') and fractal_type else None

        saved_files = plotter.plot_box_counting_loglog(
            result, segments, args.output,
            fractal_type=fractal_type,
            iteration_level=iteration_level
        )
        if not args.quiet:
            for file_path in saved_files:
                print(f"  Saved: {os.path.basename(file_path)}")
    except Exception as e:
        if not args.quiet:
            print(f"  Warning: Could not generate box counting plot: {e}")

    # Generate interface plot if segments are not too numerous
    if segments.n_segments > 0 and segments.n_segments <= 5000:
        try:
            # Pass fractal type and level if this is a generated fractal
            fractal_type = args.generate if hasattr(args, 'generate') and args.generate else None
            iteration_level = args.level if hasattr(args, 'level') and fractal_type else None

            saved_files = plotter.plot_interface_segments(
                segments, args.output,
                fractal_type=fractal_type,
                iteration_level=iteration_level
            )
            if not args.quiet:
                for file_path in saved_files:
                    print(f"  Saved: {os.path.basename(file_path)}")
        except Exception as e:
            if not args.quiet:
                print(f"  Warning: Could not generate interface plot: {e}")

    # Generate sliding window analysis plot
    try:
        saved_files = plotter.plot_sliding_window_analysis(result, segments, args.output)
        if not args.quiet:
            for file_path in saved_files:
                print(f"  Saved: {os.path.basename(file_path)}")
    except Exception as e:
        if not args.quiet:
            print(f"  Warning: Could not generate sliding window plot: {e}")


def analyze_iterations(analyzer: FastFractalAnalyzer, args) -> None:
    """Analyze fractal across multiple iteration levels."""
    if not args.generate:
        print("Error: --analyze_iterations requires --generate")
        sys.exit(1)

    if not args.quiet:
        print("\n" + "="*50)
        print("ITERATION LEVEL ANALYSIS")
        print("="*50)
        print(f"Analyzing {args.generate} curve from level {args.min_level} to {args.max_level}")

    # Get theoretical dimension
    theoretical_dimensions = {
        'koch': np.log(4) / np.log(3),      # ≈ 1.2619
        'sierpinski': np.log(3) / np.log(2), # ≈ 1.5850
    }
    theoretical_dim = theoretical_dimensions.get(args.generate)
    if theoretical_dim and not args.quiet:
        print(f"Theoretical {args.generate} dimension: {theoretical_dim:.6f}")

    print(f"\nLevel | Segments |  Dimension  |   Error   |    R²    | Valid")
    print("-" * 65)

    results = []

    for level in range(args.min_level, args.max_level + 1):
        try:
            # Generate fractal at this level
            segments = analyzer._generate_fractal(args.generate, level)

            # Analyze
            result = analyzer.compute_fractal_dimension(
                segments,
                min_box_size=args.min_box_size,
                max_box_size=args.max_box_size,
                size_factor=args.size_factor
            )

            # Calculate error if theoretical dimension available
            error = abs(result.dimension - theoretical_dim) if theoretical_dim else 0

            # Display result
            valid_mark = "✓" if result.is_valid else "✗"
            print(f"{level:5d} | {segments.n_segments:8d} | {result.dimension:10.6f} | {error:8.6f} | {result.r_squared:.6f} | {valid_mark:>5}")

            results.append({
                'level': level,
                'segments': segments.n_segments,
                'dimension': result.dimension,
                'error': error,
                'r_squared': result.r_squared,
                'valid': result.is_valid
            })

        except Exception as e:
            print(f"{level:5d} | Failed: {str(e)[:40]}...")

    # Summary statistics
    if results and not args.quiet:
        valid_results = [r for r in results if r['valid']]
        if valid_results:
            dimensions = [r['dimension'] for r in valid_results]
            errors = [r['error'] for r in valid_results]

            print(f"\nSUMMARY (valid results only):")
            print(f"Mean dimension: {np.mean(dimensions):.6f} ± {np.std(dimensions):.6f}")
            if theoretical_dim:
                print(f"Mean error: {np.mean(errors):.6f}")
                print(f"Best result: Level {min(valid_results, key=lambda x: x['error'])['level']}")

    # Generate multi-scale plot if requested
    if not args.no_plots and args.output and results:
        try:
            plot_config = PlotConfig(
                save_eps=args.eps_plots,
                save_png=True,
                use_titles=not args.no_titles,
                dpi=300
            )
            plotter = FractalPlotter(plot_config)
            saved_files = plotter.plot_multi_scale_analysis(results, args.output)

            if not args.quiet:
                print(f"\nGenerating multi-scale plots in {args.output}...")
                for file_path in saved_files:
                    print(f"  Saved: {os.path.basename(file_path)}")

        except Exception as e:
            if not args.quiet:
                print(f"  Warning: Could not generate multi-scale plot: {e}")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.analyze_iterations and not args.generate:
        print("Error: --analyze_iterations requires --generate")
        sys.exit(1)

    if args.max_level <= args.min_level:
        print("Error: --max_level must be greater than --min_level")
        sys.exit(1)

    # Default to linear region analysis if no analysis specified
    if not args.analyze_iterations and not args.base_no_opt:
        args.analyze_linear_region = True

    try:
        # Setup analyzer
        analyzer = setup_analyzer(args)

        if not args.quiet:
            print("FastFractalAnalyzer v3 - High-Performance Fractal Analysis")
            print("=" * 60)

        # Perform requested analysis
        if args.analyze_iterations:
            analyze_iterations(analyzer, args)
        else:
            # Single analysis (baseline or optimized)
            segments = load_segments(args)

            # Get theoretical dimension if available
            theoretical_dimensions = {
                'koch': np.log(4) / np.log(3),      # ≈ 1.2619
                'sierpinski': np.log(3) / np.log(2), # ≈ 1.5850
                'hilbert': 2.0,                     # Space-filling curve
                'dragon': np.log(2) / np.log(np.sqrt(0.5)),  # ≈ 2.0 (approximation)
                'minkowski': np.log(8) / np.log(3), # ≈ 1.8928
            }
            theoretical_dim = None
            if args.generate:
                theoretical_dim = theoretical_dimensions.get(args.generate)

            if args.base_no_opt:
                result = baseline_analysis(analyzer, segments, args)
            else:
                result = analyze_linear_region(analyzer, segments, args)

            display_results(result, theoretical_dim, args)

            # Save data files if output directory specified
            if args.output:
                if not args.quiet:
                    print(f"\nSaving data files to {args.output}...")

                # Save CSV results
                save_results_csv(result, theoretical_dim, segments, args)

                # Save interface segments
                save_interface_segments(segments, args)

            # Generate plots if requested
            if not args.no_plots and args.output:
                generate_plots(result, segments, args)

        if not args.quiet:
            print("\nAnalysis complete.")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()