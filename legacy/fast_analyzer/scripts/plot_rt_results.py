#!/usr/bin/env python3
"""
RT Results Plotting CLI
======================

Command-line interface for generating publication-ready plots from RT analysis results.

Usage Examples:
    # Generate all plots from single resolution
    python plot_rt_results.py /path/to/320x400/batch_results_comprehensive.csv

    # Grid convergence analysis across multiple resolutions
    python plot_rt_results.py --grid-convergence 320x400/,640x800/,1280x1600/ --output plots/convergence/

    # Publication-quality EPS figures only
    python plot_rt_results.py data.csv --format eps --style journal --output figures/

    # Specific plot types only
    python plot_rt_results.py data.csv --plots dimension,mixing --output plots/

Features:
- Individual evolution plots (dimension, mixing, RMS, multifractal)
- Grid convergence analysis across resolutions
- Publication-ready PNG and EPS output
- Customizable styling (journal, presentation)
- Comprehensive multi-panel summaries
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fractal_analyzer.plotting.rt_plotter import RTPlotter


def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Generate publication-ready plots from RT analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots from single CSV
  python plot_rt_results.py results/batch_results_comprehensive.csv

  # Grid convergence analysis
  python plot_rt_results.py --grid-convergence 320x400/,640x800/,1280x1600/ \\
                            --output plots/convergence/

  # Publication figures only (EPS format)
  python plot_rt_results.py data.csv --format eps --style journal \\
                            --output figures/

  # Specific plot types
  python plot_rt_results.py data.csv --plots dimension,mixing,rms \\
                            --output plots/evolution/

  # High-resolution presentation plots
  python plot_rt_results.py data.csv --style presentation --dpi 600 \\
                            --output presentation_plots/
        """
    )

    # Input specification (not required if showing help)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('csv_file', nargs='?',
                           help='Path to comprehensive CSV results file')
    input_group.add_argument('--grid-convergence',
                           help='Comma-separated list of resolution directories for convergence analysis')

    # Output options
    parser.add_argument('--output', '-o', type=Path, default=Path('./rt_plots'),
                       help='Output directory for plots (default: ./rt_plots)')
    parser.add_argument('--format', choices=['png', 'eps', 'both'], default='both',
                       help='Output format (default: both)')
    parser.add_argument('--style', choices=['journal', 'presentation', 'paper'],
                       default='journal', help='Plot style (default: journal)')

    # Plot selection
    parser.add_argument('--plots',
                       help='Comma-separated list of plot types: dimension,mixing,rms,multifractal,summary')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip comprehensive summary plot')

    # Styling options
    parser.add_argument('--dpi', type=int, default=300,
                       help='Resolution for PNG output (default: 300)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 6],
                       help='Figure size in inches (default: 10 6)')

    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--list-available', action='store_true',
                       help='List available plot types and exit')

    return parser


def validate_csv_file(csv_path: Path) -> bool:
    """Validate CSV file exists and has required columns."""
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        required_columns = ['time', 'fractal_dimension']

        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Required column '{col}' not found in CSV")
                return False

        print(f"✅ CSV file validated: {len(df)} records, {len(df.columns)} columns")
        return True

    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return False


def parse_grid_convergence_dirs(convergence_arg: str) -> List[Path]:
    """Parse grid convergence directory argument."""
    dirs = [Path(d.strip()) for d in convergence_arg.split(',')]

    valid_dirs = []
    for dir_path in dirs:
        csv_file = dir_path / 'batch_results_comprehensive.csv'
        if csv_file.exists():
            valid_dirs.append(csv_file)
            print(f"✅ Found convergence data: {dir_path}")
        else:
            print(f"⚠️  Missing CSV in convergence directory: {dir_path}")

    return valid_dirs


def generate_single_resolution_plots(args):
    """Generate plots from single resolution CSV file."""
    csv_path = Path(args.csv_file)

    if not validate_csv_file(csv_path):
        return False

    # Setup output formats
    if args.format == 'both':
        formats = ['png', 'eps']
    else:
        formats = [args.format]

    # Initialize plotter
    plotter = RTPlotter(
        style=args.style,
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )

    print(f"\n📊 Generating RT plots from {csv_path.name}")
    print(f"   Style: {args.style}")
    print(f"   Formats: {formats}")
    print(f"   Output: {args.output}")

    # Determine which plots to generate
    if args.plots:
        plot_types = [p.strip() for p in args.plots.split(',')]
    else:
        plot_types = ['dimension', 'mixing', 'rms', 'multifractal']
        if not args.no_summary:
            plot_types.append('summary')

    # Generate requested plots
    results = {}

    if 'dimension' in plot_types:
        results['dimension'] = plotter.plot_dimension_evolution(csv_path, args.output, formats)

    if 'mixing' in plot_types:
        results['mixing'] = plotter.plot_mixing_evolution(csv_path, args.output, formats)

    if 'rms' in plot_types:
        results['rms'] = plotter.plot_rms_evolution(csv_path, args.output, formats)

    if 'multifractal' in plot_types:
        results['multifractal'] = plotter.plot_multifractal_evolution(csv_path, args.output, formats)

    if 'summary' in plot_types:
        results['summary'] = plotter.plot_comprehensive_summary(csv_path, args.output, formats)

    # Summary
    successful = sum(results.values())
    total = len(results)
    print(f"\n✅ Generated {successful}/{total} plot types successfully")
    print(f"📁 Plots saved to: {args.output}")

    return successful > 0


def generate_grid_convergence_plots(args):
    """Generate grid convergence analysis plots."""
    csv_files = parse_grid_convergence_dirs(args.grid_convergence)

    if len(csv_files) < 2:
        print("❌ Need at least 2 valid resolution directories for convergence analysis")
        return False

    # Extract resolution labels from paths
    # Search up the directory tree to find grid resolution (e.g., "160x200", "320x400")
    resolutions = []
    for csv_file in csv_files:
        # Walk up the path to find a directory matching grid resolution pattern (NxM)
        import re
        resolution = "unknown"
        for part in csv_file.parts:
            # Match pattern like "160x200", "320x400", "640x800", "960x1200"
            if re.match(r'^\d+x\d+$', part):
                resolution = part
                break
        resolutions.append(resolution)

    # Setup output formats
    if args.format == 'both':
        formats = ['png', 'eps']
    else:
        formats = [args.format]

    # Initialize plotter
    plotter = RTPlotter(
        style=args.style,
        dpi=args.dpi,
        figsize=tuple(args.figsize)
    )

    print(f"\n📊 Generating grid convergence analysis")
    print(f"   Resolutions: {resolutions}")
    print(f"   CSV files: {len(csv_files)}")
    print(f"   Output: {args.output}")

    # Generate convergence plot
    success = plotter.plot_grid_convergence(csv_files, resolutions, args.output, formats)

    if success:
        print(f"\n✅ Grid convergence analysis completed successfully")
        print(f"📁 Plot saved to: {args.output}")
    else:
        print(f"\n❌ Grid convergence analysis failed")

    return success


def list_available_plots():
    """List available plot types."""
    print("📊 Available RT Plot Types:")
    print("=" * 40)
    print("Individual Evolution Plots:")
    print("  dimension    - Fractal dimension vs time")
    print("  mixing       - Mixing width, thickness, Dalziel h")
    print("  rms          - RMS velocity evolution (u_rms, v_rms, total_rms)")
    print("  multifractal - Generalized dimensions (D₀, D₁, D₂)")
    print("\nSummary Plots:")
    print("  summary      - Multi-panel comprehensive overview")
    print("\nGrid Convergence:")
    print("  Use --grid-convergence with multiple resolution directories")
    print("\nFormats:")
    print("  png          - High-resolution raster (quick visualization)")
    print("  eps          - Vector format (publication quality)")
    print("  both         - Generate both formats")
    print("\nStyles:")
    print("  journal      - Publication-ready with serif fonts")
    print("  presentation - Large fonts for talks")
    print("  paper        - Balanced style for manuscripts")


def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    if args.list_available:
        list_available_plots()
        return 0

    # Validate input arguments
    if not args.csv_file and not args.grid_convergence:
        print("❌ Error: Must provide either CSV file or --grid-convergence")
        parser.print_help()
        return 1

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        if args.grid_convergence:
            success = generate_grid_convergence_plots(args)
        else:
            success = generate_single_resolution_plots(args)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Plotting interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())