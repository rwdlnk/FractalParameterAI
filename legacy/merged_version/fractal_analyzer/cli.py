#!/usr/bin/env python3
# cli.py
"""Command line interface for fractal dimension analysis."""

import click
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

from .core import FractalBase
from .analysis import BoxCounter
from .visualization import FractalVisualizer
from . import FractalAnalyzer
from .analysis_tools import FractalAnalysisTools

@click.group()
def cli():
    """Fractal analyzer command line tool."""
    pass

@cli.group()
def generate():
    """Generate fractals."""
    pass

@generate.command()
@click.option("--level", default=5, type=int, help="Iteration level")
@click.option("--output", default=None, help="Output file name")
@click.argument("fractal_type", type=str)
def fractal(level, output, fractal_type):
    """Generate a fractal curve."""
    analyzer = FractalAnalyzer(fractal_type)
    
    click.echo(f"Generating {fractal_type} fractal at level {level}...")
    curve, segments = analyzer.generate_fractal(fractal_type, level)
    click.echo(f"Generated {len(segments)} line segments")
    
    if output:
        analyzer.write_segments_to_file(segments, output)
        click.echo(f"Saved segments to {output}")
    
    # Plot the fractal
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 10))
    x, y = zip(*curve)
    plt.plot(x, y, 'k-')
    plt.axis('equal')
    plt.title(f"{fractal_type.capitalize()} curve (level {level})")
    
    if output:
        plot_file = f"{os.path.splitext(output)[0]}.png"
        plt.savefig(plot_file)
        click.echo(f"Saved plot to {plot_file}")
    else:
        plt.show()

@cli.group()
def analyze():
    """Analyze fractals."""
    pass

@analyze.command()
@click.option("--min-box-size", default=0.001, type=float, help="Minimum box size")
@click.option("--max-box-size", default=None, type=float, help="Maximum box size")
@click.option("--box-factor", default=2.0, type=float, help="Box size reduction factor")
@click.option("--plot", is_flag=True, help="Create visualization plot")
@click.option("--plot-boxes", is_flag=True, help="Plot boxes on the fractal")
@click.option("--output", default=None, help="Output file name")
@click.argument("file", type=str)
def dimension(min_box_size, max_box_size, box_factor, plot, plot_boxes, output, file):
    """Calculate fractal dimension from a file."""
    # Determine fractal type from file name if possible
    import os
    filename = os.path.basename(file)
    fractal_type = None
    
    # Try to extract type from filename
    known_types = ["koch", "sierpinski", "minkowski", "hilbert", "dragon"]
    for t in known_types:
        if t in filename.lower():
            fractal_type = t
            break
    
    analyzer = FractalAnalyzer(fractal_type)
    
    click.echo(f"Reading segments from {file}...")
    segments = analyzer.read_line_segments(file)
    click.echo(f"Read {len(segments)} line segments")
    
    click.echo(f"Calculating fractal dimension...")
    fd, error, box_sizes, box_counts, bounding_box, intercept = analyzer.calculate_fractal_dimension(
        segments, min_box_size, max_box_size, box_factor)
    click.echo(f"Fractal dimension: {fd:.6f} ± {error:.6f}")
    
    if plot:
        click.echo(f"Creating visualization...")
        analyzer.plot_results(segments, box_sizes, box_counts, fd, error, bounding_box, 
                             plot_boxes=plot_boxes, custom_filename=output)
        click.echo(f"Visualization complete.")

@analyze.command()
@click.option("--min-window", default=3, type=int, help="Minimum window size")
@click.option("--max-window", default=None, type=int, help="Maximum window size")
@click.option("--plot", is_flag=True, help="Plot results")
@click.option("--output", default=None, help="Output directory")
@click.option("--level", default=5, type=int, help="Iteration level for generated fractals")
@click.argument("fractal_type", type=str)
def linear_region(min_window, max_window, plot, output, level, fractal_type):
    """Analyze the linear region selection for dimension calculation."""
    analyzer = FractalAnalyzer(fractal_type)
    analysis = FractalAnalysisTools(analyzer)
    
    # Create output directory if saving results
    if output:
        os.makedirs(output, exist_ok=True)
    
    click.echo(f"Generating {fractal_type} fractal at level {level}...")
    _, segments = analyzer.generate_fractal(fractal_type, level)
    click.echo(f"Generated {len(segments)} line segments")
    
    click.echo(f"Analyzing linear region selection...")
    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis.analyze_linear_region(
        segments, fractal_type=fractal_type, plot_results=plot,
        save_plots=(output is not None), output_dir=output
    )
    
    click.echo(f"Optimal window size: {optimal_window}")
    click.echo(f"Optimal dimension: {optimal_dimension:.6f}")
    click.echo(f"R-squared: {r_squared[windows.index(optimal_window)]:.6f}")
    
    if output:
        click.echo(f"Results saved to directory: {output}")

@cli.group()
def iterations():
    """Analyze fractal dimension across iteration levels."""
    pass

@iterations.command(name="run")
@click.option("--min-level", default=1, type=int, help="Minimum iteration level")
@click.option("--max-level", default=6, type=int, help="Maximum iteration level")
@click.option("--no-plots", is_flag=True, help="Disable plotting")
@click.option("--no-box-plot", is_flag=True, help="Disable box plotting")
@click.option("--use-linear-region", is_flag=True, help="Use linear region analysis for each level")
@click.option("--output-dir", default=None, help="Output directory for plots")
@click.argument("fractal_type", type=str)
def run_iterations(min_level, max_level, no_plots, no_box_plot, use_linear_region, output_dir, fractal_type):
    """Run iteration analysis on a fractal."""
    analyzer = FractalAnalyzer(fractal_type)
    analysis = FractalAnalysisTools(analyzer)
    
    if use_linear_region:
        click.echo(f"Running iteration analysis ({min_level}-{max_level}) with linear region method...")
        levels, dimensions, errors, r_squared, optimal_windows, _ = analysis.analyze_iterations(
            min_level=min_level,
            max_level=max_level,
            fractal_type=fractal_type,
            no_plots=no_plots,
            no_box_plot=no_box_plot,
            use_linear_region=True,
            save_plots=(output_dir is not None),
            output_dir=output_dir
        )
        # Display results with optimal window info
        for i, level in enumerate(levels):
            click.echo(f"Level {level}: D = {dimensions[i]:.6f} ± {errors[i]:.6f} (Window: {optimal_windows[i]}, R²: {r_squared[i]:.6f})")
    else:
        click.echo(f"Running iteration analysis ({min_level}-{max_level})...")
        levels, dimensions, errors, r_squared = analysis.analyze_iterations(
            min_level=min_level,
            max_level=max_level,
            fractal_type=fractal_type,
            no_plots=no_plots,
            no_box_plot=no_box_plot,
            save_plots=(output_dir is not None),
            output_dir=output_dir
        )
        # Display results
        for i, level in enumerate(levels):
            click.echo(f"Level {level}: D = {dimensions[i]:.6f} ± {errors[i]:.6f} (R²: {r_squared[i]:.6f})")
    
    if output_dir:
        click.echo(f"Results saved to directory: {output_dir}")

if __name__ == "__main__":
    cli()
