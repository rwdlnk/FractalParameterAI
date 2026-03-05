Here's the top-level README.md in a format you can easily copy:

# Fractal Analyzer

A comprehensive Python package for fractal dimension analysis with specialized capabilities for Rayleigh-Taylor instability interface analysis.

## Overview

This package provides tools for analyzing the fractal properties of curves, with a particular focus on:

1. **General fractal analysis**: Calculate fractal dimensions of any piecewise linear curve using optimized box-counting methods
2. **Rayleigh-Taylor (RT) interface analysis**: Specialized tools for extracting and analyzing interfaces from RT instability simulations
3. **Multifractal analysis**: Comprehensive analysis of multifractal properties of interfaces
4. **Temporal evolution**: Tools to track how fractal properties evolve over time in simulations
5. **Resolution convergence**: Analysis of how fractal measurements converge with increasing simulation resolution

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fractal-analyzer.git
cd fractal-analyzer

# Install the package
pip install -e .
```

## Package Structure

- **fractal_analyzer/**: Core fractal analysis functionality
- **rt_analyzer/**: RT-specific analysis tools
- **scripts/**: Utility scripts for common analysis tasks
- **examples/**: Example applications and usage demonstrations

## Basic Usage

### Fractal Analysis

```python
from fractal_analyzer import FractalAnalyzer

# Create analyzer for Koch curve
analyzer = FractalAnalyzer('koch')

# Generate a Koch curve at iteration level 4
curve, segments = analyzer.generate_fractal('koch', level=4)

# Calculate fractal dimension
fd, error, box_sizes, box_counts, bounding_box, intercept = analyzer.calculate_fractal_dimension(segments)
print(f"Fractal dimension: {fd:.6f} ± {error:.6f}")

# Visualize results
plot_file = analyzer.visualizer.plot_fractal_curve(segments, bounding_box, plot_boxes=True,
                                                 box_sizes=box_sizes, box_counts=box_counts)
```

### RT Interface Analysis

```python
from rt_analyzer import RTAnalyzer

# Create analyzer
analyzer = RTAnalyzer("./output")

# Analyze a VTK file
result = analyzer.analyze_vtk_file("RT800x800-9000.vtk")

# Process a series of VTK files for temporal analysis
analyzer.process_vtk_series("RT800x800-*.vtk", resolution=800)

# Analyze resolution convergence
analyzer.analyze_resolution_convergence(
    ["RT100x100-9000.vtk", "RT200x200-9000.vtk", "RT400x400-9000.vtk", "RT800x800-9000.vtk"],
    [100, 200, 400, 800]
)
```

## Command Line Usage

### Fractal Analysis

```bash
# Generate a fractal curve
fractal-analyzer generate fractal --level 4 koch

# Calculate fractal dimension
fractal-analyzer analyze dimension --min-box-size 0.001 --plot interface.dat

# Analyze across iteration levels
fractal-analyzer iterations run --min-level 1 --max-level 6 koch
```

### RT Analysis

```bash
# Analyze a single VTK file
rt-analyzer single RT800x800-9000.vtk

# Process a series of VTK files
rt-analyzer series "RT800x800-*.vtk" --resolution 800

# Analyze resolution convergence
rt-analyzer convergence RT100x100-9000.vtk RT200x200-9000.vtk RT400x400-9000.vtk RT800x800-9000.vtk --resolutions 100 200 400 800

# Perform multifractal analysis
rt-analyzer multifractal RT800x800-9000.vtk

# Analyze temporal evolution
rt-analyzer temporal --data-dir ./RT800x800 --time-points 1.0 3.0 5.0 7.0 9.0
```

## License

[MIT License](LICENSE)

## Citation

If you use this software in your research, please cite:

```
Douglass, R. W. (2025). Fractal Analyzer: A tool for fractal dimension analysis of Rayleigh-Taylor instability interfaces.
```
