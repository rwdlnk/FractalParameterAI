Next, here's the README.md for the rt_analyzer directory:

# rt_analyzer

Specialized tools for analyzing fractal properties of Rayleigh-Taylor instability interfaces.

## Overview

This module extends the core fractal analysis capabilities to specifically handle Rayleigh-Taylor (RT) instability interfaces extracted from simulation data. Key features include:

- VTK file processing for RT simulation outputs
- Interface extraction using contour algorithms
- Mixing layer thickness calculation
- Single and multi-resolution fractal dimension analysis
- Multifractal spectrum analysis
- Temporal evolution analysis
- Resolution convergence studies

## Components

### RTAnalyzer (rt_analyzer.py)

Main class providing:
- VTK file reading and processing
- Interface extraction and conversion to segments
- Mixing layer thickness calculation
- Fractal dimension calculation
- Multifractal spectrum analysis
- Temporal evolution analysis
- Resolution convergence extrapolation

### Visualization (rt_visualization.py)

Specialized visualization tools for:
- Temporal evolution plotting
- Multi-resolution comparison
- Mixing layer and fractal dimension correlation
- Extrapolation to infinite resolution

### Command Line Interface (cli.py)

Comprehensive CLI for:
- Single file analysis
- Series analysis
- Resolution convergence analysis
- Multifractal analysis
- Temporal analysis
- Resolution dependence studies

## Example Usage

```python
from rt_analyzer import RTAnalyzer

# Single file analysis
analyzer = RTAnalyzer("./output")
result = analyzer.analyze_vtk_file("RT800x800-9000.vtk", analyze_linear=True)
print(f"Fractal dimension: {result['fractal_dim']:.6f} ± {result['fd_error']:.6f}")

# Process a time series
analyzer.process_vtk_series("RT800x800-*.vtk", resolution=800)

# Resolution convergence analysis
analyzer.analyze_resolution_convergence(
    ["RT100x100-9000.vtk", "RT200x200-9000.vtk", "RT400x400-9000.vtk", "RT800x800-9000.vtk"],
    [100, 200, 400, 800]
)

# Multifractal analysis
analyzer.analyze_multifractal_single("RT800x800-9000.vtk", output_dir="./multifractal")

# Temporal evolution
analyzer.analyze_temporal_evolution({
    1.0: "RT800x800-1000.vtk",
    3.0: "RT800x800-3000.vtk",
    5.0: "RT800x800-5000.vtk",
    7.0: "RT800x800-7000.vtk",
    9.0: "RT800x800-9000.vtk"
})
```

## Command Line Interface

```bash
# Analyze a single VTK file
rt-analyzer single RT800x800-9000.vtk --analyze-linear

# Process a series of VTK files
rt-analyzer series "RT800x800-*.vtk" --resolution 800

# Analyze resolution convergence
rt-analyzer convergence RT100x100-9000.vtk RT200x200-9000.vtk RT400x400-9000.vtk RT800x800-9000.vtk --resolutions 100 200 400 800

# Perform multifractal analysis
rt-analyzer multifractal RT800x800-9000.vtk --qmin -5 --qmax 5 --qstep 0.5

# Analyze temporal evolution
rt-analyzer temporal --data-dir ./RT800x800 --time-points 1.0 3.0 5.0 7.0 9.0

# Analyze resolution dependence
rt-analyzer resolution --data-dir ./data --resolutions 100 200 400 800 --time 9.0
```

## VTK File Support

The module supports standard VTK rectilinear grid files containing volume fraction (F) data for RT simulations. Both cell-centered and point-centered data are supported.

## Analysis Methodology

1. The RT interface is extracted from the volume fraction field using the 0.5 contour
2. The interface is converted to line segments for fractal analysis
3. Box counting is applied across multiple scales to calculate fractal dimension
4. For multifractal analysis, q-moments are calculated to determine the multifractal spectrum

## Integration with fractal_analyzer

This module seamlessly integrates with the core `fractal_analyzer` package to leverage its optimized box counting algorithms and visualization tools, while adding RT-specific functionality for interface analysis.
