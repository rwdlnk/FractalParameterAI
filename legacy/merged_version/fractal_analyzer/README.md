Let's print the README.md for the fractal_analyzer directory next:

# fractal_analyzer

Core fractal analysis functionality for general fractal dimension calculation using box counting.

## Overview

This module provides the fundamental components for fractal dimension analysis:

- **FractalBase**: Base class with common utilities for fractal analysis
- **BoxCounter**: Implementation of the box counting algorithm for dimension calculation
- **FractalAnalyzer**: High-level interface for fractal analysis
- **FractalVisualizer**: Tools for visualizing fractals and box counting results
- **FractalAnalysisTools**: Advanced analysis tools for studying scaling regions

## Components

### FractalBase (core.py)

Base class providing:
- Line segment reading/writing
- Line-box intersection detection
- Theoretical dimensions for common fractals

### BoxCounter (analysis.py)

Optimized box counting algorithm with:
- Spatial indexing for efficient intersection tests
- Multi-scale dimension calculation
- Statistical analysis of dimension confidence

### FractalAnalyzer (main.py)

High-level interface for:
- Fractal curve generation
- Dimension calculation
- Results visualization

### FractalVisualizer (visualization.py)

Visualization tools for:
- Rendering fractal curves
- Overlay of counting boxes
- Log-log plots for dimension calculation
- Dimension vs. level plots

### FractalAnalysisTools (analysis_tools.py)

Advanced tools for:
- Linear region selection
- Multi-scale analysis
- Iteration level analysis

## Example Usage

```python
from fractal_analyzer import FractalAnalyzer

# Basic analysis of a Koch curve
analyzer = FractalAnalyzer(fractal_type='koch')
curve, segments = analyzer.generate_fractal('koch', level=4)
fd, error, box_sizes, box_counts, bbox, intercept = analyzer.calculate_fractal_dimension(segments)
print(f"Koch curve fractal dimension: {fd:.6f} ± {error:.6f}")

# Advanced linear region analysis
from fractal_analyzer.analysis_tools import FractalAnalysisTools
tools = FractalAnalysisTools(analyzer)
windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = tools.analyze_linear_region(
    segments, fractal_type='koch', plot_results=True
)
print(f"Optimal dimension: {optimal_dimension:.6f} (window: {optimal_window})")
```

## Available Fractal Types

The package includes built-in generators for common fractal types:

- `koch`: Koch curve (theoretical dimension: 1.2619)
- `sierpinski`: Sierpinski triangle (theoretical dimension: 1.5850)
- `minkowski`: Minkowski curve (theoretical dimension: 1.5)
- `hilbert`: Hilbert curve (theoretical dimension: 2.0)
- `dragon`: Dragon curve (theoretical dimension: 1.5236)

Custom fractals can be analyzed by providing segment data directly or through files.

## Command Line Interface

The module provides a command-line interface for common tasks:

```bash
# Generate a fractal curve
fractal-analyzer generate fractal --level 4 koch

# Calculate dimension
fractal-analyzer analyze dimension --min-box-size 0.001 --plot interface.dat

# Study how dimension changes with iteration level
fractal-analyzer iterations run --min-level 1 --max-level 6 koch

# Analyze the linear region selection
fractal-analyzer analyze linear-region --min-window 3 --max-window 10 --plot koch
```
