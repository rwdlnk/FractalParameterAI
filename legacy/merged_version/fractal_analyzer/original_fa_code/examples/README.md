# Fractal Analyzer Examples

This directory contains example scripts demonstrating how to use the `fractal-analyzer` package for calculating and analyzing fractal dimensions. These examples range from basic usage to advanced analysis techniques.

## Basic Examples

### Koch Curve Analysis

- `koch_1.py`: Simple example demonstrating how to generate a Koch curve fractal at iteration level 5 and calculate its fractal dimension using the linear region analysis method.

- `koch_2.py`: Similar to `koch_1.py` but with better output formatting and variable definitions.

- `koch_2_enhanced.py`: Enhanced Koch curve analysis that saves all visualization plots to a timestamped directory, including a combined visualization with multiple analysis elements.

## Iteration Analysis Examples

### Standard Iteration Analysis

- `iterations_basic.py`: Demonstrates how fractal dimension varies with iteration depth (level 1 to 6 by default) using the standard box counting method. Creates plots showing dimension convergence and saves detailed analysis reports.

### Linear Region Iteration Analysis

- `iterations_linear_region.py`: Demonstrates the enhanced approach of using linear region analysis at each iteration level for more accurate dimension calculation. Creates a heatmap visualization showing how window size selection affects dimension calculations across different iteration levels.

- `iterations_enhanced.py`: Comprehensive iteration analysis with detailed convergence analysis, extrapolation predictions, and multiple visualization plots. Supports both standard box counting and linear region methods.

## Advanced Examples

- `advanced_visualization.py`: Command-line tool for generating and analyzing various types of fractals with customizable parameters and extensive visualization options. Creates detailed reports, combined visualizations, and summary plots.

## Usage Examples

### Basic Linear Region Analysis

```python
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

# Create analyzer and tools
analyzer = FractalAnalyzer('koch')
analysis = FractalAnalysisTools(analyzer)

# Generate fractal
_, segments = analyzer.generate_fractal('koch', 5)

# Analyze optimal linear region for dimension calculation
windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis.analyze_linear_region(
    segments, 'koch', plot_results=True, plot_boxes=True)

print(f"Optimal dimension: {optimal_dimension:.6f}")
```

### Iteration Analysis with Linear Region

```python
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

# Create analyzer and tools
analyzer = FractalAnalyzer('koch')
analysis = FractalAnalysisTools(analyzer)

# Analyze how fractal dimension varies with iteration level
# Using optimal linear region selection for each level
levels, dimensions, errors, r_squared, optimal_windows, all_windows_data = analysis.analyze_iterations(
    min_level=1,
    max_level=8,
    fractal_type='koch',
    use_linear_region=True
)

# Print results
for i, level in enumerate(levels):
    print(f"Level {level}: D = {dimensions[i]:.6f} ± {errors[i]:.6f}, Window: {optimal_windows[i]}")
```

## Command Line Interface

Most examples support command-line arguments. For instance:

```bash
# Run advanced visualization on a Sierpinski fractal at iteration level 6
python advanced_visualization.py -t sierpinski -i 6 --output sierpinski_analysis

# Run enhanced iteration analysis with linear region method
python iterations_enhanced.py -t koch --min-level 1 --max-level 8 --use-linear-region
```

## Common Parameters

Most examples support the following command-line parameters:

- `-t, --type`: Fractal type (koch, sierpinski, minkowski, hilbert, dragon)
- `--min-level`, `--max-level`: Iteration level range for analysis
- `--use-linear-region`: Use optimal linear region analysis for each iteration level
- `-o, --output`: Output directory for saved plots and reports
- `--no-show`: Disable plot display (save only)

## Output Files

The example scripts generate various output files:

- Fractal curve plots (with or without box overlay)
- Log-log plots of box counting analysis
- Dimension vs. window size analysis plots
- Dimension vs. iteration level convergence plots
- Window selection heatmaps
- Combined visualizations with multiple plot types
- Detailed analysis reports in text format

All output files are saved to a timestamped directory by default, or to a specified output directory.
