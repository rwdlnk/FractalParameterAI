# Fractal Analyzer

A Python package for analyzing fractal dimensions using the box-counting method. This tool provides a comprehensive set of utilities for generating fractal curves, calculating their dimensions, and visualizing the results.

## Features

- Generate various fractal types (Koch, Sierpinski, Minkowski, Hilbert, Dragon)
- Calculate fractal dimensions using optimized box-counting algorithm
- Analyze how linear region selection affects measured dimensions
- Analyze how iteration level affects fractal dimension approximation
- Advanced dimension calculation using optimal linear region selection at each iteration level
- Visualize fractals and box-counting overlay
- Process line segment data from external sources (e.g., coastlines, interfaces between fluids)

## Installation

```bash
pip install fractal-analyzer
```

## Usage

### Command Line Interface

The package provides a comprehensive command-line interface:

```bash
fractal-analyzer [options]
```

#### Basic Options:

```bash
# Generate and analyze a Koch curve at level 5
fractal-analyzer --generate koch --level 5

# Analyze a file containing line segments
fractal-analyzer --file coastline.txt

# Disable plotting
fractal-analyzer --generate koch --level 5 --no_plot
```

#### Advanced Analysis:

```bash
# Analyze how linear region selection affects dimension
fractal-analyzer --generate sierpinski --level 6 --analyze_linear_region

# Analyze how iteration level affects dimension (standard method)
fractal-analyzer --generate koch --analyze_iterations --min_level 1 --max_level 8

# Analyze iterations using optimal linear region selection for each level
fractal-analyzer --generate koch --analyze_iterations --min_level 1 --max_level 8 --use_linear_region
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--file PATH` | Path to file containing line segments | None |
| `--generate {koch,sierpinski,minkowski,hilbert,dragon}` | Generate a fractal curve of specified type | None |
| `--level INT` | Level for fractal generation | 5 |
| `--min_box_size FLOAT` | Minimum box size for calculation | 0.001 |
| `--max_box_size FLOAT` | Maximum box size for calculation | Auto-determined |
| `--box_size_factor FLOAT` | Factor by which to reduce box size in each step | 2.0 |
| `--no_plot` | Disable plotting | False |
| `--no_box_plot` | Disable box overlay in the curve plot | False |
| `--analyze_iterations` | Analyze how iteration depth affects measured dimension | False |
| `--min_level INT` | Minimum curve level for iteration analysis | 1 |
| `--max_level INT` | Maximum curve level for iteration analysis | 8 |
| `--analyze_linear_region` | Analyze how linear region selection affects dimension | False |
| `--use_linear_region` | Use optimal linear region selection for each iteration level | False |
| `--fractal_type {koch,sierpinski,minkowski,hilbert,dragon}` | Specify fractal type for analysis when using `--file` | None |
| `--trim_boundary INT` | Number of box counts to trim from each end of the data | 0 |

### Python API

```python
from fractal_analyzer import FractalAnalyzer

# Create an analyzer
analyzer = FractalAnalyzer('koch')

# Generate a fractal
points, segments = analyzer.generate_fractal('koch', 5)

# Calculate dimension
dimension, error, box_sizes, box_counts, bbox, intercept = analyzer.calculate_fractal_dimension(segments)
print(f"Fractal dimension: {dimension:.6f} ± {error:.6f}")

# Visualize the fractal
analyzer.plot_results(segments, box_sizes, box_counts, dimension, error, bbox)
```

### Advanced Analysis Examples

#### Linear Region Analysis

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

#### Iteration Analysis with Optimal Linear Region Selection

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

## Input File Format

The analyzer accepts files containing line segments in the following format:

```
x1 y1 x2 y2
x1 y1 x2 y2
...
```

Where each line defines a line segment from point (x1, y1) to (x2, y2). Values can be separated by spaces or commas.

## Theoretical Dimensions

The package includes the following theoretical fractal dimensions for comparison:

| Fractal Type | Theoretical Dimension |
|--------------|------------------------|
| Koch         | 1.2619                 |
| Sierpinski   | 1.5850                 |
| Minkowski    | 1.5000                 |
| Hilbert      | 2.0000                 |
| Dragon       | 1.5236                 |

## Advanced Usage

### Linear Region Selection

The `analyze_linear_region` function uses a sliding window approach to identify the optimal scaling region for dimension calculation. This helps to:

- Find the most reliable linear region in the log-log plot
- Improve accuracy of dimension calculations
- Avoid including non-scaling parts of the curve in the calculation

### Iteration Level Analysis

The `analyze_iterations` function examines how fractal dimension varies with iteration depth:

- **Standard Method**: Uses direct box counting at each level
- **Linear Region Method**: Uses optimal linear region selection at each level (with `use_linear_region=True`)

The linear region method provides more accurate dimension estimates, especially at lower iteration levels where the fractal structure is less developed.

### Visualization Options

The package provides several visualization options:

- Fractal curve plots with or without box overlay
- Log-log plots of box counts vs. box sizes
- Dimension vs. window size analysis plots
- Dimension vs. iteration level plots
- Combined visualizations with multiple plot types

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
