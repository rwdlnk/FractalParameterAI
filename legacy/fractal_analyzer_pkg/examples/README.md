# Fractal Analyzer Examples

This directory contains examples demonstrating the use of the fractal-analyzer package for analyzing fractal dimensions and generating visualizations.

## Basic Examples

### Koch Curve Analysis (koch_2_enhanced.py)

A comprehensive example demonstrating how to generate and analyze a Koch curve at a specific iteration level, with plot saving capabilities.

```bash
python koch_2_enhanced.py
```

This will:
1. Generate a Koch curve at iteration level 5
2. Perform box counting and fractal dimension analysis
3. Create multiple visualization types (curve, log-log plot, dimension analysis)
4. Generate a combined visualization with all analysis elements
5. Save all plots to a timestamped output directory

## Advanced Examples

### Advanced Visualization (advanced_visualization.py)

A flexible command-line tool for analyzing various fractal types with customizable parameters.

```bash
python advanced_visualization.py --type koch --iterations 6 --output my_results
```

Options:
- `--type` or `-t`: Fractal type (koch, sierpinski, minkowski, hilbert, dragon)
- `--iterations` or `-i`: Number of iterations for fractal generation
- `--output` or `-o`: Custom output directory for saved plots
- `--no-boxes`: Disable box counting visualization (faster for complex fractals)
- `--no-show`: Do not display plots interactively (only save to files)

This will:
1. Generate the specified fractal at the requested iteration level
2. Perform detailed fractal dimension analysis
3. Create a comprehensive set of visualizations
4. Generate a summary report with analysis results
5. Save all outputs to the specified directory

## Generated Visualizations

Each analysis run produces the following visualizations:

1. **Fractal curve**: The generated fractal shape
2. **Log-log plot**: Box count vs. box size in logarithmic scale
3. **Dimension analysis plot**: Analysis of how window size affects dimension calculation
4. **Combined analysis**: A comprehensive visualization with all elements in one image

## Important Notes

- Ensure your main FractalAnalyzer class properly initializes the visualizer with a reference to the base object:
  ```python
  self.visualizer = FractalVisualizer(fractal_type, self.base)
  ```
  
- If you encounter a "NoneType has no attribute" error related to the visualizer, you can add this fix to your code:
  ```python
  if not hasattr(analyzer.visualizer, 'base') or analyzer.visualizer.base is None:
      analyzer.visualizer.base = analyzer.base
  ```

## Custom Analysis

These examples can be adapted to analyze other fractals or piecewise linear curves by:

1. Adding a new fractal type to the `FractalBase.THEORETICAL_DIMENSIONS` dictionary (if known)
2. Implementing the fractal generation method in the `FractalAnalyzer` class
3. Running the advanced visualization script with the new fractal type

For fractal dimension calculation of custom data (e.g., fluid interfaces), you can:
1. Load your data as line segments
2. Pass the segments directly to the `analyze_linear_region` method
3. Use the optional parameters to customize the analysis and visualization

## Requirements

- Python 3.6 or higher
- NumPy
- SciPy
- Matplotlib
