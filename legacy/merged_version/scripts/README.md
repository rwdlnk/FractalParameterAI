Here's the README.md for the scripts directory:

# scripts

Utility scripts for performing common fractal analysis tasks on RT interfaces.

## Overview

This directory contains utility scripts that simplify common analysis workflows by using both the `fractal_analyzer` and `rt_analyzer` packages. These scripts handle specific tasks like:

- Multifractal analysis
- Temporal evolution analysis
- Resolution dependence analysis

## Available Scripts

### run_multifractal.py

Performs multifractal analysis on RT interfaces with various options:

```bash
# Single file analysis
./run_multifractal.py --data-dir ./data --type single --time 9.0 --resolution 800

# Temporal evolution analysis
./run_multifractal.py --data-dir ./data --type temporal --resolution 800 --time-points 1.0 3.0 5.0 7.0 9.0

# Resolution dependence analysis
./run_multifractal.py --data-dir ./data --type resolution --time 9.0 --resolutions 100 200 400 800
```

### analyze_temporal.py

Analyzes how fractal properties evolve over time:

```bash
# Analyze multiple resolutions
./analyze_temporal.py --resolutions 100 200 400 800 --output ./temporal_analysis --pattern "./data/RT{resolution}x{resolution}/*.vtk"

# Analyze specific time points
./analyze_temporal.py --resolutions 800 --times 1.0 3.0 5.0 7.0 9.0 --output ./temporal_specific
```

### resolution_analysis.py

Analyzes how fractal properties converge with increasing resolution:

```bash
# Basic usage
./resolution_analysis.py --resolutions 100 200 400 800 --time 9.0 --output ./resolution_analysis
```

## Example Usage in Research Workflow

A typical research workflow using these scripts might look like:

1. First, analyze a single time point at various resolutions:
   ```bash
   ./resolution_analysis.py --resolutions 100 200 400 800 --time 9.0 --output ./res_analysis
   ```

2. Then, analyze the temporal evolution at the highest resolution:
   ```bash
   ./analyze_temporal.py --resolutions 800 --output ./temporal_analysis
   ```

3. Finally, perform detailed multifractal analysis:
   ```bash
   ./run_multifractal.py --data-dir ./data --type temporal --resolution 800
   ```

## Customization

These scripts can be easily modified to suit specific analysis needs. See the source code comments for details on additional parameters and customization options.

## Integration with Main Packages

These scripts serve as high-level interfaces to the functionality provided by the `fractal_analyzer` and `rt_analyzer` packages. They demonstrate how to combine various aspects of the packages for specific analysis tasks.
