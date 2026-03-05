# FastFractalAnalyzer v3

High-performance fractal dimension analysis with three-part optimization framework and publication-quality plotting.

## Overview

FastFractalAnalyzer v3 implements a comprehensive three-part optimization framework for accurate fractal dimension computation:

1. **Grid-Optimized Box Counting** - Eliminates discretization artifacts through adaptive grid testing
2. **Enhanced Boundary Artifact Removal** - Statistical detection and removal of boundary effects
3. **Sliding Window Scaling Region Selection** - Objective identification of optimal linear scaling regions

## Key Features

### **Algorithm Excellence**
- **Three-part optimization framework** with only 2 user parameters (θ_slope=0.12, θ_R²=0.99)
- **Grid optimization** with adaptive resolution (2×2, 3×3, 4×4 based on complexity)
- **Boundary artifact detection** using statistical criteria
- **Sliding window analysis** for objective scaling region selection
- **Excellent accuracy**: <1% error on validation fractals

### **Built-in Fractal Generators**
- **Koch Curve** - Classic self-similar fractal
- **Sierpinski Triangle** - Triangular subdivision fractal
- **Dragon Curve** - Space-filling curve
- **Minkowski Sausage** - Rectangular bulge pattern
- **Hilbert Curve** - Space-filling curve (up to 1M+ segments)

### **Advanced Performance**
- **JIT compilation** with Numba for numerical algorithms
- **Spatial indexing** with Liang-Barsky line clipping
- **LRU caching** with 10⁴-10⁵× speedups for repeated computations
- **VTK file support** for Rayleigh-Taylor simulation analysis

### **Publication-Quality Plotting**
- **Enhanced visualization** with fractal context (type and iteration level)
- **Multiple formats** (PNG, EPS) for journal submission
- **Statistical quality indicators** (R² values, validity markers)
- **Professional layouts** with clean metadata positioning

## Quick Start

### CLI Usage
```bash
# Analyze Koch curve level 6
python -m fractal_analyzer.cli --generate koch --level 6 --output ./plots --eps_plots

# Analyze iteration progression
python -m fractal_analyzer.cli --generate sierpinski --analyze_iterations --min_level 3 --max_level 7 --output ./plots

# Analyze VTK simulation data
python -m fractal_analyzer.cli --vtk simulation.vtk --output ./plots

# Analyze segment file
python -m fractal_analyzer.cli --file segments.txt --output ./plots
```

### Python API
```python
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer

# Create analyzer with default settings
analyzer = FastFractalAnalyzer()

# Generate mathematical fractal
segments = analyzer._generate_fractal('koch', level=6)

# Compute fractal dimension
result = analyzer.compute_fractal_dimension(segments)
print(f"Dimension: {result.dimension:.6f} (R² = {result.r_squared:.6f})")

# Create publication plots
from fractal_analyzer.plotting.fractal_plots import FractalPlotter, PlotConfig
config = PlotConfig(save_eps=True, save_png=True)
plotter = FractalPlotter(config)
plotter.plot_box_counting_loglog(result, segments, './plots',
                                fractal_type='koch', iteration_level=6)
```

## Algorithm Validation

Excellent accuracy across all fractal types:
- **Koch Curve**: <0.2% error
- **Sierpinski Triangle**: 0.5% error (level 6), 0.2% error (level 7)
- **Dragon Curve**: <1% error
- **Minkowski Sausage**: <1% error
- **Hilbert Curve**: <3% error approaching space-filling dimension

## Performance

- **Grid optimization**: 10-40% improvement over standard box counting
- **Caching system**: 10⁴-10⁵× speedups for repeated computations
- **Spatial indexing**: 3×+ speedup for complex datasets
- **Scalability**: Handles 1M+ segments (Hilbert Level 10)

## Requirements

- Python 3.9+
- NumPy 1.21+
- SciPy 1.7+
- Numba 0.56+

## Installation

```bash
pip install fast-fractal-analyzer
```

## License

MIT License