# Fractal Dimension Analysis Tool

A comprehensive Python toolkit for fractal dimension analysis using advanced box counting methods with grid optimization and enhanced boundary detection. Designed for both theoretical fractal validation and real-world applications including computational fluid dynamics interface analysis.

## Overview

This repository provides a complete fractal dimension analysis framework featuring:

- **Advanced box counting algorithms** with grid position optimization
- **Enhanced boundary artifact detection** and removal
- **Support for rectangular grids** and high aspect ratio data
- **Comprehensive validation suite** using known theoretical fractals
- **Real-world application examples** including Rayleigh-Taylor interface analysis
- **Publication-quality output** with EPS support for journal submission

## Key Features

### 🔬 **Scientific Rigor**
- Validated against theoretical fractals (Koch, Sierpinski, Minkowski, Hilbert, Dragon)
- Grid optimization reduces quantization error by 0.01-0.05 typical improvement
- Enhanced boundary detection outperforms manual trimming
- Statistical robustness testing and parameter sensitivity analysis

### 🚀 **Advanced Algorithms**
- **Grid position optimization**: Minimizes discretization bias in box counting
- **Linear region optimization**: Automatic selection of optimal scaling windows
- **Enhanced boundary detection**: Multi-criteria artifact identification and removal
- **Rectangular grid support**: Handles non-square computational grids with aspect ratio validation

### 🌊 **Real-World Applications**
- **Fluid dynamics interfaces**: Rayleigh-Taylor, Kelvin-Helmholtz instabilities
- **Geophysical data**: Coastlines, topographic surfaces, fault networks
- **Medical imaging**: Vascular networks, tumor boundaries, tissue interfaces
- **Materials science**: Fracture surfaces, porous media, composite structures

### 📊 **Professional Output**
- Publication-quality EPS plots for journal submission
- Comprehensive analysis reports with statistical metrics
- Automated parameter optimization and validation
- Integration with computational fluid dynamics workflows

## Installation

### Prerequisites
```bash
# Required Python packages
pip install numpy matplotlib scipy numba pathlib argparse
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/rwdlnk/Fractal-Dimension-Analyzer.git
cd Fractal-Dimension-Analyzer

# Test installation with Koch curve validation
python fractal_analyzer.py --generate koch --level 5
```

## Repository Structure

```
fractal-dimension-analysis/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── fractal_analyzer.py               # Main analysis tool (complete toolkit)
├── examples/                         # Comprehensive example suite
│   ├── koch_validation/              # Method validation & accuracy testing
│   │   ├── koch_validation.py
│   │   ├── README.md
│   │   └── run_analysis/
│   ├── rt_interface_analysis/        # Real-world fluid dynamics application
│   │   ├── rt_interface_analysis.py
│   │   ├── README.md
│   │   ├── sample_data/
│   │   └── run_analysis/
│   ├── dragon_robustness/            # Algorithm stress testing
│   │   ├── dragon_robustness.py
│   │   ├── README.md
│   │   └── run_analysis/
│   ├── sierpinski_boundary_effects/  # Boundary artifact analysis
│   │   ├── sierpinski_boundary_analysis.py
│   │   ├── README.md
│   │   └── run_analysis/
│   ├── minkowski_grid_optimization/  # Grid positioning effects
│   │   ├── minkowski_grid_optimization.py
│   │   ├── README.md
│   │   └── run_analysis/
│   └── hilbert_scaling_analysis/     # Basic workflow & method comparison
│       ├── hilbert_scaling_analysis.py
│       ├── README.md
│       └── run_analysis/
├── data/                             # Sample datasets and validation data
└── docs/                            # Additional documentation
```

## Quick Start Guide

### 1. **Validate Your Installation** (Koch Curve)
```bash
# Test with known theoretical fractal
python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region
# Expected: D ≈ 1.2619 ± 0.001
```

### 2. **Analyze Real Data** (RT Interface)
```bash
# Process fluid dynamics interface data
python fractal_analyzer.py --file interface_data.txt --rt_interface --validate_grid
# Auto-detects rectangular grids, applies physics-based parameters
```

### 3. **Compare Methods** (Hilbert Curve)
```bash
# Compare standard vs optimized box counting
python fractal_analyzer.py --generate hilbert --level 4 --optimization_comparison
# Demonstrates grid optimization benefits
```

### 4. **Advanced Analysis** (Custom Parameters)
```bash
# Full control over analysis parameters
python fractal_analyzer.py --file data.txt \
  --analyze_linear_region \
  --box_size_factor 1.4 \
  --use_grid_optimization \
  --eps_plots --no_titles
```

## Learning Path

### **Recommended Order for New Users:**

#### 1. **Start with Hilbert** (`examples/hilbert_scaling_analysis/`)
- **Purpose**: Learn basic workflow and method comparison
- **Key concepts**: Standard vs optimized methods, space-filling curves
- **Run**: `python hilbert_scaling_analysis.py --simple_workflow`

#### 2. **Validate with Koch** (`examples/koch_validation/`)
- **Purpose**: Verify algorithm accuracy with known theoretical dimension
- **Key concepts**: Convergence analysis, method validation
- **Run**: `python koch_validation.py`

#### 3. **Apply to Real Data** (`examples/rt_interface_analysis/`)
- **Purpose**: Learn real-world application workflow
- **Key concepts**: Grid auto-detection, physics-based parameters
- **Run**: `python rt_interface_analysis.py`

#### 4. **Test Robustness** (`examples/dragon_robustness/`)
- **Purpose**: Understand algorithm limits and stability
- **Key concepts**: Parameter sensitivity, stress testing
- **Run**: `python dragon_robustness.py --parameter_sweep`

#### 5. **Master Boundaries** (`examples/sierpinski_boundary_effects/`)
- **Purpose**: Handle complex boundary artifacts
- **Key concepts**: Enhanced detection, multi-scale effects
- **Run**: `python sierpinski_boundary_analysis.py --boundary_comparison`

#### 6. **Optimize Precision** (`examples/minkowski_grid_optimization/`)
- **Purpose**: Maximize measurement accuracy
- **Key concepts**: Quantization error, grid positioning effects
- **Run**: `python minkowski_grid_optimization.py --comprehensive_optimization`

## Core Analysis Methods

### **Standard Box Counting**
```bash
# Basic fractal dimension calculation
python fractal_analyzer.py --file data.txt --disable_grid_optimization
```

### **Grid-Optimized Box Counting**
```bash
# Enhanced accuracy through grid position optimization
python fractal_analyzer.py --file data.txt --use_grid_optimization
```

### **Linear Region Analysis**
```bash
# Automatic optimal scaling window selection
python fractal_analyzer.py --file data.txt --analyze_linear_region
```

### **Rectangular Grid Support**
```bash
# Handle non-square computational grids
python fractal_analyzer.py --file data.txt --nx 160 --ny 200 --validate_grid
```

## Algorithm Validation

### **Theoretical Fractals** (Known Dimensions)
| Fractal | Theoretical D | Typical Accuracy | Use Case |
|---------|---------------|------------------|----------|
| **Koch Curve** | 1.2619 | < 0.5% error | Method validation |
| **Sierpinski Triangle** | 1.5850 | < 1% error | Boundary testing |
| **Minkowski Sausage** | 1.5000 | < 0.1% error | Grid optimization |
| **Hilbert Curve** | 2.0000 | < 1% error | Space-filling analysis |
| **Dragon Curve** | 1.5236 | < 2% error | Robustness testing |

### **Validation Results**
- **Accuracy**: Consistently < 1% error for levels 4-6
- **Robustness**: Stable across parameter variations (CV < 2%)
- **Performance**: Grid optimization provides 0.01-0.05 improvement
- **Quality**: R² values > 0.998 for well-behaved fractals

## Real-World Applications

### **Computational Fluid Dynamics**
```python
# Rayleigh-Taylor interface analysis
from fractal_analyzer import FractalAnalyzer

analyzer = FractalAnalyzer()
segments = analyzer.read_line_segments('RT160x200_interface.txt')
results = analyzer.analyze_rt_interface(segments, nx=160, ny=200)
print(f"Interface complexity: D = {results[5]:.3f}")
```

### **Batch Processing**
```bash
# Process multiple time snapshots
for file in RT*_interface.txt; do
    python fractal_analyzer.py --file "$file" --rt_interface --no_plots
done
```

### **Publication Workflow**
```bash
# Generate journal-quality figures
python fractal_analyzer.py --file data.txt \
  --analyze_linear_region \
  --eps_plots --no_titles \
  --validate_grid
```

## Advanced Features

### **Grid Optimization**
- **Adaptive sampling**: More samples for smaller boxes
- **Quantization error reduction**: Typically 0.01-0.05 improvement
- **Performance scaling**: O(n^1.2) complexity for most cases
- **Statistical validation**: Percentile analysis vs random positions

### **Enhanced Boundary Detection**
- **Multi-criteria optimization**: Accuracy + quality + data retention
- **Scale-dependent adaptation**: Different strategies by complexity level
- **Automatic artifact identification**: Slope deviation and R² analysis
- **Robustness validation**: Consistency across parameter choices

### **Rectangular Grid Support**
- **Auto-detection**: Extracts grid info from file headers
- **Aspect ratio validation**: Warns about anisotropy effects (ratio > 3:1)
- **Physics-based parameters**: Grid-aware minimum box size estimation
- **Quality assessment**: Grid-specific validation metrics

## Technical Specifications

### **Input Data Formats**
- **Line segments**: `x1,y1 x2,y2` format (one per line)
- **Headers supported**: Auto-detection of grid configuration
- **Coordinate systems**: Any consistent physical units
- **File sizes**: Tested up to 50,000+ segments

### **Output Formats**
- **Plots**: PNG (default) or EPS (publication quality)
- **Data**: CSV export of box counting results
- **Reports**: Comprehensive analysis summaries
- **Validation**: Statistical robustness metrics

### **Performance**
- **Speed**: ~1-10 seconds for typical datasets
- **Memory**: Efficient spatial indexing for large datasets
- **Scalability**: Handles 10K+ segments with grid optimization
- **Accuracy**: Validated against theoretical fractals

## Citation

If you use this software in your research, please cite:

```bibtex
@software{fractal_analyzer_2025,
  title={Fractal Dimension Analysis Tool with Grid Optimization},
  author={[Rod Douglass]},
  year={2025},
  url={https://github.com/rwdlnk/box-counting-dimension-paper-2025},
  note={Enhanced box counting algorithms for scientific analysis}
}
```

## Contributing

We welcome contributions! Please see our guidelines:

### **Research Applications**
- Share your use cases and validation results
- Contribute domain-specific examples
- Report accuracy benchmarks for your field

### **Algorithm Improvements**
- Enhanced boundary detection methods
- New grid optimization strategies
- Performance optimizations

### **Documentation**
- Tutorial improvements
- Example clarifications
- Best practices guides

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

### **Documentation**
- **Examples**: Complete tutorial suite in `examples/` directory
- **Method validation**: Theoretical fractal test cases
- **Best practices**: Guidelines for different application domains

### **Community**
- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Share use cases and ask questions
- **Development**: Contribute improvements and extensions

### **Academic Support**
- **Methodology**: Detailed algorithm descriptions
- **Validation**: Comprehensive accuracy benchmarks
- **Publication**: Journal-ready output formats

## Acknowledgments

This work builds upon established fractal analysis methods and incorporates advances in:
- Grid position optimization for box counting accuracy
- Enhanced boundary artifact detection
- Rectangular grid support for computational fluid dynamics
- Statistical validation and robustness testing

Special thanks to the fractal analysis community for theoretical foundations and validation datasets.

---

## Quick Reference

### **Essential Commands**
```bash
# Validate installation
python fractal_analyzer.py --generate koch --level 5

# Analyze real data with optimization
python fractal_analyzer.py --file data.txt --analyze_linear_region

# Compare methods
python fractal_analyzer.py --generate hilbert --optimization_comparison

# Publication output
python fractal_analyzer.py --file data.txt --eps_plots --no_titles

# Rectangular grid analysis
python fractal_analyzer.py --file RT160x200.txt --rt_interface --validate_grid
```

### **Key Parameters**
- `--use_grid_optimization`: Enable enhanced accuracy (recommended)
- `--analyze_linear_region`: Automatic optimal scaling window
- `--validate_grid`: Check rectangular grid compatibility
- `--eps_plots`: Publication-quality vector graphics
- `--no_titles`: Journal submission format

### **Expected Accuracy**
- **Koch curves**: < 0.5% error from theoretical
- **Space-filling curves**: < 1% error from D = 2.0
- **Real interfaces**: Typically D = 1.1-1.8 for fluid systems
- **Grid optimization**: 0.01-0.05 typical improvement

🚀 **Ready to analyze fractals with scientific rigor and publication quality!**
