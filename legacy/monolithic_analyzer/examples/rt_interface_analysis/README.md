# RT Interface Fractal Analysis

This example demonstrates fractal dimension analysis of Rayleigh-Taylor interfaces from computational fluid dynamics simulations, showcasing the complete workflow from simulation data to publication-ready results.

## Overview

Rayleigh-Taylor interfaces provide an excellent real-world application for fractal analysis because:
- **Complex geometry**: Turbulent mixing creates fractal-like structures
- **Physical relevance**: Dimension relates to mixing efficiency and interface area
- **Grid-based data**: Demonstrates square grid handling capabilities
- **No theoretical dimension**: Tests analysis methods without known "correct" answer

## Key Concepts Demonstrated

### 1. **Real-World Data Handling**
- Auto-detection of grid resolution from file headers
- Handling square grids with standard aspect ratios
- Physics-based parameter estimation for fluid interfaces
- Data validation and quality assessment

### 2. **Complete Workflow**
- Data loading and validation
- Grid configuration analysis
- Parameter optimization for fluid interfaces
- Publication-quality visualization

### 3. **Square Grid Analysis**
- Standard square grid analysis
- Physics-based minimum box size estimation
- Grid optimization for fluid interfaces

## Input Data Format

### File Structure
```
# Interface data for t = 19.600000
# Method: dalziel
# Grid: 400×400
# ANALYSIS VALIDITY SUMMARY
# Overall Status: VALID
# Cells across mixing layer: 399.0
0.0030980,0.0162500 0.0049211,0.0125789
0.0000117,0.0174883 0.0037500,0.0155936
...
```

### Header Information
- **Time stamp**: Simulation time for interface
- **Detection method**: Algorithm used to extract interface
- **Grid dimensions**: Computational grid size (auto-detected)
- **Validation status**: Quality assessment from simulation

### Data Format
- **Line segments**: One per line in format `x1,y1 x2,y2`
- **Coordinates**: Physical coordinates (not grid indices)
- **Units**: Typically non-dimensional simulation units

## Usage Examples

### Basic RT Analysis
```bash
# Direct analysis with fractal_analyzer
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --plot_separate

# With grid optimization
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --use_grid_optimization

# Generate sample data and analyze
python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region
```

### Advanced Analysis
```bash
# Custom box counting parameters
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --box_size_factor 1.4 --trim_boundary 1

# Publication-quality output
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --eps_plots --no_titles --plot_separate

# Parameter sensitivity testing
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --box_size_factor 1.2
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --box_size_factor 1.6
```

### Square Grid Analysis
```bash
# Different square grid sizes
python fractal_analyzer.py --file RT200x200_interface.txt --analyze_linear_region
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region
python fractal_analyzer.py --file RT800x800_interface.txt --analyze_linear_region
```

## Expected Results

### Typical RT Interface Dimensions
- **Early time** (t < 10): D ≈ 1.1-1.3 (relatively smooth)
- **Intermediate time** (t = 10-20): D ≈ 1.3-1.6 (developing complexity)
- **Late time** (t > 20): D ≈ 1.5-1.8 (fully developed mixing)

### Grid Configuration Effects
- **Square grids**: Isotropic results
- **Higher resolution**: Better captures fine structure
- **Lower resolution**: May miss small-scale features

### Quality Indicators
- **R² values**: Should be > 0.98 for good linear scaling
- **Grid optimization**: Usually provides 0.01-0.05 improvement
- **Physical range**: Dimensions should be 1.0 ≤ D ≤ 2.0

## Output Files

### Plots Generated
- `fractal_curve.png/eps` - Interface geometry
- `box_counting_loglog.png/eps` - Box counting analysis
- `linear_region_analysis.png/eps` - Optimization analysis (if using combined plots)
- `sliding_window_analysis.png/eps` - Window optimization (if using separate plots)

### Console Output
```
==== ANALYZING LINEAR REGION SELECTION ====

Physics-based auto-estimated min_box_size: 0.00400000
Final box size range: 0.00400000 to 0.20000000
Expected number of box sizes: 15

Box counting with grid optimization:
  Box size  |  Min count |  Grid tests  | Improv | Time (s)
-----------------------------------------------------------
  0.200000  |      147  |          4   |   8.1% |   0.15
  0.133333  |      198  |          9   |  12.2% |   0.18
  ...

Results:
Optimal scaling region: points 5 to 10
Box size range: 0.01333300 to 0.00592593
Fractal Dimension: 1.456789 ± 0.012345
R-squared: 0.998765
```

## Understanding the Results

### Physical Interpretation

#### **Dimension Ranges**
- **D < 1.2**: Very smooth interface, minimal mixing
- **1.2 ≤ D < 1.5**: Moderate complexity, developing turbulence
- **1.5 ≤ D < 1.8**: High complexity, active mixing
- **D ≥ 1.8**: Very complex, advanced turbulent mixing

#### **Time Evolution**
RT interfaces typically show:
1. **Initial growth**: Smooth, low dimension
2. **Linear growth**: Increasing complexity
3. **Nonlinear mixing**: High dimension, plateauing

### Grid Effects

#### **Square Grid Considerations**
- **Low resolution** (< 200×200): May miss fine structure
- **Medium resolution** (200-600): Good balance
- **High resolution** (> 600): Captures fine details but computationally expensive

### Quality Assessment

#### ✅ **Good Analysis Indicators**
- R² > 0.98
- Dimension in physical range (1.0-2.0)
- Grid optimization shows improvement
- Sufficient interface segments (> 100)

#### ⚠️ **Warning Signs**
- R² < 0.95
- Dimension outside physical range
- Very few interface segments (< 50)

#### ❌ **Problem Indicators**
- R² < 0.9
- Dimension < 1.0 or > 2.0
- Analysis fails to complete

## Parameter Sensitivity

### Testing Different Parameters
```bash
# Test different box size factors
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --box_size_factor 1.2
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --box_size_factor 1.4
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --box_size_factor 1.6

# Test boundary trimming
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --trim_boundary 0
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --trim_boundary 1
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --trim_boundary 2
```

### Understanding Sensitivity
- **Box size factor**: Controls scaling between box sizes
- **Boundary trimming**: Removes potential edge effects
- **Grid optimization**: Minimizes box positioning artifacts

## Best Practices

### For RT Interface Analysis
1. **Use square grid data** for consistent results
2. **Apply physics-based parameter estimation**
3. **Verify dimensions are physically reasonable**
4. **Compare with known scaling regimes**
5. **Use grid optimization** for improved accuracy

### For Publication
1. **Use `--eps_plots --no_titles`** for journal submission
2. **Use `--plot_separate`** for individual publication plots
3. **Document parameter choices**
4. **Provide physical interpretation**
5. **Include R² values** for quality assessment

## Troubleshooting

### Common Issues

#### **Auto-detection Fails**
```bash
# Use square grid data with detectable naming
# Rename file to RT400x400_interface.txt format
```

#### **Poor R² Values**
- Check interface quality
- Adjust box size range with `--min_box_size` and `--max_box_size`
- Verify adequate segment count

#### **Unphysical Dimensions**
- Check input data quality
- Verify coordinate scaling
- Adjust analysis parameters

### File Format Issues
- Ensure segments are in `x1,y1 x2,y2` format
- Check for proper header format
- Verify square grid dimensions in filename

## Integration with Simulation Workflow

### Typical CFD Analysis Pipeline
1. **Run RT simulation** → interface data files
2. **Extract interfaces** → segment files with headers
3. **Fractal analysis** → dimension measurements
4. **Physical interpretation** → mixing characterization

### Batch Processing
```bash
# Process multiple time snapshots
for file in RT400x400_t*.txt; do
    python fractal_analyzer.py --file "$file" --analyze_linear_region --no_plot
done
```

## Sample Data Generation

### Creating Test Data
```bash
# Generate Koch curve for algorithm validation
python fractal_analyzer.py --generate koch --level 5 --analyze_linear_region

# Generate Sierpinski triangle
python fractal_analyzer.py --generate sierpinski --level 4 --analyze_linear_region

# Generate Dragon curve
python fractal_analyzer.py --generate dragon --level 6 --analyze_linear_region
```

## Advanced Features

### Grid Optimization
```bash
# With grid optimization (recommended)
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --use_grid_optimization

# Without grid optimization (for comparison)
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --disable_grid_optimization
```

### Custom Analysis Parameters
```bash
# Custom box size range
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --min_box_size 0.001 --max_box_size 0.1

# Custom box size factor
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --box_size_factor 1.3

# Boundary trimming
python fractal_analyzer.py --file RT400x400_interface.txt --analyze_linear_region --trim_boundary 2
```

## References

1. Rayleigh, Lord (1883). "Investigation of the character of the equilibrium of an incompressible heavy fluid of variable density". *Proceedings of the London Mathematical Society*.
2. Taylor, G.I. (1950). "The instability of liquid surfaces when accelerated in a direction perpendicular to their planes". *Proceedings of the Royal Society A*.
3. Dalziel, S.B. (1993). "Rayleigh-Taylor instability: experiments with image analysis". *Dynamics of Atmospheres and Oceans*.

## Next Steps

After RT interface analysis:
1. **Compare dimensions across time** for evolution studies
2. **Correlate with mixing metrics** (e.g., mixing efficiency)
3. **Validate against experimental data** when available
4. **Apply to other fluid interfaces** (Kelvin-Helmholtz, etc.)

---

💡 **Pro Tip**: RT interfaces provide excellent validation for fractal algorithms since they're geometrically complex but physically constrained to reasonable dimension ranges!
