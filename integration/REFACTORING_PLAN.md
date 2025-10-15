# RT Analyzer Full Refactoring Plan

## Objective
Fully integrate RT analyzer capabilities into the FractalParameterAI framework with AI-enhanced parameter selection for all analysis types.

## Current State (rt-analyzer-integration branch)
- ✅ Minimal wrapper around original rt_analyzer.py
- ✅ AI-enhanced fractal dimension analysis working
- ✅ Interface extraction via original rt_analyzer
- ✅ Feedback learning system integrated
- ✅ Tested on RT160x200-10000.vtk (D=1.664, R²=0.997)

## Target Capabilities

### 1. Interface Extraction (Standalone)
- [ ] Parse VTK files directly (no dependency on original rt_analyzer)
- [ ] Extract RT interface from density/concentration fields
- [ ] Support multiple VTK formats (structured/unstructured grids)
- [ ] Handle time-series VTK data

### 2. AI-Enhanced Fractal Analysis
- [x] Single-timestep fractal dimension (currently working)
- [ ] Temporal evolution analysis (dimension vs time)
- [ ] AI parameter adaptation per timestep
- [ ] Detect regime changes in temporal data

### 3. Power Spectrum Analysis
- [ ] 1D power spectrum along interface
- [ ] 2D power spectrum of interface height field
- [ ] AI-suggested frequency range selection
- [ ] Spectral slope estimation with confidence intervals

### 4. Multifractal Analysis
- [ ] Generalized dimension spectrum D(q)
- [ ] Singularity spectrum f(α)
- [ ] AI parameter selection for multifractal box-counting
- [ ] Learning from multifractal results

### 5. Mixing Analysis
- [ ] Mixing zone width measurement
- [ ] Molecular mixing fraction
- [ ] Reynolds number effects
- [ ] AI-optimized sampling parameters

### 6. Visualization & Reporting
- [ ] Interface plots with analysis overlays
- [ ] Temporal evolution plots
- [ ] Power spectrum plots
- [ ] Multifractal spectra visualization
- [ ] Comprehensive analysis reports

## Architecture Design

### Module Structure
```
integration/
├── rt_analyzer_ai.py          # Current minimal wrapper (preserved)
├── rt_analyzer_refactored.py  # New full implementation
├── rt_vtk_parser.py           # VTK parsing utilities
├── rt_interface_extraction.py # Interface detection algorithms
├── rt_temporal_analysis.py    # Time-series analysis
├── rt_power_spectrum.py       # Spectral analysis
├── rt_multifractal.py         # Multifractal analysis
├── rt_mixing_analysis.py      # Mixing zone analysis
└── rt_visualization.py        # Plotting and reporting
```

### Integration Points with AI Framework
1. **Feature extraction** (core/interface_features.py)
   - Extend for RT-specific features (mixing zone, spectral properties)

2. **Parameter selection** (core/feedback_system.py)
   - Add rt_interface type with RT-specific parameters
   - Learn optimal parameters for power spectrum, multifractal

3. **Feedback learning** (feedback_data.jsonl)
   - Record results from all analysis types
   - Build experience database for RT interfaces

### Data Flow
```
VTK File(s)
    ↓
Interface Extraction
    ↓
Feature Analysis (AI Framework)
    ↓
Interface Type Classification
    ↓
AI Parameter Selection
    ↓
Multiple Analysis Types:
  - Fractal Dimension
  - Power Spectrum
  - Multifractal
  - Mixing Metrics
    ↓
Feedback Recording
    ↓
Visualization & Reports
```

## Implementation Phases

### Phase 1: VTK Parsing (Foundation)
- [ ] Implement standalone VTK reader
- [ ] Extract structured grid data
- [ ] Identify interface from scalar fields
- [ ] Test on existing RT VTK files

### Phase 2: Extend Interface Classification
- [ ] Add `rt_interface` type to classifier
- [ ] Extract RT-specific features (mixing zone width, spectral properties)
- [ ] Update parameter suggestions for RT interfaces

### Phase 3: Temporal Analysis
- [ ] Multi-timestep fractal dimension tracking
- [ ] Dimension growth rate analysis
- [ ] Regime detection (early/intermediate/late time)
- [ ] AI parameter adaptation across time

### Phase 4: Power Spectrum
- [ ] Implement 1D/2D FFT-based spectrum
- [ ] AI-suggested frequency range
- [ ] Spectral slope fitting
- [ ] Learning from spectral results

### Phase 5: Multifractal Analysis
- [ ] Implement multifractal box-counting
- [ ] D(q) spectrum calculation
- [ ] f(α) singularity spectrum
- [ ] AI parameter selection for multifractal

### Phase 6: Mixing Analysis
- [ ] Mixing zone width detection
- [ ] Molecular mixing estimation
- [ ] Integration with other metrics

### Phase 7: Visualization & Polish
- [ ] Unified plotting system
- [ ] Report generation
- [ ] Documentation
- [ ] Examples and tutorials

## Success Criteria
- [ ] All analysis types working standalone (no original rt_analyzer dependency)
- [ ] AI parameter selection for each analysis type
- [ ] Temporal evolution analysis functional
- [ ] Results match or exceed original rt_analyzer quality
- [ ] Learning system improves results over time
- [ ] Comprehensive documentation

## Timeline
- This is research-quality refactoring, no rush
- Each phase can be developed and tested incrementally
- Preserve working wrapper during development
- Can cherry-pick features back to wrapper branch if needed

## Notes
- Original rt_analyzer.py location: `/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalAnalyzer/fractal_analyzer/core/rt_analyzer.py`
- Use as reference but don't modify original
- All new code in FractalParameterAI repo
- Maintain compatibility with existing FastFractalAnalyzer
