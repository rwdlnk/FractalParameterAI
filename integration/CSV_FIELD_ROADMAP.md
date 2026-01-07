# CSV Field Expansion Roadmap

This document tracks the planned expansion of the temporal evolution CSV output as new analysis types are implemented in the refactored RT analyzer.

## Current Status: Phase 1 Complete (19 fields)

### Phase 1: Basic Fractal Analysis ✅
**Status:** Implemented
**Fields:** 19

| Field Name | Type | Description |
|------------|------|-------------|
| `time` | float | Simulation time |
| `file_path` | string | VTK filename |
| `grid_nx` | int | Grid size in x |
| `grid_ny` | int | Grid size in y |
| `interface_segments` | int | Number of interface segments |
| `interface_points` | int | Number of interface points |
| `fractal_dimension` | float | Box-counting fractal dimension |
| `fractal_r_squared` | float | Log-log fit quality |
| `fractal_interface_type` | string | AI-classified interface type |
| `fractal_analysis_time` | float | Fractal computation time (s) |
| `interface_extraction_time` | float | Interface extraction time (s) |
| `total_analysis_time` | float | Total analysis time (s) |
| `characteristic_length` | float | Interface characteristic length |
| `tortuosity` | float | Path tortuosity measure |
| `complexity_score` | float | Interface complexity metric |
| `connectivity_ratio` | float | Segment connectivity ratio |
| `initial_delta` | float | AI-suggested initial box size |
| `delta_factor` | float | AI-suggested scaling factor |
| `num_steps` | int | AI-suggested number of scales |

---

## Planned Expansions

### Phase 2: RT-Specific Classification (~6 fields)
**Status:** Planned
**Target Fields:** 25 total (+6)

Rayleigh-Taylor specific geometric analysis:

| Field Name | Type | Description |
|------------|------|-------------|
| `bubble_count` | int | Number of rising bubbles detected |
| `spike_count` | int | Number of falling spikes detected |
| `bubble_penetration_height` | float | Maximum bubble height |
| `spike_penetration_depth` | float | Maximum spike depth |
| `mixing_width` | float | h = h_bubble + h_spike |
| `growth_rate_alpha` | float | α from h = αAtg²t² |

**Implementation Notes:**
- Requires bubble/spike detection algorithm
- Needs top/bottom interface separation
- Growth rate α computed from temporal series

---

### Phase 3: Power Spectrum Analysis (~5 fields)
**Status:** Planned
**Target Fields:** 30 total (+5)

Spectral characteristics of interface roughness:

| Field Name | Type | Description |
|------------|------|-------------|
| `power_spectrum_slope` | float | Power law exponent β (E ∝ k^-β) |
| `power_spectrum_r_squared` | float | Power law fit quality |
| `power_spectrum_cutoff_wavelength` | float | Smallest resolved scale |
| `power_spectrum_energy` | float | Total spectral energy |
| `dominant_wavelength` | float | Peak wavelength in spectrum |

**Implementation Notes:**
- FFT of interface height function h(x)
- Log-log fit for power law exponent
- Identify dominant instability wavelength

---

### Phase 3: Multifractal Analysis (~7 fields)
**Status:** Planned
**Target Fields:** 37 total (+7)

Generalized dimensions and singularity spectrum:

| Field Name | Type | Description |
|------------|------|-------------|
| `multifractal_d0` | float | Capacity dimension (box-counting) |
| `multifractal_d1` | float | Information dimension (entropy) |
| `multifractal_d2` | float | Correlation dimension |
| `multifractal_alpha_min` | float | Minimum Hölder exponent |
| `multifractal_alpha_max` | float | Maximum Hölder exponent |
| `multifractal_width` | float | Spectrum width (α_max - α_min) |
| `multifractal_symmetry` | float | Asymmetry of f(α) spectrum |

**Implementation Notes:**
- Generalized box-counting for D_q spectrum
- Legendre transform to get f(α) spectrum
- Width indicates multifractal strength

---

### Phase 3: Mixing Analysis (~4 fields)
**Status:** Planned
**Target Fields:** 41 total (+4)

Scalar mixing metrics:

| Field Name | Type | Description |
|------------|------|-------------|
| `mixing_length` | float | Characteristic mixing scale |
| `mixing_efficiency` | float | η = ε_p / (ε_p + ε_k) |
| `mixed_fraction` | float | Fraction of domain that's mixed |
| `unmixed_fraction` | float | Fraction of domain unmixed |

**Implementation Notes:**
- Requires VOF field (f) analysis
- Mixed cells: 0.01 < f < 0.99
- May need scalar variance tracking

---

### Phase 3: RMS Velocity Components (~5 fields)
**Status:** Planned
**Target Fields:** 46 total (+5)

Turbulence statistics from velocity fields:

| Field Name | Type | Description |
|------------|------|-------------|
| `u_rms` | float | RMS horizontal velocity |
| `v_rms` | float | RMS vertical velocity |
| `velocity_magnitude_rms` | float | RMS total velocity magnitude |
| `turbulent_kinetic_energy` | float | TKE = 0.5(u'² + v'²) |
| `reynolds_stress` | float | Reynolds stress u'v' |

**Implementation Notes:**
- Requires velocity fields (u, v) in VTK
- Compute fluctuating components u', v'
- Spatial averaging for RMS values

---

## Final Target

**Total Fields:** ~50-60
**Complete Coverage:**
- ✅ Phase 1: Fractal dimension (19 fields)
- ⏳ Phase 2: RT classification (6 fields)
- ⏳ Phase 3: Power spectrum (5 fields)
- ⏳ Phase 3: Multifractal (7 fields)
- ⏳ Phase 3: Mixing (4 fields)
- ⏳ Phase 3: Velocity/turbulence (5 fields)
- 🔮 Future: Additional metrics as needed

## Implementation Strategy

1. **All new fields will be automatically added** as each analysis type is implemented
2. **CSV structure is forward-compatible** - `extrasaction='ignore'` handles missing fields
3. **JSON output always complete** - Contains full nested structure regardless of CSV
4. **Backward compatibility** - Old CSV files remain valid with subset of columns

## Usage Notes

- CSV is ideal for time-series plotting and spreadsheet analysis
- JSON preserves complete analysis results with full metadata
- Use pandas for advanced CSV analysis: `df = pd.read_csv('temporal_evolution.csv')`
- Filter columns of interest: `df[['time', 'fractal_dimension', 'mixing_efficiency']]`

---

**Last Updated:** 2025-10-15
**Maintained in:** `integration/rt_analyzer_refactored.py` (lines 466-653)
