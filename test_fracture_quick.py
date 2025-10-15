#!/usr/bin/env python3
"""Quick test of fracture analysis without rotation"""
import sys
sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.performance_config import PerformanceMode
from core.interface_features import extract_interface_features, suggest_optimal_parameters_adaptive

# Load data
segments = SegmentArray.from_file('experiments/geologic_data/synthetic_fractures/synthetic_fractures.txt')

# Extract features
import numpy as np
segments_array = np.column_stack([
    segments.segments[:, 0, 0], segments.segments[:, 0, 1],
    segments.segments[:, 1, 0], segments.segments[:, 1, 1]
])
features = extract_interface_features(segments_array)

# Get AI suggestions (without rotation for speed)
params = suggest_optimal_parameters_adaptive(features, implementation="basic_box_counting")

print(f"Domain: {features['bbox_width']:.3f} × {features['bbox_height']:.3f}")
print(f"Char length: {features['characteristic_length']:.3f}")
print(f"AI suggested initial_delta: {params.get('initial_delta', 'N/A')}")

# Test WITHOUT rotation for speed
analyzer = FastFractalAnalyzer(performance_mode=PerformanceMode.FAST)
result = analyzer.analyze_segments(
    segments,
    initial_delta=params['initial_delta'],
    delta_factor=params['delta_factor'],
    num_steps=int(params['num_steps'])
)

if result:
    print(f"\nDimension: {result.dimension:.6f}")
    print(f"R²: {result.r_squared:.6f}")

    if 1.3 <= result.dimension <= 1.5:
        print("✅ IN EXPECTED RANGE (1.3-1.5)!")
    elif result.dimension > 2.0:
        print(f"❌ UNPHYSICAL: D > 2.0")
    elif result.dimension < 1.0:
        print(f"❌ UNPHYSICAL: D < 1.0")
