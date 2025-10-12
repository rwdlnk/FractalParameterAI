#!/usr/bin/env python3
"""
Regenerate Enhanced Fractal Library with Complete Feature Set
Generates the 5 classic fractals with all advanced characterization measures:
- Koch curves (L1-L7)
- Dragon curves (L3-L12)
- Hilbert curves (L2-L8)
- Minkowski sausage (L1-L6)
- Sierpinski triangle (L3-L8)
"""

import sys
import os
import time
import json
import numpy as np
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from core.interface_features import extract_interface_features, classify_interface_type
from core.advanced_characterization import AdvancedCharacterizationEngine

def regenerate_fractal_with_enhanced_features(fractal_type, level, segments, theoretical_dim=None):
    """Generate complete feature set for a fractal including advanced characterization."""

    print(f"  📊 Analyzing {fractal_type} L{level} ({segments.n_segments:,} segments)")

    # Basic geometric features (23 measures)
    # Convert SegmentArray to numpy array format expected by extract_interface_features
    # SegmentArray format: (n_segments, 2, 2) -> [[x1,y1], [x2,y2]]
    # Target format: (n_segments, 4) -> [x1, y1, x2, y2]
    segments_array = np.column_stack([
        segments.segments[:, 0, 0],  # x1
        segments.segments[:, 0, 1],  # y1
        segments.segments[:, 1, 0],  # x2
        segments.segments[:, 1, 1]   # y2
    ])
    basic_features = extract_interface_features(segments_array)
    interface_type = classify_interface_type(basic_features)

    # Advanced characterization (4 methods with multiple features each)
    print(f"     🔬 Running advanced characterization...")

    # Initialize and run advanced characterization (reuse segments_array from above)
    engine = AdvancedCharacterizationEngine()
    advanced_results = engine.analyze_comprehensive(segments_array)

    # Combine all features
    enhanced_features = basic_features.copy()

    # Add advanced features from each method
    if 'feature_results' in advanced_results:
        for method_name, method_features in advanced_results['feature_results'].items():
            if method_features:
                for feature_name, feature_value in method_features.items():
                    enhanced_features[f"{method_name}_{feature_name}"] = feature_value

    # Create comprehensive record
    record = {
        'timestamp': time.time(),
        'fractal_type': fractal_type,
        'level': level,
        'theoretical_dimension': theoretical_dim,
        'interface_type': interface_type,
        'n_segments': segments.n_segments,
        'basic_features': basic_features,
        'enhanced_features': enhanced_features,
        'advanced_methods': {
            'consensus_parameters': advanced_results.get('consensus_parameters', {}),
            'individual_parameters': advanced_results.get('individual_parameters', {}),
            'confidences': advanced_results.get('confidences', {}),
            'analysis_times': advanced_results.get('analysis_times', {}),
            'total_analysis_time': advanced_results.get('total_analysis_time', 0.0)
        }
    }

    return record

def main():
    print("🔬 ENHANCED FRACTAL LIBRARY REGENERATION")
    print("=" * 60)
    print("Generating 5 classic fractals with complete advanced characterization")
    print("Features: Basic geometric (23) + Advanced methods (4 × multiple)")
    print()

    analyzer = FastFractalAnalyzer()
    all_records = []

    # 1. Koch Curves (L1-L7)
    print("🌟 1. Koch Curves")
    for level in range(1, 8):
        segments = analyzer._generate_koch_curve(level)
        record = regenerate_fractal_with_enhanced_features('koch', level, segments, 1.2619)
        all_records.append(record)

    # 2. Dragon Curves (L3-L12)
    print("\n🐉 2. Dragon Curves")
    for level in range(3, 13):
        segments = analyzer._generate_dragon_curve(level)
        record = regenerate_fractal_with_enhanced_features('dragon', level, segments, 2.0)
        all_records.append(record)

    # 3. Hilbert Curves (L2-L8)
    print("\n📐 3. Hilbert Curves")
    for level in range(2, 9):
        segments = analyzer._generate_hilbert_curve(level)
        record = regenerate_fractal_with_enhanced_features('hilbert', level, segments, 2.0)
        all_records.append(record)

    # 4. Minkowski Sausage (L1-L6)
    print("\n🌭 4. Minkowski Sausage")
    for level in range(1, 7):
        segments = analyzer._generate_minkowski_curve(level)
        record = regenerate_fractal_with_enhanced_features('minkowski', level, segments, 1.5)
        all_records.append(record)

    # 5. Sierpinski Triangle (L3-L8)
    print("\n🔺 5. Sierpinski Triangle")
    for level in range(3, 9):
        segments = analyzer._generate_sierpinski_triangle(level)
        record = regenerate_fractal_with_enhanced_features('sierpinski', level, segments, 1.585)
        all_records.append(record)

    # Save enhanced library
    output_file = 'enhanced_fractal_library.json'
    print(f"\n💾 Saving enhanced fractal library...")
    print(f"   📁 File: {output_file}")
    print(f"   📊 Records: {len(all_records)}")

    # Convert numpy types to JSON-serializable types
    serializable_records = convert_numpy_types(all_records)

    with open(output_file, 'w') as f:
        json.dump(serializable_records, f, indent=2)

    # Generate summary statistics
    print(f"\n📈 Enhanced Library Summary:")
    print(f"   • Total fractals: {len(all_records)}")
    print(f"   • Feature types: Basic geometric + Advanced characterization")
    print(f"   • Methods: Power spectrum, Wavelet, Curvature, Self-similarity")

    fractal_counts = {}
    for record in all_records:
        fractal_type = record['fractal_type']
        fractal_counts[fractal_type] = fractal_counts.get(fractal_type, 0) + 1

    for fractal_type, count in fractal_counts.items():
        print(f"   • {fractal_type.capitalize()}: {count} levels")

    print(f"\n✅ Enhanced fractal library regeneration complete!")
    print(f"This enhanced dataset provides comprehensive features for improved AI parameter prediction.")

if __name__ == "__main__":
    main()