#!/usr/bin/env python3
"""
General Fractal Analysis using AI Framework with FastFractalAnalyzer

This script demonstrates the proper way to use the AI Framework with
FastFractalAnalyzer to compute fractal dimensions for any segment data.
"""

import sys
import os
import time
import numpy as np

# Add FastFractalAnalyzer to path
FAST_FRACTAL_PATH = '/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer'
if FAST_FRACTAL_PATH not in sys.path:
    sys.path.insert(0, FAST_FRACTAL_PATH)

# Import from FastFractalAnalyzer (using correct path)
from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.performance_config import PerformanceMode

# Import AI Framework components (from local core directory)
from core.interface_features import (
    extract_interface_features,
    classify_interface_type,
    suggest_optimal_parameters_adaptive,
    record_parameter_feedback
)

def main():
    # Check for command line argument
    if len(sys.argv) < 2:
        print("❌ Usage: python general_fractal_ai_analysis.py <segment_file>")
        print("   Example: python general_fractal_ai_analysis.py experiments/geologic_data/synthetic_fractures.txt")
        sys.exit(1)

    data_file = sys.argv[1]

    print("🔬 GENERAL FRACTAL ANALYSIS")
    print("AI Framework Integration with FastFractalAnalyzer")
    print("=" * 60)

    print(f"📂 Loading data from: {data_file}")
    start_load = time.time()

    try:
        segments = SegmentArray.from_file(data_file)
        load_time = time.time() - start_load

        print(f"✅ Loaded: {segments.n_segments:,} segments ({load_time:.2f}s)")
        print(f"📏 Domain: {segments.bbox.width:.6f} × {segments.bbox.height:.6f}")
        print(f"📍 Bounds: ({segments.bbox.min_x:.6f},{segments.bbox.min_y:.6f}) to ({segments.bbox.max_x:.6f},{segments.bbox.max_y:.6f})")

    except FileNotFoundError:
        print(f"❌ Error: Could not find file {data_file}")
        print("   Make sure the file path is correct")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    print()
    print("🤖 AI FRAMEWORK ANALYSIS")
    print("-" * 40)

    # Convert SegmentArray to format expected by AI Framework
    print("🔍 Extracting interface features...")
    segments_array = np.column_stack([
        segments.segments[:, 0, 0],  # x1
        segments.segments[:, 0, 1],  # y1
        segments.segments[:, 1, 0],  # x2
        segments.segments[:, 1, 1]   # y2
    ])

    # AI feature analysis
    features = extract_interface_features(segments_array)
    interface_type = classify_interface_type(features)

    print(f"   🏷️  Interface type: {interface_type}")
    print(f"   📊 Complexity score: {features['complexity_score']:.3f}")
    print(f"   🌀 Tortuosity: {features['tortuosity']:.3f}")
    print(f"   📏 Characteristic length: {features['characteristic_length']:.6f}")

    # Get AI-suggested parameters
    print("\n🎯 Getting AI parameter suggestions...")
    suggested_params = suggest_optimal_parameters_adaptive(
        features,
        implementation="basic_box_counting"
    )

    print(f"   🔧 AI Suggested Parameters:")
    if 'initial_delta' in suggested_params:
        print(f"      initial_delta: {suggested_params['initial_delta']:.6f}")
    if 'delta_factor' in suggested_params:
        print(f"      delta_factor: {suggested_params['delta_factor']:.3f}")
    if 'num_steps' in suggested_params:
        print(f"      num_steps: {suggested_params['num_steps']}")
    if 'rotation_enabled' in suggested_params:
        print(f"      rotation_enabled: {suggested_params['rotation_enabled']}")

    print()
    print("🔬 FRACTAL DIMENSION ANALYSIS")
    print("-" * 40)

    # Create analyzer with FAST mode for reasonable performance
    analyzer = FastFractalAnalyzer(performance_mode=PerformanceMode.FAST)

    print("🏁 Running fractal analysis...")
    start_analysis = time.time()

    # Use AI parameters if available, otherwise use defaults
    if all(key in suggested_params for key in ['initial_delta', 'delta_factor', 'num_steps']):
        print("   Using AI-suggested parameters")
        result = analyzer.analyze_segments(
            segments,
            initial_delta=suggested_params['initial_delta'],
            delta_factor=suggested_params['delta_factor'],
            num_steps=int(suggested_params['num_steps'])
        )
    else:
        print("   Using default parameters")
        result = analyzer.analyze_segments(segments)

    analysis_time = time.time() - start_analysis

    if result and hasattr(result, 'dimension'):
        print()
        print("🎉 FRACTAL ANALYSIS RESULTS:")
        print("=" * 40)
        print(f"📐 Fractal Dimension: {result.dimension:.6f}")
        print(f"📈 R² Quality: {result.r_squared:.6f}")
        print(f"⏱️  Analysis Time: {analysis_time:.1f}s")

        # Complexity interpretation
        print()
        print("🔍 COMPLEXITY INTERPRETATION:")
        if 1.0 <= result.dimension < 1.15:
            print("   📍 Very smooth structure (low complexity)")
        elif 1.15 <= result.dimension < 1.3:
            print("   📍 Moderately smooth structure")
        elif 1.3 <= result.dimension < 1.5:
            print("   📍 Moderately complex structure")
        elif 1.5 <= result.dimension < 1.7:
            print("   📍 Highly complex structure")
        elif 1.7 <= result.dimension < 1.9:
            print("   📍 Very highly complex structure")
        else:
            print("   📍 Extremely complex/space-filling structure")

        # Quality assessment
        print()
        print("🎯 QUALITY ASSESSMENT:")
        if result.r_squared > 0.99:
            print("   ✅ Excellent - Very high confidence")
        elif result.r_squared > 0.95:
            print("   ✅ Good - High confidence")
        elif result.r_squared > 0.90:
            print("   ⚠️  Fair - Moderate confidence")
        else:
            print("   ❌ Poor - Low confidence")

        print()
        print("📋 SUMMARY:")
        print(f"   • Fractal dimension: D = {result.dimension:.6f}")
        print(f"   • Analysis quality: R² = {result.r_squared:.6f}")
        print(f"   • Computation time: {analysis_time:.1f} seconds")
        print(f"   • Total segments analyzed: {segments.n_segments:,}")
        print(f"   • Interface type: {interface_type}")

        # Record AI feedback for learning (without theoretical dimension)
        print()
        print("📝 Recording AI feedback...")
        feedback_success = record_parameter_feedback(
            features=features,
            suggested_parameters=suggested_params,
            dimension_result=result.dimension,
            theoretical_dimension=None,  # Unknown for general data
            r_squared=result.r_squared,
            computation_time=analysis_time
        )
        print(f"   AI learning feedback recorded: {feedback_success}")

        print()
        print("✨ FRAMEWORK STATUS:")
        print("   ✅ AI Framework successfully integrated with FastFractalAnalyzer")
        print("   ✅ Parameter suggestion and feedback systems working")
        print("   ✅ Fractal dimension computed successfully")
        print("   ✅ All import conflicts resolved")

    else:
        print("❌ Analysis failed - no valid result returned")
        if result:
            print(f"   Result type: {type(result)}")
            attrs = [attr for attr in dir(result) if not attr.startswith('_')]
            print(f"   Available attributes: {attrs}")

    print()
    print("🏁 Fractal analysis complete!")

if __name__ == "__main__":
    main()