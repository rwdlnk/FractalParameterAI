#!/usr/bin/env python3
"""
Fixed Minkowski Diagnosis - Using proper API without grid_optimization parameter
"""

import sys
import time
import numpy as np
sys.path.append('.')

from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.box_counting import VectorizedBoxCounter
from fractal_analyzer.core.performance_config import PerformanceMode
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer

def diagnose_minkowski_performance():
    """Test Minkowski curve performance with fixed API usage."""
    print("🔍 MINKOWSKI PERFORMANCE DIAGNOSIS (FIXED)")
    print("=" * 60)

    # Test different levels
    test_levels = [4, 5]

    for level in test_levels:
        print(f"\n📊 Testing Minkowski L{level}")
        print("-" * 40)

        try:
            # Generate Minkowski curve
            analyzer = FastFractalAnalyzer()
            segments = analyzer._generate_minkowski_curve(level)

            print(f"Generated: {segments.n_segments:,} segments")
            print(f"Bbox: {segments.bbox.width:.3f} x {segments.bbox.height:.3f}")

            # Test different performance modes
            modes = [
                (PerformanceMode.FAST, "FAST"),
                (PerformanceMode.BALANCED, "BALANCED")
            ]

            for mode, name in modes:
                print(f"\n🧪 Testing {name} mode:")

                # Create analyzer with proper API
                test_analyzer = FastFractalAnalyzer(
                    performance_mode=mode,
                    boundary_artifact_removal=False,
                    min_r_squared=0.95,
                    min_scaling_decades=1.0
                )

                start_time = time.time()
                result = test_analyzer.analyze_segments(segments)
                elapsed_time = time.time() - start_time

                if result and hasattr(result, 'dimension'):
                    theoretical = 1.5
                    error_pct = abs(result.dimension - theoretical) / theoretical * 100

                    print(f"   Dimension: {result.dimension:.6f}")
                    print(f"   Error: {error_pct:.2f}%")
                    print(f"   R²: {result.r_squared:.6f}")
                    print(f"   Time: {elapsed_time:.1f}s")

                    if error_pct < 5.0:
                        print(f"   ✅ Good accuracy ({name})")
                    else:
                        print(f"   ⚠️  High error ({name})")
                else:
                    print(f"   ❌ Analysis failed ({name})")

        except Exception as e:
            print(f"❌ Error testing Minkowski L{level}: {e}")

    print("\n🎯 Minkowski diagnosis complete")

if __name__ == "__main__":
    diagnose_minkowski_performance()