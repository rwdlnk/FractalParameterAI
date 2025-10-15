#!/usr/bin/env python3
"""
Manual fracture network analysis with explicitly controlled parameters.
Bypasses AI learning to test with known-good parameters.
"""

import sys
import os
import time
import numpy as np

# Add paths
FAST_FRACTAL_PATH = '/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer'
if FAST_FRACTAL_PATH not in sys.path:
    sys.path.insert(0, FAST_FRACTAL_PATH)

from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.performance_config import PerformanceMode

def main():
    print("🔬 MANUAL FRACTURE NETWORK ANALYSIS")
    print("=" * 60)

    data_file = 'synthetic_fractures.txt'

    print(f"📂 Loading: {data_file}")
    segments = SegmentArray.from_file(data_file)

    # Analyze data properties
    lengths = np.sqrt((segments.segments[:, 1, 0] - segments.segments[:, 0, 0])**2 +
                      (segments.segments[:, 1, 1] - segments.segments[:, 0, 1])**2)

    print(f"✅ Loaded: {segments.n_segments} segments")
    print(f"📏 Domain: {segments.bbox.width:.6f} × {segments.bbox.height:.6f}")
    print(f"📊 Total length: {np.sum(lengths):.3f}")
    print(f"📊 Mean segment: {np.mean(lengths):.6f}")
    print(f"📊 Range: [{np.min(lengths):.6f}, {np.max(lengths):.6f}]")
    print()

    # Test multiple parameter sets
    analyzer = FastFractalAnalyzer(performance_mode=PerformanceMode.FAST)

    test_params = [
        {
            'name': 'Conservative (start at domain/2)',
            'initial_delta': 0.5,
            'delta_factor': 1.5,
            'num_steps': 10
        },
        {
            'name': 'Medium (start at domain/3)',
            'initial_delta': 0.33,
            'delta_factor': 1.6,
            'num_steps': 12
        },
        {
            'name': 'Fine (start at domain/5)',
            'initial_delta': 0.2,
            'delta_factor': 1.7,
            'num_steps': 12
        },
        {
            'name': 'Very Fine (start at domain/10)',
            'initial_delta': 0.1,
            'delta_factor': 1.8,
            'num_steps': 15
        }
    ]

    print("🧪 PARAMETER TESTING")
    print("=" * 60)

    for params in test_params:
        print(f"\n{params['name']}:")
        print(f"  δ₀={params['initial_delta']:.3f}, factor={params['delta_factor']:.1f}, steps={params['num_steps']}")

        start_time = time.time()
        result = analyzer.analyze_segments(
            segments,
            initial_delta=params['initial_delta'],
            delta_factor=params['delta_factor'],
            num_steps=params['num_steps']
        )
        elapsed = time.time() - start_time

        if result and hasattr(result, 'dimension'):
            print(f"  ✅ D = {result.dimension:.6f}, R² = {result.r_squared:.6f}, t = {elapsed:.1f}s")

            # Interpretation
            if 1.2 <= result.dimension <= 1.6 and result.r_squared > 0.95:
                print(f"  🎯 IN EXPECTED RANGE (1.3-1.5)!")
            elif result.dimension > 2.0:
                print(f"  ⚠️  UNPHYSICAL: D > 2.0 for 2D embedding")
            elif result.dimension < 1.0:
                print(f"  ⚠️  UNPHYSICAL: D < 1.0 for line segments")
            elif result.r_squared < 0.95:
                print(f"  ⚠️  LOW CONFIDENCE: R² < 0.95")
        else:
            print(f"  ❌ Analysis failed")

    print("\n" + "=" * 60)
    print("📝 NOTES:")
    print("  • Expected D for this network: 1.3 - 1.5")
    print("  • initial_delta must be < domain size")
    print("  • Need good range of box sizes for accurate slope")

if __name__ == "__main__":
    main()
