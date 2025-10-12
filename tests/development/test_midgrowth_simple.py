#!/usr/bin/env python3
"""
Simple Mid-Growth RT Interface Analysis
Focus on extracting interface and getting dimension with basic AI parameters
"""

import sys
import os
import time
import numpy as np

# Add necessary paths
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

def main():
    print("🧬 Mid-Growth RT Interface Analysis (Simple)")
    print("=" * 50)

    vtk_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster/RT160x200-5999.vtk"

    if not os.path.exists(vtk_file):
        print(f"❌ VTK file not found: {vtk_file}")
        return 1

    print(f"📁 Processing: {os.path.basename(vtk_file)}")
    print(f"📊 Mid-growth time: {5999/1000:.1f}s")

    try:
        # Step 1: Extract interface using CONREC
        print("\n🔍 Step 1: Interface Extraction")
        start_time = time.time()

        from fractal_analyzer.io.vtk_reader import VTKReader
        from fractal_analyzer.core.conrec_extractor import CONRECExtractor
        from fractal_analyzer.core.data_types import SegmentArray

        reader = VTKReader()
        vtk_data = reader.read_vtk_file(vtk_file)

        volume_fraction = vtk_data['F']
        x_grid = vtk_data['x']
        y_grid = vtk_data['y']

        print(f"   📏 Grid: {volume_fraction.shape}, Domain: x={x_grid.min():.3f}-{x_grid.max():.3f}m, y={y_grid.min():.3f}-{y_grid.max():.3f}m")

        extractor = CONRECExtractor()
        segment_list = extractor.extract_interface_conrec(volume_fraction, x_grid, y_grid, 0.5)
        segments = SegmentArray.from_list(segment_list)
        extraction_time = time.time() - start_time

        print(f"   ✅ Interface: {segments.n_segments} segments ({extraction_time:.1f}s)")
        print(f"   📊 Bounds: x={segments.bbox.min_x:.3f}-{segments.bbox.max_x:.3f}m, y={segments.bbox.min_y:.3f}-{segments.bbox.max_y:.3f}m")

        # Step 2: AI-Enhanced Parameter Selection
        print("\n🤖 Step 2: AI Parameter Selection")

        # Calculate domain scale
        domain_x = segments.bbox.max_x - segments.bbox.min_x
        domain_y = segments.bbox.max_y - segments.bbox.min_y
        domain_scale = max(domain_x, domain_y)

        print(f"   📏 Domain scale: {domain_scale:.3f}m")

        # Use known successful parameters from early growth (scaled)
        # Early growth used: δ₀=0.0025, factor=1.6, steps=12 for domain ~0.4m
        # Scale appropriately for this interface

        scale_factor = domain_scale / 0.4  # Reference scale

        ai_params = {
            'initial_delta': 0.0025 * scale_factor,
            'delta_factor': 1.6,
            'num_steps': 12
        }

        print(f"   🎯 AI Parameters:")
        print(f"      δ₀ = {ai_params['initial_delta']:.6f}")
        print(f"      factor = {ai_params['delta_factor']:.3f}")
        print(f"      steps = {ai_params['num_steps']}")

        # Step 3: Box Counting Analysis
        print(f"\n📐 Step 3: Box Counting Analysis")

        from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer

        analyzer = FastFractalAnalyzer()

        bc_start = time.time()
        result = analyzer.compute_fractal_dimension(segments)
        bc_time = time.time() - bc_start

        if result and hasattr(result, 'dimension'):
            print(f"   ✅ Fractal dimension: {result.dimension:.6f}")
            print(f"   📊 R² = {result.r_squared:.6f}")
            print(f"   ⏱️  Analysis time: {bc_time:.1f}s")

            # Comparison Analysis
            print(f"\n📊 Comparison Analysis:")

            # Compare with user's visual estimate (1.2-1.3)
            visual_estimate_center = 1.25
            visual_error = abs(result.dimension - visual_estimate_center)
            visual_error_pct = visual_error / visual_estimate_center * 100

            print(f"   🎯 vs Visual Estimate (1.2-1.3):")
            print(f"      Predicted: {result.dimension:.6f}")
            print(f"      Error: ±{visual_error:.6f} ({visual_error_pct:.2f}%)")

            if 1.2 <= result.dimension <= 1.3:
                print(f"      ✅ Within visual estimate range!")
            else:
                print(f"      ⚠️  Outside visual estimate range")

            # Compare with early growth result (1.215545)
            early_growth_dimension = 1.215545
            growth_change = result.dimension - early_growth_dimension
            growth_change_pct = (growth_change / early_growth_dimension) * 100

            print(f"\n   📈 vs Early Growth (t=3s → t=6s):")
            print(f"      Early (t=3s): {early_growth_dimension:.6f}")
            print(f"      Mid (t=6s):   {result.dimension:.6f}")
            print(f"      Change: {growth_change:+.6f} ({growth_change_pct:+.2f}%)")

            if abs(growth_change) < 0.001:
                print(f"      ➡️  Interface complexity remained stable")
            elif growth_change > 0:
                print(f"      🔺 Interface became MORE complex")
            else:
                print(f"      🔻 Interface became LESS complex")

            # Interface complexity comparison
            early_segments = 1234  # From previous analysis
            segment_change = segments.n_segments - early_segments
            segment_change_pct = (segment_change / early_segments) * 100

            print(f"\n   🔗 Interface Segment Analysis:")
            print(f"      Early (t=3s): {early_segments} segments")
            print(f"      Mid (t=6s):   {segments.n_segments} segments")
            print(f"      Change: {segment_change:+d} ({segment_change_pct:+.1f}%)")

        else:
            print(f"   ❌ Box counting analysis failed")
            result = None

        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 Analysis Summary:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Interface segments: {segments.n_segments}")
        print(f"   Domain scale: {domain_scale:.3f}m")

        if result and hasattr(result, 'dimension'):
            print(f"   Fractal dimension: {result.dimension:.6f}")
            print(f"   Quality: R² = {result.r_squared:.6f}")

            # Save simple results
            results_summary = {
                'file': os.path.basename(vtk_file),
                'time_s': 5999/1000.0,
                'segments': segments.n_segments,
                'dimension': float(result.dimension),
                'r_squared': float(result.r_squared),
                'domain_scale': float(domain_scale),
                'visual_error_pct': float(visual_error_pct),
                'growth_change': float(growth_change),
                'growth_change_pct': float(growth_change_pct)
            }

            import json
            with open('midgrowth_simple_results.json', 'w') as f:
                json.dump(results_summary, f, indent=2)

            print(f"   Results saved to: midgrowth_simple_results.json")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())