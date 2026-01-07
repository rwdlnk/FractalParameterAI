#!/usr/bin/env python3
"""
Check interface features for early timestep files to determine appropriate thresholds.
"""

import numpy as np
import sys
sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')

from integration.rt_vtk_parser import VTKParser
from integration.rt_interface_extraction import InterfaceExtractor
from core.interface_features import extract_interface_features

# Check first few files
files = [
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-0.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-200.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-400.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-599.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-799.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-999.vtk',
]

parser = VTKParser(debug=False)
extractor = InterfaceExtractor(method='conrec', debug=False)

print("="*80)
print("INTERFACE FEATURES FOR EARLY TIMESTEPS")
print("="*80)
print(f"{'File':<25} {'Time':>8} {'Complexity':>12} {'Tortuosity':>12} {'Char Length':>14} {'Segments':>10}")
print("-"*80)

for vtk_file in files:
    try:
        # Parse and extract
        vtk_data = parser.parse_vtk_file(vtk_file)
        if not vtk_data:
            continue

        interface_data = extractor.extract_interface(
            vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid, level=0.5
        )
        if not interface_data:
            continue

        # Extract features
        features = extract_interface_features(interface_data.segments)

        # Print results
        basename = vtk_file.split('/')[-1]
        print(f"{basename:<25} {vtk_data.time:>8.3f} {features['complexity_score']:>12.6f} "
              f"{features['tortuosity']:>12.6f} {features['characteristic_length']:>14.6e} "
              f"{len(interface_data.segments):>10}")

    except Exception as e:
        print(f"Error processing {vtk_file}: {e}")

print("-"*80)
print("\nThreshold recommendation:")
print("  For complexity < X AND tortuosity < Y, skip box counting")
print("  Suggested: complexity < 0.5 AND tortuosity < 1.01")
