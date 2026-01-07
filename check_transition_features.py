#!/usr/bin/env python3
"""
Check interface features around the transition from straight to wavy (t=1.799 to t=1.999).
"""

import numpy as np
import sys
sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')

from integration.rt_vtk_parser import VTKParser
from integration.rt_interface_extraction import InterfaceExtractor
from core.interface_features import extract_interface_features

# Check files around the transition
files = [
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-999.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-1199.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-1399.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-1599.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-1799.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-1999.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-2199.vtk',
    '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-2399.vtk',
]

parser = VTKParser(debug=False)
extractor = InterfaceExtractor(method='conrec', debug=False)

print("="*95)
print("INTERFACE FEATURES AROUND TRANSITION (t=1.0 to t=2.4)")
print("="*95)
print(f"{'File':<25} {'Time':>8} {'Complexity':>12} {'Tortuosity':>12} {'Char Length':>14} {'Segments':>10} {'Status':>10}")
print("-"*95)

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

        # Determine if it would be skipped
        would_skip = (features['complexity_score'] < 0.5 and features['tortuosity'] < 1.03)
        status = "SKIP" if would_skip else "ANALYZE"

        # Print results
        basename = vtk_file.split('/')[-1]
        print(f"{basename:<25} {vtk_data.time:>8.3f} {features['complexity_score']:>12.6f} "
              f"{features['tortuosity']:>12.6f} {features['characteristic_length']:>14.6e} "
              f"{len(interface_data.segments):>10} {status:>10}")

    except Exception as e:
        print(f"Error processing {vtk_file}: {e}")

print("-"*95)
print("\nCurrent threshold: complexity < 0.5 AND tortuosity < 1.03")
print("Recommendation: Adjust threshold based on where visual transition occurs")
