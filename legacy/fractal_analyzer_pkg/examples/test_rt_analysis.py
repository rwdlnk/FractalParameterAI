#!/usr/bin/env python3

"""
Test script for RTI analysis
"""

from fractal_analyzer.applications.rt import RTAnalyzer

# Check if FractalAnalysisTools exists
print("\nChecking for FractalAnalysisTools...")
try:
    from fractal_analyzer.analysis_tools import FractalAnalysisTools
    print("FractalAnalysisTools class exists in the package.")
except ImportError as e:
    print(f"Error importing FractalAnalysisTools: {str(e)}")
    print("This is why linear region analysis isn't being used!")
# Examine the implementation
print("\n--- Examining FractalAnalysisTools Implementation ---")
try:
    from fractal_analyzer.analysis_tools import FractalAnalysisTools
    
    # Print the source code if possible
    import inspect
    src = inspect.getsource(FractalAnalysisTools.analyze_linear_region)
    print("analyze_linear_region source code:")
    print(src)
except Exception as e:
    print(f"Error examining FractalAnalysisTools: {str(e)}")

# Create analyzer
analyzer = RTAnalyzer("./test_output")

# Specify VTK file
vtk_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/svof/build-Release/test_data/fracConv/800x800/RT800x800-7999.vtk"

# Analyze with linear region analysis
print("Analyzing with linear region analysis...")
result = analyzer.analyze_vtk_file(vtk_file, analyze_linear=True, trim_boundary=1)
print(f"Fractal dimension (linear region): {result['fractal_dim']:.6f} ± {result['fd_error']:.6f}")

# Analyze without linear region analysis
print("\nAnalyzing without linear region analysis...")
result = analyzer.analyze_vtk_file(vtk_file, analyze_linear=False)
print(f"Fractal dimension (standard): {result['fractal_dim']:.6f} ± {result['fd_error']:.6f}")
