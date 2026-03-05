from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

# Create analyzer and tools
analyzer = FractalAnalyzer('koch')
analysis = FractalAnalysisTools(analyzer)

# Generate fractal
_, segments = analyzer.generate_fractal('koch', 5)

# Analyze optimal linear region for dimension calculation
windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis.analyze_linear_region(
    segments, 'koch', plot_results=True, plot_boxes=True)

print(f"Optimal dimension: {optimal_dimension:.6f}")
