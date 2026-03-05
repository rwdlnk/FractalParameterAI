from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

# Create analyzer and tools

fractal = 'koch'
analyzer = FractalAnalyzer(fractal)
analysis = FractalAnalysisTools(analyzer)

iter = 5
# Generate fractal
_, segments = analyzer.generate_fractal(fractal, iter)

# Analyze optimal linear region for dimension calculation
windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis.analyze_linear_region(
    segments, fractal, plot_results=True, plot_boxes=True)

print(f"Optimal dimension ({fractal}: {iter} iterations) : {optimal_dimension:.6f}")


