#!/bin/bash

#python ../../fractal_analyzer-v25.py --generate dragon --level 6 --box_size_factor 1.5 --analyze_linear_region
#python ../../fractal_analyzer-v25.py --generate dragon --level 11 --box_size_factor 1.5 --analyze_linear_region --analyze_iterations --max_level 11
#python3 ~/Research/FractalAnalyzer/fractal_analyzer/core/fractal_analyzer.py --generate dragon --level 11 --box_size_factor 1.5 --analyze_linear_region --plot_separate --analyze_iterations --min_level 3 --max_level 11
#python3 ~/Research/FractalAnalyzer/fractal_analyzer/core/fractal_analyzer_fixed.py --generate dragon --level 13 \
#  --analyze_linear_region --plot_separate --analyze_iterations --min_level 5 --max_level 13
#  python3 ~/Research/FractalAnalyzer/fractal_analyzer/core/fractal_analyzer_fixed.py --generate koch --level 9 \
#  --analyze_linear_region --plot_separate --disable_grid_optimization --analyze_iterations --max_level 9
#  python3 ~/Research/FractalAnalyzer/fractal_analyzer/core/fractal_analyzer.py --generate koch --level 9 \
#  --analyze_linear_region --plot_separate --disable_grid_optimization --analyze_iterations --max_level 9
python3 ~/Research/FractalAnalyzer/fractal_analyzer/core/fractal_analyzer.py --generate koch --level 7 \
--plot_separate --no_title --eps_plots --disable_grid_optimization

