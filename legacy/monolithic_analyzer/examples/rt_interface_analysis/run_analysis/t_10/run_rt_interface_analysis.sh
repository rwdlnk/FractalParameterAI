#!/bin/bash

python3 /media/rod/ResearchII_III/ResearchIII/githubRepos/Fractal-Dimension-Analyzer/fractal_analyzer.py \
  --file ../../sample_data/RT160x200-10000.dat \
  --analyze_linear_region \
  --use_grid_optimization \
  --trim_boundary 2 \
  --no_titles \
  --plot_separate \
  --eps_plots
