#!/bin/bash

python3 /media/rod/ResearchII_III/ResearchIII/githubRepos/Fractal-Dimension-Analyzer/fractal_analyzer.py \
  --generate koch --level 7 \
  --analyze_linear_region \
  --plot_separate --no_title \
  --analyze_iterations --max_level 7

