
#!/bin/bash

# parameters are NOT default, but from FractalParameterAI run on the same data
python /media/rod/ResearchII_III/ResearchIII/githubRepos/FractalAnalyzer/fractal_analyzer/core/simple_analyzer/simple_analyzer.py synthetic_fractures.txt \
  --initial-delta 0.31 \
  --delta-factor 2.0 \
  --num-steps 10 

