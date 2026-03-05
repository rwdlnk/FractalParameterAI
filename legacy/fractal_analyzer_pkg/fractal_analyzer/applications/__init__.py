"""
Applications module for fractal_analyzer package.

This module contains specialized applications of the fractal_analyzer core functionality
for specific domains and use cases.
"""

# Import submodules explicitly to make them available at the package level
try:
    from . import rt
except ImportError:
    # RT module may not be available if dependencies are missing
    pass
