#!/usr/bin/env python3
"""
Setup script for Fractal Dimension Analysis Tool
Supporting material for publication on fractal analysis methods
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fractal-dimension-analyzer",
    version="1.0.0",
    author="[Rod Douglass]",  # Replace with actual author
    author_email="[rwdlanm@gmail.com]",  # Replace with actual email
    description="Advanced fractal dimension analysis with grid optimization for scientific applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/rwdlnk/Fractal-Dimension-Analyzer",  # Replace with actual URL
    
    # Package information
    py_modules=["fractal_analyzer"],
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements("requirements_minimal.txt"),
    extras_require={
        "full": read_requirements("requirements.txt"),
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    
    # Entry points for command-line usage
    entry_points={
        "console_scripts": [
            "fractal-analyzer=fractal_analyzer:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for discovery
    keywords="fractal dimension box-counting computational-fluid-dynamics scientific-computing",
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["README.md", "requirements*.txt", "LICENSE"],
    },
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/rwdlnk/Fractal-Dimension-Analyzer#readme",
        "Source": "https://github.com/rwdlnk/Fractal-Dimension-Analyzer",
        "Tracker": "https://github.com/rwdlnk/Fractal-Dimension-Analyzer/issues",
    },
)
