#!/usr/bin/env python3
"""
Setup configuration for FastFractalAnalyzer v3.
Completely rewritten for maximum performance with vectorized algorithms.
"""

from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

# Read the README file for long description
try:
    with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "FastFractalAnalyzer: High-performance fractal dimension analysis"

setup(
    name="fast-fractal-analyzer",
    version="3.0.0",
    author="Rod Douglass",
    author_email="rwdlanm@gmail.com",
    description="High-performance fractal dimension analysis with vectorized algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rwdlnk/FastFractalAnalyzer",
    project_urls={
        "Bug Tracker": "https://github.com/rwdlnk/FastFractalAnalyzer/issues",
        "Source Code": "https://github.com/rwdlnk/FastFractalAnalyzer",
    },

    packages=find_packages(exclude=["tests", "tests.*", "benchmarks", "benchmarks.*"]),

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=[
        "fractal", "dimension", "box-counting", "rayleigh-taylor",
        "vectorized", "numpy", "performance", "cfd", "analysis"
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "numba>=0.56.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
        ],
        "plotting": [
            "matplotlib>=3.5.0",
        ],
        "all": [
            "pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "ruff>=0.1.0",
            "matplotlib>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fast-fractal=fractal_analyzer.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)