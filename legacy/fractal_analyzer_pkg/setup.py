"""
Setup script for the fractal_analyzer package.
"""

from setuptools import setup, find_packages

setup(
    name="fractal_analyzer",
    version="0.25.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "scipy>=1.4.0",
        "pandas>=1.0.0",
        "scikit-image>=0.16.0",
    ],
    extras_require={
        "rt": ["scikit-image>=0.16.0"],
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    entry_points={
        'console_scripts': [
            'fractal-analyzer=fractal_analyzer.cli:main',
            'fractal-rt=fractal_analyzer.applications.rt.rt_cli:main',
        ],
    },
    author="Rod Douglass",
    author_email="rwdlanm@gmail.com",
    description="A comprehensive fractal and multifractal analysis toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fractal_analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
)
