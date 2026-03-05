from setuptools import setup, find_packages

setup(
    name="fractal-analyzer",
    version="0.25.0",  # Increment version
    description="Universal tool for fractal dimension analysis with RT application",
    author="Rod Douglass",
    author_email="rwdlanm@gmail.com",
    packages=["fractal_analyzer", "rt_analyzer", "fractal_analyzer.generators"],
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "numba",
        "pandas",
        "scikit-image",  # For RT contour extraction
    ],
    entry_points={
        'console_scripts': [
            'fractal-analyzer=fractal_analyzer.cli:cli',
            'rt-analyzer=rt_analyzer.cli:main',
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
