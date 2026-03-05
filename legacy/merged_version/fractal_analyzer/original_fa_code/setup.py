from setuptools import setup, find_packages

setup(
    name="fractal-analyzer",
    version="0.24.0",
    description="Universal tool for fractal dimension analysis using box counting method",
    author="Rod Douglass",
    author_email="rwdlanm@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "numba",
    ],
    entry_points={
        'console_scripts': [
            'fractal-analyzer=fractal_analyzer.cli:main',
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
