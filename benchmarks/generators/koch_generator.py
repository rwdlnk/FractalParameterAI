#!/usr/bin/env python3
"""
Koch Curve Generator for Testing AI Parameter Selection
Generates Koch curves at different iterations for benchmarking.
"""

import numpy as np
import sys


def generate_koch_curve(p1, p2, iterations):
    """
    Generate Koch curve segments recursively.

    Args:
        p1: Starting point [x, y]
        p2: Ending point [x, y]
        iterations: Number of Koch iterations

    Returns:
        List of segments [[x1, y1, x2, y2], ...]
    """
    if iterations == 0:
        return [[p1[0], p1[1], p2[0], p2[1]]]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Divide into three equal parts
    p1_3 = [p1[0] + dx / 3, p1[1] + dy / 3]
    p2_3 = [p1[0] + 2 * dx / 3, p1[1] + 2 * dy / 3]

    # Calculate the peak point (equilateral triangle)
    angle = np.arctan2(dy, dx)
    length = np.sqrt(dx**2 + dy**2) / 3
    peak = [
        p1_3[0] + length * np.cos(angle + np.pi / 3),
        p1_3[1] + length * np.sin(angle + np.pi / 3)
    ]

    # Recursively generate 4 segments
    segments = []
    segments.extend(generate_koch_curve(p1, p1_3, iterations - 1))
    segments.extend(generate_koch_curve(p1_3, peak, iterations - 1))
    segments.extend(generate_koch_curve(peak, p2_3, iterations - 1))
    segments.extend(generate_koch_curve(p2_3, p2, iterations - 1))

    return segments


def save_koch_curve(segments, filename):
    """Save Koch curve segments to file."""
    with open(filename, 'w') as f:
        f.write("# Koch curve segments: x1 y1 x2 y2\n")
        f.write(f"# Number of segments: {len(segments)}\n")
        f.write(f"# Theoretical dimension: {np.log(4) / np.log(3):.6f}\n")
        f.write("#\n")

        for segment in segments:
            f.write(f"{segment[0]:.6f} {segment[1]:.6f} {segment[2]:.6f} {segment[3]:.6f}\n")


def main():
    """Generate Koch curves for testing."""

    print("="*60)
    print("KOCH CURVE GENERATOR")
    print("="*60)

    # Generate different iterations
    iterations_to_test = [1, 2, 3, 4]

    for iteration in iterations_to_test:
        print(f"\nGenerating Koch curve iteration {iteration}...")

        # Start with horizontal line from (0,0) to (3,0)
        p1 = [0.0, 0.0]
        p2 = [3.0, 0.0]

        segments = generate_koch_curve(p1, p2, iteration)

        # Save to file
        filename = f"/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI/benchmarks/data/koch_curves/koch_iteration_{iteration}.txt"
        save_koch_curve(segments, filename)

        theoretical_dim = np.log(4) / np.log(3)
        print(f"  Segments: {len(segments)}")
        print(f"  Theoretical dimension: {theoretical_dim:.6f}")
        print(f"  Saved to: {filename}")


if __name__ == "__main__":
    main()