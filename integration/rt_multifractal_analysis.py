#!/usr/bin/env python3
"""
RT Multifractal Analysis

Computes multifractal spectrum for RT interface structures:
- Generalized dimensions D_q
- Singularity spectrum f(α)
- Multifractal width and asymmetry
- Box-counting based analysis

This module is part of the FractalParameterAI framework.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import time


@dataclass
class MultifractalSpectrum:
    """Container for multifractal analysis results."""
    # Generalized dimensions
    d0: float  # Capacity dimension (box-counting dimension)
    d1: float  # Information dimension
    d2: float  # Correlation dimension
    d_inf: float  # D_∞ (minimum value of D_q)
    d_minus_inf: float  # D_-∞ (maximum value of D_q)

    # Singularity spectrum
    alpha_min: float  # Minimum singularity strength
    alpha_max: float  # Maximum singularity strength
    alpha_0: float  # α at f(α) maximum
    f_alpha_max: float  # Maximum f(α) value

    # Spectrum characteristics
    spectrum_width: float  # Δα = α_max - α_min
    spectrum_asymmetry: float  # (α_0 - α_min) / (α_max - α_min)

    # Full spectrum data
    q_values: np.ndarray  # Array of q values used
    d_q: np.ndarray  # D_q for each q
    alpha: np.ndarray  # α values for f(α) curve
    f_alpha: np.ndarray  # f(α) values

    # Quality metrics
    d_q_r_squared: np.ndarray  # R² for each D_q fit
    mean_r_squared: float  # Mean R² across all q

    # Analysis metadata
    n_scales: int  # Number of box sizes used
    analysis_time: float


class MultifractalAnalyzer:
    """
    Computes multifractal spectrum using box-counting method.

    The multifractal formalism:
    1. Partition space into boxes of size ε
    2. Compute probability p_i(ε) for each box
    3. Calculate partition function: Z_q(ε) = Σ p_i^q
    4. Compute τ(q) from scaling: Z_q(ε) ~ ε^τ(q)
    5. Generalized dimensions: D_q = τ(q)/(q-1)
    6. Singularity spectrum: α(q) = dτ/dq, f(α) = qα - τ
    """

    def __init__(self,
                 q_min: float = -5.0,
                 q_max: float = 5.0,
                 n_q: int = 21,
                 debug: bool = False):
        """
        Initialize multifractal analyzer.

        Args:
            q_min: Minimum q value
            q_max: Maximum q value
            n_q: Number of q values
            debug: Enable debug output
        """
        self.q_min = q_min
        self.q_max = q_max
        self.n_q = n_q
        self.debug = debug

        # Create q values array, avoiding q=1 (singularity in D_q formula)
        q_values = np.linspace(q_min, q_max, n_q)
        # Remove q values very close to 1
        self.q_values = q_values[np.abs(q_values - 1.0) > 0.1]

    def analyze_multifractal(self,
                            segments: np.ndarray,
                            initial_delta: float,
                            delta_factor: float = 1.5,
                            num_steps: int = 12) -> Optional[MultifractalSpectrum]:
        """
        Compute multifractal spectrum from interface segments.

        Args:
            segments: Interface segments as (N, 4) array [x1, y1, x2, y2]
            initial_delta: Initial box size
            delta_factor: Factor for box size progression
            num_steps: Number of box sizes

        Returns:
            MultifractalSpectrum object or None on failure
        """
        start_time = time.time()

        if self.debug:
            print(f"   📊 Computing multifractal spectrum...")
            print(f"      Segments: {len(segments)}")
            print(f"      Q range: [{self.q_min}, {self.q_max}] ({len(self.q_values)} values)")
            print(f"      Scales: {num_steps}")

        # Generate box sizes
        deltas = [initial_delta * (delta_factor ** i) for i in range(num_steps)]
        deltas = np.array(deltas)

        if self.debug:
            print(f"      Box sizes: {deltas[0]:.6f} to {deltas[-1]:.6f}")

        # ============================================
        # 1. Compute box counts and probabilities for each scale
        # ============================================
        box_data = []
        for delta in deltas:
            counts, total_length = self._count_boxes(segments, delta)
            if total_length > 0:
                probabilities = counts / total_length
                box_data.append({
                    'delta': delta,
                    'counts': counts,
                    'probabilities': probabilities,
                    'n_boxes': len(counts)
                })
            else:
                if self.debug:
                    print(f"      ⚠️  Zero total length at δ={delta:.6f}")

        if len(box_data) < 3:
            if self.debug:
                print(f"   ❌ Insufficient scales for multifractal analysis")
            return None

        # ============================================
        # 2. Compute partition function Z_q(ε) for each q
        # ============================================
        tau_q = []
        d_q = []
        tau_r_squared = []

        for q in self.q_values:
            # Compute Z_q for each scale
            z_q = []
            deltas_used = []

            for data in box_data:
                probs = data['probabilities']
                delta = data['delta']

                # Partition function: Z_q(ε) = Σ p_i^q
                if q != 0:
                    # For q ≠ 0: Z_q = Σ p_i^q
                    z = np.sum(probs ** q)
                else:
                    # For q = 0: Z_0 = number of non-empty boxes
                    z = np.sum(probs > 0)

                if z > 0:
                    z_q.append(z)
                    deltas_used.append(delta)

            if len(z_q) >= 3:
                # Fit τ(q) from Z_q(ε) ~ ε^τ(q)
                log_delta = np.log(deltas_used)
                log_z = np.log(z_q)

                # Linear regression: log(Z_q) = τ(q) * log(ε) + const
                coeffs = np.polyfit(log_delta, log_z, 1)
                tau = coeffs[0]

                # Compute R²
                predicted = coeffs[0] * log_delta + coeffs[1]
                ss_res = np.sum((log_z - predicted) ** 2)
                ss_tot = np.sum((log_z - np.mean(log_z)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                tau_q.append(tau)
                tau_r_squared.append(r_squared)

                # Generalized dimension: D_q = τ(q) / (q - 1)
                if abs(q - 1.0) > 0.01:
                    dq = tau / (q - 1.0)
                else:
                    # D_1 requires special handling (limit as q→1)
                    dq = np.nan
                d_q.append(dq)
            else:
                tau_q.append(np.nan)
                d_q.append(np.nan)
                tau_r_squared.append(0.0)

        tau_q = np.array(tau_q)
        d_q = np.array(d_q)
        tau_r_squared = np.array(tau_r_squared)

        # Remove NaN values
        valid_mask = ~np.isnan(d_q)
        q_valid = self.q_values[valid_mask]
        d_q_valid = d_q[valid_mask]
        tau_q_valid = tau_q[valid_mask]
        r_squared_valid = tau_r_squared[valid_mask]

        if len(d_q_valid) < 3:
            if self.debug:
                print(f"   ❌ Insufficient valid D_q values")
            return None

        # ============================================
        # 3. Extract key dimensions
        # ============================================
        # D_0: box-counting dimension (capacity dimension)
        d0_idx = np.argmin(np.abs(q_valid))
        d0 = d_q_valid[d0_idx]

        # D_1: information dimension (requires special computation)
        # For now, use linear interpolation near q=1
        if len(q_valid) > 1:
            # Find points bracketing q=1
            below_1 = q_valid < 1.0
            above_1 = q_valid > 1.0
            if np.any(below_1) and np.any(above_1):
                q_below = q_valid[below_1][-1]
                q_above = q_valid[above_1][0]
                d_below = d_q_valid[below_1][-1]
                d_above = d_q_valid[above_1][0]
                # Linear interpolation
                d1 = d_below + (1.0 - q_below) * (d_above - d_below) / (q_above - q_below)
            else:
                d1 = d0  # Fallback
        else:
            d1 = d0

        # D_2: correlation dimension
        d2_idx = np.argmin(np.abs(q_valid - 2.0))
        d2 = d_q_valid[d2_idx]

        # D_∞ and D_-∞
        d_inf = np.min(d_q_valid)
        d_minus_inf = np.max(d_q_valid)

        if self.debug:
            print(f"      D_0 = {d0:.4f}, D_1 = {d1:.4f}, D_2 = {d2:.4f}")

        # ============================================
        # 4. Compute singularity spectrum f(α)
        # ============================================
        # α(q) = dτ/dq
        # f(α) = q*α - τ

        # Compute derivative dτ/dq numerically
        alpha_values = np.gradient(tau_q_valid, q_valid)
        f_alpha_values = q_valid * alpha_values - tau_q_valid

        # Find spectrum characteristics
        f_max_idx = np.argmax(f_alpha_values)
        alpha_0 = alpha_values[f_max_idx]
        f_alpha_max = f_alpha_values[f_max_idx]

        alpha_min = np.min(alpha_values)
        alpha_max = np.max(alpha_values)
        spectrum_width = alpha_max - alpha_min

        # Asymmetry: (α_0 - α_min) / (α_max - α_min)
        if spectrum_width > 1e-6:
            spectrum_asymmetry = (alpha_0 - alpha_min) / spectrum_width
        else:
            spectrum_asymmetry = 0.5  # Symmetric by default

        if self.debug:
            print(f"      Spectrum width Δα = {spectrum_width:.4f}")
            print(f"      Asymmetry = {spectrum_asymmetry:.4f}")

        # ============================================
        # 5. Package results
        # ============================================
        analysis_time = time.time() - start_time
        mean_r_squared = np.mean(r_squared_valid)

        spectrum = MultifractalSpectrum(
            d0=d0,
            d1=d1,
            d2=d2,
            d_inf=d_inf,
            d_minus_inf=d_minus_inf,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            alpha_0=alpha_0,
            f_alpha_max=f_alpha_max,
            spectrum_width=spectrum_width,
            spectrum_asymmetry=spectrum_asymmetry,
            q_values=q_valid,
            d_q=d_q_valid,
            alpha=alpha_values,
            f_alpha=f_alpha_values,
            d_q_r_squared=r_squared_valid,
            mean_r_squared=mean_r_squared,
            n_scales=len(box_data),
            analysis_time=analysis_time
        )

        if self.debug:
            print(f"      ✅ Analysis complete in {analysis_time:.4f}s")

        return spectrum

    def _count_boxes(self, segments: np.ndarray, delta: float) -> Tuple[np.ndarray, float]:
        """
        Count segment lengths in each box of size delta.

        Args:
            segments: Interface segments
            delta: Box size

        Returns:
            Tuple of (box_counts, total_length)
        """
        # Find bounding box
        x_coords = segments[:, [0, 2]].ravel()
        y_coords = segments[:, [1, 3]].ravel()

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Create grid
        nx = int(np.ceil((x_max - x_min) / delta)) + 1
        ny = int(np.ceil((y_max - y_min) / delta)) + 1

        # Dictionary to store length in each box
        box_lengths = {}

        total_length = 0.0

        # For each segment, compute which boxes it intersects
        for seg in segments:
            x1, y1, x2, y2 = seg
            seg_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += seg_length

            # Box indices for segment endpoints
            ix1 = int((x1 - x_min) / delta)
            iy1 = int((y1 - y_min) / delta)
            ix2 = int((x2 - x_min) / delta)
            iy2 = int((y2 - y_min) / delta)

            # Simplified: assign full segment length to box containing midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ix_mid = int((mid_x - x_min) / delta)
            iy_mid = int((mid_y - y_min) / delta)

            # Ensure within bounds
            ix_mid = max(0, min(ix_mid, nx - 1))
            iy_mid = max(0, min(iy_mid, ny - 1))

            box_key = (ix_mid, iy_mid)
            if box_key not in box_lengths:
                box_lengths[box_key] = 0.0
            box_lengths[box_key] += seg_length

        # Convert to array
        if len(box_lengths) > 0:
            counts = np.array(list(box_lengths.values()))
        else:
            counts = np.array([])

        return counts, total_length


def main():
    """Test multifractal analysis."""
    import argparse
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from integration.rt_vtk_parser import VTKParser
    from integration.rt_interface_extraction import InterfaceExtractor

    parser = argparse.ArgumentParser(description='Test multifractal analysis')
    parser.add_argument('vtk_file', help='Path to VTK file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Parse VTK and extract interface
    print(f"📂 Loading VTK file...")
    vtk_parser = VTKParser(debug=args.debug)
    vtk_data = vtk_parser.parse_vtk_file(args.vtk_file)

    if not vtk_data:
        print(f"❌ Failed to parse VTK file")
        return

    print(f"\n🔍 Extracting interface...")
    interface_extractor = InterfaceExtractor(debug=args.debug)
    interface_data = interface_extractor.extract_interface(
        vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid
    )

    if not interface_data:
        print(f"❌ Failed to extract interface")
        return

    # Analyze multifractal
    print(f"\n🌀 Analyzing multifractal spectrum...")
    analyzer = MultifractalAnalyzer(debug=args.debug)

    # Use reasonable box sizes
    char_length = interface_data.metadata['bounds']['width']
    initial_delta = char_length / 20

    spectrum = analyzer.analyze_multifractal(
        interface_data.segments,
        initial_delta=initial_delta,
        delta_factor=1.5,
        num_steps=10
    )

    if not spectrum:
        print(f"❌ Multifractal analysis failed")
        return

    # Print results
    print(f"\n{'='*70}")
    print(f"🎉 MULTIFRACTAL ANALYSIS RESULTS")
    print(f"{'='*70}")

    print(f"\n📊 Generalized Dimensions:")
    print(f"   D₀ (capacity): {spectrum.d0:.6f}")
    print(f"   D₁ (information): {spectrum.d1:.6f}")
    print(f"   D₂ (correlation): {spectrum.d2:.6f}")
    print(f"   D_∞: {spectrum.d_inf:.6f}")
    print(f"   D₋∞: {spectrum.d_minus_inf:.6f}")

    print(f"\n🌀 Singularity Spectrum:")
    print(f"   α_min: {spectrum.alpha_min:.6f}")
    print(f"   α_0: {spectrum.alpha_0:.6f}")
    print(f"   α_max: {spectrum.alpha_max:.6f}")
    print(f"   f(α)_max: {spectrum.f_alpha_max:.6f}")

    print(f"\n📐 Spectrum Characteristics:")
    print(f"   Width Δα: {spectrum.spectrum_width:.6f}")
    print(f"   Asymmetry: {spectrum.spectrum_asymmetry:.6f}")

    print(f"\n✅ Quality:")
    print(f"   Mean R²: {spectrum.mean_r_squared:.6f}")
    print(f"   Scales used: {spectrum.n_scales}")
    print(f"   Analysis time: {spectrum.analysis_time:.4f}s")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
