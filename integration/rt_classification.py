#!/usr/bin/env python3
"""
Rayleigh-Taylor Instability Classification Module

Provides RT-specific analysis for bubble/spike identification and growth rate analysis.

Key Features:
- Bubble vs spike detection
- Penetration height measurements
- Mixing width calculation
- Growth rate analysis (α from h = αAtg²t²)

Based on RT instability physics where:
- Bubbles: Light fluid rising into heavy fluid (positive curvature)
- Spikes: Heavy fluid falling into light fluid (negative curvature)
- Mixing width grows as h ≈ αAtg²t² where α ≈ 0.05-0.07 for classical RT
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RTFeatures:
    """Container for RT-specific features."""
    bubble_count: int
    spike_count: int
    bubble_penetration_height: float
    spike_penetration_depth: float
    mixing_width: float
    bubble_positions: List[Tuple[float, float]]  # (x, y) coordinates
    spike_positions: List[Tuple[float, float]]   # (x, y) coordinates
    interface_mean_height: float
    interface_amplitude: float


class RTClassifier:
    """
    Rayleigh-Taylor instability classifier.

    Detects and characterizes bubbles (rising) and spikes (falling)
    in RT interface data.
    """

    def __init__(self,
                 min_feature_width: float = 0.01,
                 curvature_threshold: float = 0.0,
                 debug: bool = False):
        """
        Initialize RT classifier.

        Args:
            min_feature_width: Minimum width to consider a bubble/spike
            curvature_threshold: Threshold for curvature detection
            debug: Enable debug output
        """
        self.min_feature_width = min_feature_width
        self.curvature_threshold = curvature_threshold
        self.debug = debug

    def analyze_rt_interface(self,
                            interface_data,
                            initial_height: Optional[float] = None) -> Dict:
        """
        Perform complete RT analysis on interface data.

        Args:
            interface_data: InterfaceData object from rt_interface_extraction
            initial_height: Initial interface height (for growth rate)

        Returns:
            Dictionary with RT classification results
        """
        if self.debug:
            print(f"   🔬 RT Classification Analysis")

        # Extract interface points and segments
        segments = interface_data.segments  # Shape (N, 4): [x1, y1, x2, y2]
        bounds = interface_data.metadata['bounds']

        # Get interface as continuous path
        points = self._segments_to_points(segments)

        if len(points) < 3:
            if self.debug:
                print(f"      ⚠️  Too few points for RT analysis")
            return self._empty_results()

        # Sort points by x-coordinate for analysis
        points = points[np.argsort(points[:, 0])]

        # Compute mean interface height
        mean_height = np.mean(points[:, 1])

        if initial_height is None:
            initial_height = mean_height

        # Detect bubbles and spikes
        bubbles, spikes = self._detect_features(points, mean_height)

        # Compute penetration heights
        if len(bubbles) > 0:
            bubble_heights = [y for (x, y) in bubbles]
            bubble_penetration = max(bubble_heights) - mean_height
        else:
            bubble_penetration = 0.0

        if len(spikes) > 0:
            spike_depths = [y for (x, y) in spikes]
            spike_penetration = mean_height - min(spike_depths)
        else:
            spike_penetration = 0.0

        # Mixing width (total)
        mixing_width = bubble_penetration + spike_penetration

        # Interface amplitude (peak-to-peak)
        y_min, y_max = bounds['y_min'], bounds['y_max']
        amplitude = y_max - y_min

        if self.debug:
            print(f"      🫧 Bubbles detected: {len(bubbles)}")
            print(f"      📌 Spikes detected: {len(spikes)}")
            print(f"      ⬆️  Bubble penetration: {bubble_penetration:.6f}")
            print(f"      ⬇️  Spike penetration: {spike_penetration:.6f}")
            print(f"      📏 Mixing width: {mixing_width:.6f}")
            print(f"      📊 Amplitude: {amplitude:.6f}")

        # Compile results
        results = {
            'bubble_count': len(bubbles),
            'spike_count': len(spikes),
            'bubble_penetration_height': bubble_penetration,
            'spike_penetration_depth': spike_penetration,
            'mixing_width': mixing_width,
            'interface_mean_height': mean_height,
            'interface_amplitude': amplitude,
            'bubble_positions': bubbles,
            'spike_positions': spikes,
            'total_features': len(bubbles) + len(spikes)
        }

        return results

    def _segments_to_points(self, segments: np.ndarray) -> np.ndarray:
        """
        Convert line segments to ordered point array.

        Optimized version using rounding for duplicate detection instead of
        nested loops. For 25K segments this is ~1000× faster.

        Args:
            segments: Array of shape (N, 4) with [x1, y1, x2, y2]

        Returns:
            Array of shape (M, 2) with unique points
        """
        # Extract all endpoints
        points_list = []
        for seg in segments:
            x1, y1, x2, y2 = seg
            points_list.append([x1, y1])
            points_list.append([x2, y2])

        # Convert to array
        points = np.array(points_list)

        # Remove duplicates using vectorized approach - O(N log N) instead of O(N²)
        # Round to 10 decimal places for duplicate detection
        tolerance_decimals = 10
        points_rounded = np.round(points, decimals=tolerance_decimals)

        # Use numpy's unique to get first occurrence of each unique point
        _, unique_indices = np.unique(points_rounded, axis=0, return_index=True)

        # Sort indices to maintain original order
        unique_indices = np.sort(unique_indices)

        return points[unique_indices]

    def _detect_features(self,
                        points: np.ndarray,
                        mean_height: float) -> Tuple[List[Tuple[float, float]],
                                                      List[Tuple[float, float]]]:
        """
        Detect bubbles and spikes using local extrema.

        Bubbles are local maxima above mean height.
        Spikes are local minima below mean height.

        Args:
            points: Array of (x, y) points sorted by x
            mean_height: Mean interface height

        Returns:
            Tuple of (bubble_positions, spike_positions)
        """
        bubbles = []
        spikes = []

        if len(points) < 3:
            return bubbles, spikes

        x = points[:, 0]
        y = points[:, 1]

        # Find local maxima (bubbles) and minima (spikes)
        # Use a simple window-based approach
        window = max(3, len(points) // 20)  # Adaptive window size

        for i in range(window, len(points) - window):
            # Check if this is a local maximum
            local_window_y = y[i-window:i+window+1]

            if y[i] == np.max(local_window_y) and y[i] > mean_height:
                # This is a bubble (local max above mean)
                # Check if it's far enough from existing bubbles
                is_new_bubble = True
                for bx, by in bubbles:
                    if abs(x[i] - bx) < self.min_feature_width:
                        # Too close to existing bubble
                        if y[i] > by:
                            # This one is higher, replace
                            bubbles.remove((bx, by))
                        else:
                            is_new_bubble = False
                        break

                if is_new_bubble:
                    bubbles.append((x[i], y[i]))

            elif y[i] == np.min(local_window_y) and y[i] < mean_height:
                # This is a spike (local min below mean)
                # Check if it's far enough from existing spikes
                is_new_spike = True
                for sx, sy in spikes:
                    if abs(x[i] - sx) < self.min_feature_width:
                        # Too close to existing spike
                        if y[i] < sy:
                            # This one is lower, replace
                            spikes.remove((sx, sy))
                        else:
                            is_new_spike = False
                        break

                if is_new_spike:
                    spikes.append((x[i], y[i]))

        return bubbles, spikes

    def _empty_results(self) -> Dict:
        """Return empty RT results."""
        return {
            'bubble_count': 0,
            'spike_count': 0,
            'bubble_penetration_height': 0.0,
            'spike_penetration_depth': 0.0,
            'mixing_width': 0.0,
            'interface_mean_height': 0.0,
            'interface_amplitude': 0.0,
            'bubble_positions': [],
            'spike_positions': [],
            'total_features': 0
        }


def compute_growth_rate(temporal_results: Dict,
                       atwood_number: float = 0.5,
                       gravity: float = 9.81) -> Dict:
    """
    Compute RT growth rate α from temporal evolution.

    h = α * A * g * t²

    Args:
        temporal_results: Temporal analysis results with RT classification
        atwood_number: A = (ρ_heavy - ρ_light) / (ρ_heavy + ρ_light)
        gravity: Gravitational acceleration (m/s²)

    Returns:
        Dictionary with growth rate analysis
    """
    if not temporal_results.get('success') or 'results' not in temporal_results:
        return {'error': 'Invalid temporal results'}

    results = temporal_results['results']

    # Extract time series
    times = []
    mixing_widths = []

    for result in results:
        if 'rt_classification' in result:
            rt = result['rt_classification']
            times.append(result['time'])
            mixing_widths.append(rt['mixing_width'])

    if len(times) < 2:
        return {'error': 'Not enough temporal data for growth rate'}

    times = np.array(times)
    mixing_widths = np.array(mixing_widths)

    # Fit h = α * A * g * t²
    # Rearrange: h / (A * g * t²) = α

    t_squared = times ** 2

    # Linear fit (should pass through origin)
    # h = (α * A * g) * t²

    coeff = np.polyfit(t_squared, mixing_widths, 1)
    alpha_times_Ag = coeff[0]

    # Extract α
    alpha = alpha_times_Ag / (atwood_number * gravity)

    # Compute R² of fit
    predicted = alpha_times_Ag * t_squared + coeff[1]
    ss_res = np.sum((mixing_widths - predicted) ** 2)
    ss_tot = np.sum((mixing_widths - np.mean(mixing_widths)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'growth_rate_alpha': alpha,
        'alpha_fit_r_squared': r_squared,
        'atwood_number': atwood_number,
        'gravity': gravity,
        'time_range': (times[0], times[-1]),
        'mixing_width_range': (mixing_widths[0], mixing_widths[-1])
    }
