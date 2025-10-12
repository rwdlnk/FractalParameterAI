#!/usr/bin/env python3
"""
Power Spectrum Analysis for Enhanced Fractal Parameter Selection

Analyzes the frequency content of interface roughness to directly predict
optimal box counting parameters. This advanced measure reveals characteristic
scales that traditional geometric features miss.

Key insight: Dominant frequencies → optimal δ₀, spectral slopes → δ-factor
"""

import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


def segments_to_height_function(segments, num_points=1024):
    """
    Convert segment array to height function h(x) for spectral analysis.

    Args:
        segments: Nx4 array of [x1, y1, x2, y2] segments
        num_points: Number of interpolation points for uniform sampling

    Returns:
        x: Uniform x coordinates
        y: Height function h(x)
    """
    if len(segments) == 0:
        return np.array([]), np.array([])

    # Extract all x,y points from segments
    points = []
    for seg in segments:
        points.extend([(seg[0], seg[1]), (seg[2], seg[3])])

    # Remove duplicates and sort by x
    points = list(set(points))
    points.sort(key=lambda p: p[0])

    if len(points) < 2:
        return np.array([]), np.array([])

    x_raw = np.array([p[0] for p in points])
    y_raw = np.array([p[1] for p in points])

    # Create uniform x grid
    x_min, x_max = x_raw.min(), x_raw.max()
    x_uniform = np.linspace(x_min, x_max, num_points)

    # Interpolate to uniform grid
    y_uniform = np.interp(x_uniform, x_raw, y_raw)

    return x_uniform, y_uniform


def calculate_power_spectral_density(x, y, method='welch'):
    """
    Calculate power spectral density of height function.

    Args:
        x: X coordinates (uniform spacing)
        y: Height function values
        method: 'welch', 'periodogram', or 'multitaper'

    Returns:
        freqs: Frequency array
        psd: Power spectral density
    """
    if len(x) < 4 or len(y) < 4:
        return np.array([]), np.array([])

    # Calculate sampling frequency
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    fs = 1.0 / dx

    # Remove mean (detrend)
    y_detrended = y - np.mean(y)

    try:
        if method == 'welch':
            # Welch method - robust for noisy data
            freqs, psd = signal.welch(y_detrended, fs=fs, nperseg=min(256, len(y)//4))
        elif method == 'periodogram':
            # Simple periodogram
            freqs, psd = signal.periodogram(y_detrended, fs=fs)
        elif method == 'multitaper':
            # Multitaper method - best for short signals
            freqs, psd = signal.periodogram(y_detrended, fs=fs, window='dpss')
        else:
            raise ValueError(f"Unknown method: {method}")

        # Remove DC component
        if len(freqs) > 1:
            freqs = freqs[1:]
            psd = psd[1:]

        return freqs, psd

    except Exception as e:
        print(f"PSD calculation failed: {e}")
        return np.array([]), np.array([])


def fit_power_law_slope(freqs, psd, freq_range=None):
    """
    Fit power law slope β to PSD: S(f) ∝ f^(-β)

    Args:
        freqs: Frequency array
        psd: Power spectral density
        freq_range: [f_min, f_max] for fitting, None for auto

    Returns:
        slope: Power law exponent β
    """
    if len(freqs) < 3 or len(psd) < 3:
        return np.nan

    # Remove zeros and negative values
    valid_mask = (freqs > 0) & (psd > 0)
    if np.sum(valid_mask) < 3:
        return np.nan

    freqs_valid = freqs[valid_mask]
    psd_valid = psd[valid_mask]

    # Auto-select frequency range if not provided
    if freq_range is None:
        # Use middle 60% of frequency range to avoid edge effects
        f_min = np.percentile(freqs_valid, 20)
        f_max = np.percentile(freqs_valid, 80)
    else:
        f_min, f_max = freq_range

    # Select fitting range
    fit_mask = (freqs_valid >= f_min) & (freqs_valid <= f_max)
    if np.sum(fit_mask) < 3:
        return np.nan

    freqs_fit = freqs_valid[fit_mask]
    psd_fit = psd_valid[fit_mask]

    try:
        # Linear fit in log-log space
        log_freqs = np.log10(freqs_fit)
        log_psd = np.log10(psd_fit)

        # Remove any remaining invalid values
        finite_mask = np.isfinite(log_freqs) & np.isfinite(log_psd)
        if np.sum(finite_mask) < 3:
            return np.nan

        coeffs = np.polyfit(log_freqs[finite_mask], log_psd[finite_mask], 1)
        slope = -coeffs[0]  # Negative because S(f) ∝ f^(-β)

        return slope

    except Exception:
        return np.nan


def calculate_spectral_bandwidth(freqs, psd, fraction=0.95):
    """
    Calculate spectral bandwidth containing given fraction of total power.

    Args:
        freqs: Frequency array
        psd: Power spectral density
        fraction: Fraction of power to include (0.95 = 95%)

    Returns:
        bandwidth: Frequency bandwidth
    """
    if len(freqs) == 0 or len(psd) == 0:
        return np.nan

    # Calculate cumulative power
    total_power = np.trapz(psd, freqs)
    if total_power <= 0:
        return np.nan

    cumulative_power = np.cumsum(psd) * (freqs[1] - freqs[0])
    normalized_cumulative = cumulative_power / total_power

    # Find frequency range containing specified fraction
    lower_idx = np.argmax(normalized_cumulative >= (1 - fraction) / 2)
    upper_idx = np.argmax(normalized_cumulative >= fraction + (1 - fraction) / 2)

    if upper_idx > lower_idx:
        bandwidth = freqs[upper_idx] - freqs[lower_idx]
    else:
        bandwidth = freqs[-1] - freqs[0]

    return bandwidth


def extract_power_spectrum_features(segments, method='welch', num_points=1024):
    """
    Extract comprehensive power spectrum features for AI parameter selection.

    Args:
        segments: Nx4 array of [x1, y1, x2, y2] segments
        method: PSD calculation method
        num_points: Interpolation points

    Returns:
        features: Dictionary of spectral features
    """
    # Convert to height function
    x, y = segments_to_height_function(segments, num_points)

    if len(x) < 4:
        return {
            'dominant_frequency': np.nan,
            'spectral_slope': np.nan,
            'high_frequency_content': np.nan,
            'spectral_bandwidth': np.nan,
            'peak_frequency_ratio': np.nan,
            'spectral_centroid': np.nan,
            'spectral_rolloff': np.nan,
            'spectral_energy_ratio': np.nan
        }

    # Calculate power spectral density
    freqs, psd = calculate_power_spectral_density(x, y, method)

    if len(freqs) == 0:
        return {
            'dominant_frequency': np.nan,
            'spectral_slope': np.nan,
            'high_frequency_content': np.nan,
            'spectral_bandwidth': np.nan,
            'peak_frequency_ratio': np.nan,
            'spectral_centroid': np.nan,
            'spectral_rolloff': np.nan,
            'spectral_energy_ratio': np.nan
        }

    # Feature 1: Dominant frequency (peak frequency)
    peak_idx = np.argmax(psd)
    dominant_frequency = freqs[peak_idx]

    # Feature 2: Spectral slope (power law exponent)
    spectral_slope = fit_power_law_slope(freqs, psd)

    # Feature 3: High frequency content (> 75th percentile)
    freq_75th = np.percentile(freqs, 75)
    high_freq_mask = freqs > freq_75th
    total_power = np.trapz(psd, freqs)
    if total_power > 0:
        high_frequency_content = np.trapz(psd[high_freq_mask], freqs[high_freq_mask]) / total_power
    else:
        high_frequency_content = 0.0

    # Feature 4: Spectral bandwidth (95% power)
    spectral_bandwidth = calculate_spectral_bandwidth(freqs, psd, 0.95)

    # Feature 5: Peak frequency ratio (dominant/mean)
    mean_frequency = np.trapz(freqs * psd, freqs) / total_power if total_power > 0 else 0
    peak_frequency_ratio = dominant_frequency / mean_frequency if mean_frequency > 0 else 1.0

    # Feature 6: Spectral centroid (center of mass in frequency)
    spectral_centroid = mean_frequency

    # Feature 7: Spectral rolloff (95% cumulative energy)
    cumulative_energy = np.cumsum(psd)
    rolloff_idx = np.argmax(cumulative_energy >= 0.95 * cumulative_energy[-1])
    spectral_rolloff = freqs[rolloff_idx] if rolloff_idx > 0 else freqs[-1]

    # Feature 8: Low/high frequency energy ratio
    median_freq = np.median(freqs)
    low_freq_mask = freqs <= median_freq
    high_freq_mask = freqs > median_freq

    low_energy = np.trapz(psd[low_freq_mask], freqs[low_freq_mask])
    high_energy = np.trapz(psd[high_freq_mask], freqs[high_freq_mask])
    spectral_energy_ratio = low_energy / high_energy if high_energy > 0 else np.inf

    return {
        'dominant_frequency': dominant_frequency,
        'spectral_slope': spectral_slope,
        'high_frequency_content': high_frequency_content,
        'spectral_bandwidth': spectral_bandwidth,
        'peak_frequency_ratio': peak_frequency_ratio,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'spectral_energy_ratio': spectral_energy_ratio
    }


def suggest_parameters_from_spectrum(spectral_features, domain_scale=1.0):
    """
    Suggest box counting parameters based on spectral analysis.

    Key relationships:
    - dominant_frequency → initial_delta (inverse relationship)
    - spectral_slope → delta_factor (steeper slope → larger factor)
    - spectral_bandwidth → num_steps (wider bandwidth → more steps)

    Args:
        spectral_features: Features from extract_power_spectrum_features()
        domain_scale: Characteristic domain scale for normalization

    Returns:
        suggested_params: Dictionary with initial_delta, delta_factor, num_steps
    """
    # Extract key features
    dom_freq = spectral_features.get('dominant_frequency', 1.0)
    slope = spectral_features.get('spectral_slope', 1.5)
    bandwidth = spectral_features.get('spectral_bandwidth', 1.0)
    high_freq_content = spectral_features.get('high_frequency_content', 0.2)

    # Handle NaN values with reasonable defaults
    if not np.isfinite(dom_freq) or dom_freq <= 0:
        dom_freq = 1.0 / domain_scale if domain_scale > 0 else 1.0
    if not np.isfinite(slope):
        slope = 1.5  # Typical fractal slope
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = dom_freq
    if not np.isfinite(high_freq_content):
        high_freq_content = 0.2

    # Parameter 1: Initial delta (inverse of dominant frequency)
    # Higher frequency → smaller initial boxes needed
    initial_delta = domain_scale / (dom_freq * 10)  # Factor of 10 for reasonable scale

    # Clamp to reasonable range
    min_delta = domain_scale * 0.001  # 0.1% of domain
    max_delta = domain_scale * 0.2    # 20% of domain
    initial_delta = np.clip(initial_delta, min_delta, max_delta)

    # Parameter 2: Delta factor (from spectral slope)
    # Steeper negative slopes suggest more scale separation → larger factor
    if slope > 2.5:
        delta_factor = 1.8  # Strong scaling → large factor
    elif slope > 1.5:
        delta_factor = 1.5  # Moderate scaling
    elif slope > 0.5:
        delta_factor = 1.3  # Weak scaling
    else:
        delta_factor = 1.2  # Very weak or no scaling

    # Adjust for high frequency content
    if high_freq_content > 0.3:
        delta_factor *= 0.9  # More high freq → smaller factor for finer resolution

    # Parameter 3: Number of steps (from bandwidth and slope)
    # Wider bandwidth or steeper slope → more steps needed
    base_steps = 12

    # Bandwidth adjustment
    if bandwidth > 2 * dom_freq:
        steps_adjustment = 1.5  # Wide bandwidth → more steps
    elif bandwidth < 0.5 * dom_freq:
        steps_adjustment = 0.7  # Narrow bandwidth → fewer steps
    else:
        steps_adjustment = 1.0

    # Slope adjustment
    if slope > 2.0:
        steps_adjustment *= 1.3  # Strong scaling → more steps for better fit
    elif slope < 1.0:
        steps_adjustment *= 0.8  # Weak scaling → fewer steps sufficient

    num_steps = int(base_steps * steps_adjustment)
    num_steps = np.clip(num_steps, 8, 25)  # Reasonable range

    return {
        'initial_delta': initial_delta,
        'delta_factor': delta_factor,
        'num_steps': num_steps,
        'method': 'power_spectrum_analysis',
        'confidence': calculate_spectral_confidence(spectral_features)
    }


def calculate_spectral_confidence(spectral_features):
    """
    Calculate confidence in spectral-based parameter suggestions.

    High confidence when:
    - Clear dominant frequency peak
    - Well-defined spectral slope
    - Reasonable bandwidth

    Args:
        spectral_features: Features from extract_power_spectrum_features()

    Returns:
        confidence: Float 0-1 indicating confidence level
    """
    confidence = 1.0

    # Check for NaN values (reduce confidence)
    nan_count = sum(1 for v in spectral_features.values() if not np.isfinite(v))
    confidence *= (1.0 - 0.1 * nan_count)  # -10% per NaN feature

    # Check spectral slope quality
    slope = spectral_features.get('spectral_slope', np.nan)
    if np.isfinite(slope):
        if 0.5 <= slope <= 3.0:
            confidence *= 1.0  # Good slope range
        else:
            confidence *= 0.7  # Unusual slope
    else:
        confidence *= 0.5  # No slope measurement

    # Check peak clarity
    peak_ratio = spectral_features.get('peak_frequency_ratio', 1.0)
    if np.isfinite(peak_ratio):
        if peak_ratio > 1.5:
            confidence *= 1.0  # Clear peak
        elif peak_ratio > 1.2:
            confidence *= 0.9  # Moderate peak
        else:
            confidence *= 0.7  # Weak peak

    return np.clip(confidence, 0.0, 1.0)