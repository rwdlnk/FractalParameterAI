#!/usr/bin/env python3
"""
Wavelet Analysis for Multi-Scale Fractal Characterization

Performs continuous and discrete wavelet transforms to analyze interface
complexity at multiple scales. This directly matches the multi-scale nature
of box counting and provides superior parameter prediction.

Key insight: Wavelet energy at different scales → optimal δ ranges and steps
"""

import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets not available. Using built-in wavelet implementation.")


def morlet_wavelet(t, f0=6.0):
    """
    Generate Morlet wavelet for continuous wavelet transform.

    Args:
        t: Time array
        f0: Central frequency parameter

    Returns:
        wavelet: Complex Morlet wavelet
    """
    return np.exp(1j * 2 * np.pi * f0 * t) * np.exp(-t**2 / 2) / np.sqrt(np.pi**0.25)


def continuous_wavelet_transform(signal_data, scales, wavelet_func=None):
    """
    Perform continuous wavelet transform using convolution.

    Args:
        signal_data: 1D input signal
        scales: Array of scales to analyze
        wavelet_func: Wavelet function (default: Morlet)

    Returns:
        coefficients: 2D array (scales, time)
        freqs: Corresponding frequencies
    """
    if wavelet_func is None:
        wavelet_func = morlet_wavelet

    n = len(signal_data)
    coefficients = np.zeros((len(scales), n), dtype=complex)

    # Prepare frequency domain signal
    signal_fft = np.fft.fft(signal_data)
    freqs_signal = np.fft.fftfreq(n)

    for i, scale in enumerate(scales):
        # Generate wavelet at this scale
        if scale <= 0:
            continue

        # Time array for wavelet
        t_wavelet = np.arange(-3*scale, 3*scale, 1.0)
        if len(t_wavelet) == 0:
            continue

        wavelet = wavelet_func(t_wavelet / scale) / np.sqrt(scale)

        # Pad or trim wavelet to signal length
        if len(wavelet) > n:
            # Trim wavelet
            start = len(wavelet) // 2 - n // 2
            wavelet = wavelet[start:start+n]
        else:
            # Pad wavelet
            pad_before = (n - len(wavelet)) // 2
            pad_after = n - len(wavelet) - pad_before
            wavelet = np.pad(wavelet, (pad_before, pad_after), mode='constant')

        # Convolution in frequency domain
        wavelet_fft = np.fft.fft(np.conj(wavelet[::-1]))
        coefficients[i] = np.fft.ifft(signal_fft * wavelet_fft)

    # Calculate corresponding frequencies
    freqs = 1.0 / scales

    return coefficients, freqs


def extract_wavelet_features(segments, scales=None, method='continuous'):
    """
    Extract comprehensive wavelet features for AI parameter selection.

    Args:
        segments: Nx4 array of [x1, y1, x2, y2] segments
        scales: Array of scales to analyze (auto-generated if None)
        method: 'continuous' or 'discrete'

    Returns:
        features: Dictionary of wavelet-based features
    """
    from .power_spectrum_features import segments_to_height_function

    # Convert to height function
    x, y = segments_to_height_function(segments, num_points=512)

    if len(x) < 8:
        return {
            'wavelet_energy_distribution': np.array([]),
            'dominant_scales': np.array([]),
            'scale_energy_ratio': np.nan,
            'wavelet_entropy': np.nan,
            'intermittency_factor': np.nan,
            'multiscale_complexity': np.nan,
            'energy_concentration': np.nan,
            'scale_bandwidth': np.nan
        }

    # Auto-generate scales if not provided
    if scales is None:
        domain_size = x[-1] - x[0]
        # Create logarithmic scale distribution
        min_scale = domain_size / 256  # Finest scale
        max_scale = domain_size / 4    # Coarsest scale
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 32)

    # Remove mean and normalize
    y_processed = y - np.mean(y)
    if np.std(y_processed) > 0:
        y_processed = y_processed / np.std(y_processed)

    # Perform wavelet transform
    if method == 'continuous':
        coefficients, freqs = continuous_wavelet_transform(y_processed, scales)
    elif method == 'discrete' and PYWT_AVAILABLE:
        # Use PyWavelets if available
        coefficients_list = pywt.wavedec(y_processed, 'db4', level=6)
        # Convert to format similar to CWT
        coefficients = np.array([np.abs(c)**2 for c in coefficients_list])
        freqs = 1.0 / (2**np.arange(len(coefficients_list)))
    else:
        # Fallback to continuous
        coefficients, freqs = continuous_wavelet_transform(y_processed, scales)

    # Calculate energy at each scale
    energy_per_scale = np.mean(np.abs(coefficients)**2, axis=1)
    total_energy = np.sum(energy_per_scale)

    if total_energy == 0:
        return {
            'wavelet_energy_distribution': energy_per_scale,
            'dominant_scales': np.array([]),
            'scale_energy_ratio': np.nan,
            'wavelet_entropy': np.nan,
            'intermittency_factor': np.nan,
            'multiscale_complexity': np.nan,
            'energy_concentration': np.nan,
            'scale_bandwidth': np.nan
        }

    # Normalize energy distribution
    energy_distribution = energy_per_scale / total_energy

    # Feature 1: Dominant scales (peaks in energy distribution)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(energy_distribution, height=0.05)
    if len(peaks) > 0:
        dominant_scales = scales[peaks] if len(scales) == len(energy_distribution) else freqs[peaks]
    else:
        # If no clear peaks, use maximum energy scale
        max_idx = np.argmax(energy_distribution)
        dominant_scales = np.array([scales[max_idx] if len(scales) == len(energy_distribution) else freqs[max_idx]])

    # Feature 2: Scale energy ratio (large scale vs small scale energy)
    n_scales = len(energy_distribution)
    large_scale_energy = np.sum(energy_distribution[:n_scales//3])  # First third (large scales)
    small_scale_energy = np.sum(energy_distribution[2*n_scales//3:])  # Last third (small scales)
    scale_energy_ratio = large_scale_energy / small_scale_energy if small_scale_energy > 0 else np.inf

    # Feature 3: Wavelet entropy (measure of energy distribution uniformity)
    # Remove zeros to avoid log(0)
    nonzero_energy = energy_distribution[energy_distribution > 0]
    if len(nonzero_energy) > 1:
        wavelet_entropy = -np.sum(nonzero_energy * np.log2(nonzero_energy))
    else:
        wavelet_entropy = 0.0

    # Feature 4: Intermittency factor (measure of spiky vs smooth behavior)
    # Calculate local maxima in coefficients
    max_coeffs = np.max(np.abs(coefficients), axis=1)
    mean_coeffs = np.mean(np.abs(coefficients), axis=1)
    intermittency_factor = np.mean(max_coeffs / (mean_coeffs + 1e-10))

    # Feature 5: Multiscale complexity (variance across scales)
    multiscale_complexity = np.var(energy_distribution) if len(energy_distribution) > 1 else 0.0

    # Feature 6: Energy concentration (how concentrated energy is)
    # Gini coefficient-like measure
    sorted_energy = np.sort(energy_distribution)
    n = len(sorted_energy)
    if n > 1:
        cumsum = np.cumsum(sorted_energy)
        energy_concentration = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    else:
        energy_concentration = 1.0

    # Feature 7: Scale bandwidth (effective range of important scales)
    # Find scales containing 80% of energy
    cumulative_energy = np.cumsum(energy_distribution)
    low_idx = np.argmax(cumulative_energy >= 0.1)
    high_idx = np.argmax(cumulative_energy >= 0.9)

    if len(scales) == len(energy_distribution):
        scale_bandwidth = scales[high_idx] - scales[low_idx] if high_idx > low_idx else scales[-1] - scales[0]
    else:
        scale_bandwidth = freqs[high_idx] - freqs[low_idx] if high_idx > low_idx else freqs[-1] - freqs[0]

    return {
        'wavelet_energy_distribution': energy_distribution,
        'dominant_scales': dominant_scales,
        'scale_energy_ratio': scale_energy_ratio,
        'wavelet_entropy': wavelet_entropy,
        'intermittency_factor': intermittency_factor,
        'multiscale_complexity': multiscale_complexity,
        'energy_concentration': energy_concentration,
        'scale_bandwidth': abs(scale_bandwidth)
    }


def suggest_parameters_from_wavelets(wavelet_features, domain_scale=1.0):
    """
    Suggest box counting parameters based on wavelet analysis.

    Key relationships:
    - dominant_scales → initial_delta and step progression
    - scale_energy_ratio → delta_factor (more large scale → bigger factor)
    - wavelet_entropy → num_steps (more uniform → more steps needed)
    - energy_concentration → convergence strategy

    Args:
        wavelet_features: Features from extract_wavelet_features()
        domain_scale: Characteristic domain scale for normalization

    Returns:
        suggested_params: Dictionary with initial_delta, delta_factor, num_steps
    """
    # Extract key features
    dominant_scales = wavelet_features.get('dominant_scales', np.array([]))
    scale_energy_ratio = wavelet_features.get('scale_energy_ratio', 1.0)
    wavelet_entropy = wavelet_features.get('wavelet_entropy', 1.0)
    energy_concentration = wavelet_features.get('energy_concentration', 0.5)
    multiscale_complexity = wavelet_features.get('multiscale_complexity', 0.1)
    scale_bandwidth = wavelet_features.get('scale_bandwidth', domain_scale)

    # Handle NaN and invalid values
    if len(dominant_scales) == 0:
        primary_scale = domain_scale * 0.1  # Default to 10% of domain
    else:
        primary_scale = dominant_scales[0] if not np.isnan(dominant_scales[0]) else domain_scale * 0.1

    if not np.isfinite(scale_energy_ratio):
        scale_energy_ratio = 1.0
    if not np.isfinite(wavelet_entropy):
        wavelet_entropy = 1.0
    if not np.isfinite(energy_concentration):
        energy_concentration = 0.5
    if not np.isfinite(multiscale_complexity):
        multiscale_complexity = 0.1
    if not np.isfinite(scale_bandwidth):
        scale_bandwidth = domain_scale

    # Parameter 1: Initial delta (from dominant scale)
    # Dominant scale indicates characteristic size of features
    initial_delta = primary_scale * 2.0  # Start with boxes ~2x the dominant feature size

    # Clamp to reasonable range
    min_delta = domain_scale * 0.001  # 0.1% of domain
    max_delta = domain_scale * 0.3    # 30% of domain
    initial_delta = np.clip(initial_delta, min_delta, max_delta)

    # Parameter 2: Delta factor (from scale energy ratio)
    # High large-scale energy → more scale separation → larger factor
    if scale_energy_ratio > 3.0:
        delta_factor = 1.8  # Strong large-scale dominance
    elif scale_energy_ratio > 1.5:
        delta_factor = 1.6  # Moderate large-scale dominance
    elif scale_energy_ratio > 0.7:
        delta_factor = 1.4  # Balanced energy
    elif scale_energy_ratio > 0.3:
        delta_factor = 1.3  # Small-scale dominance
    else:
        delta_factor = 1.2  # Strong small-scale dominance

    # Adjust for energy concentration
    if energy_concentration > 0.8:
        delta_factor *= 0.9  # Energy concentrated → smaller steps
    elif energy_concentration < 0.3:
        delta_factor *= 1.1  # Energy spread → larger steps

    # Parameter 3: Number of steps (from entropy and complexity)
    base_steps = 12

    # Entropy adjustment (higher entropy → more uniform → more steps needed)
    if wavelet_entropy > 3.0:
        entropy_factor = 1.4  # High entropy
    elif wavelet_entropy > 2.0:
        entropy_factor = 1.2  # Moderate entropy
    elif wavelet_entropy > 1.0:
        entropy_factor = 1.0  # Low entropy
    else:
        entropy_factor = 0.8  # Very low entropy

    # Complexity adjustment
    if multiscale_complexity > 0.2:
        complexity_factor = 1.3  # High complexity
    elif multiscale_complexity > 0.1:
        complexity_factor = 1.1  # Moderate complexity
    else:
        complexity_factor = 0.9  # Low complexity

    # Scale bandwidth adjustment
    normalized_bandwidth = scale_bandwidth / domain_scale
    if normalized_bandwidth > 0.5:
        bandwidth_factor = 1.2  # Wide range of scales
    elif normalized_bandwidth > 0.2:
        bandwidth_factor = 1.0  # Moderate range
    else:
        bandwidth_factor = 0.8  # Narrow range

    # Combine factors
    steps_factor = entropy_factor * complexity_factor * bandwidth_factor
    num_steps = int(base_steps * steps_factor)
    num_steps = np.clip(num_steps, 8, 30)  # Reasonable range

    return {
        'initial_delta': initial_delta,
        'delta_factor': delta_factor,
        'num_steps': num_steps,
        'method': 'wavelet_analysis',
        'confidence': calculate_wavelet_confidence(wavelet_features),
        'scale_insights': {
            'primary_scale': primary_scale,
            'scale_energy_ratio': scale_energy_ratio,
            'entropy': wavelet_entropy,
            'energy_concentration': energy_concentration
        }
    }


def calculate_wavelet_confidence(wavelet_features):
    """
    Calculate confidence in wavelet-based parameter suggestions.

    High confidence when:
    - Clear dominant scales
    - Good energy distribution
    - Reasonable entropy values

    Args:
        wavelet_features: Features from extract_wavelet_features()

    Returns:
        confidence: Float 0-1 indicating confidence level
    """
    confidence = 1.0

    # Check for valid dominant scales
    dominant_scales = wavelet_features.get('dominant_scales', np.array([]))
    if len(dominant_scales) == 0:
        confidence *= 0.6  # No clear dominant scale
    elif len(dominant_scales) > 3:
        confidence *= 0.8  # Too many dominant scales (noisy)

    # Check energy distribution
    energy_dist = wavelet_features.get('wavelet_energy_distribution', np.array([]))
    if len(energy_dist) == 0:
        confidence *= 0.5  # No energy distribution
    else:
        # Check if energy is reasonably distributed (not all in one scale)
        max_energy_fraction = np.max(energy_dist) if len(energy_dist) > 0 else 1.0
        if max_energy_fraction > 0.9:
            confidence *= 0.7  # Too concentrated
        elif max_energy_fraction < 0.1:
            confidence *= 0.8  # Too spread out

    # Check for NaN features
    feature_values = [
        wavelet_features.get('scale_energy_ratio', np.nan),
        wavelet_features.get('wavelet_entropy', np.nan),
        wavelet_features.get('intermittency_factor', np.nan),
        wavelet_features.get('multiscale_complexity', np.nan),
        wavelet_features.get('energy_concentration', np.nan),
        wavelet_features.get('scale_bandwidth', np.nan)
    ]

    nan_count = sum(1 for v in feature_values if not np.isfinite(v))
    confidence *= (1.0 - 0.1 * nan_count)  # -10% per NaN feature

    # Check entropy range (should be reasonable)
    entropy = wavelet_features.get('wavelet_entropy', np.nan)
    if np.isfinite(entropy):
        if 0.5 <= entropy <= 5.0:
            confidence *= 1.0  # Good entropy range
        else:
            confidence *= 0.8  # Unusual entropy

    return np.clip(confidence, 0.0, 1.0)


def analyze_fractal_scaling_from_wavelets(wavelet_features):
    """
    Analyze fractal scaling properties from wavelet decomposition.

    Args:
        wavelet_features: Features from extract_wavelet_features()

    Returns:
        scaling_analysis: Dictionary with fractal scaling insights
    """
    energy_dist = wavelet_features.get('wavelet_energy_distribution', np.array([]))
    dominant_scales = wavelet_features.get('dominant_scales', np.array([]))

    if len(energy_dist) == 0:
        return {
            'scaling_exponent': np.nan,
            'scaling_range': np.nan,
            'self_similarity': np.nan,
            'fractal_character': 'unknown'
        }

    # Estimate scaling exponent from energy distribution
    # Power law: E(scale) ∝ scale^β
    scales = np.logspace(-2, 0, len(energy_dist))  # Normalized scales

    # Remove zeros for log-log fit
    valid_mask = (energy_dist > 0) & (scales > 0)
    if np.sum(valid_mask) < 3:
        return {
            'scaling_exponent': np.nan,
            'scaling_range': np.nan,
            'self_similarity': np.nan,
            'fractal_character': 'insufficient_data'
        }

    log_scales = np.log10(scales[valid_mask])
    log_energy = np.log10(energy_dist[valid_mask])

    # Linear fit in log-log space
    try:
        coeffs = np.polyfit(log_scales, log_energy, 1)
        scaling_exponent = coeffs[0]

        # Calculate R² for fit quality
        y_pred = np.polyval(coeffs, log_scales)
        ss_res = np.sum((log_energy - y_pred)**2)
        ss_tot = np.sum((log_energy - np.mean(log_energy))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Scaling range (range where fit is good)
        scaling_range = scales[valid_mask][-1] - scales[valid_mask][0]

        # Self-similarity measure (how well power law fits)
        self_similarity = r_squared

        # Characterize fractal behavior
        if r_squared > 0.8:
            if -2.0 < scaling_exponent < 0:
                fractal_character = 'strong_fractal'
            elif -3.0 < scaling_exponent < -2.0:
                fractal_character = 'moderate_fractal'
            else:
                fractal_character = 'weak_fractal'
        else:
            fractal_character = 'non_fractal'

    except:
        scaling_exponent = np.nan
        scaling_range = np.nan
        self_similarity = np.nan
        fractal_character = 'analysis_failed'

    return {
        'scaling_exponent': scaling_exponent,
        'scaling_range': scaling_range,
        'self_similarity': self_similarity,
        'fractal_character': fractal_character
    }