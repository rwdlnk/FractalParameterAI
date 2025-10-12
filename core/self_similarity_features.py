#!/usr/bin/env python3
"""
Self-Similarity Detection for Direct Fractal Scaling Analysis

Analyzes self-similarity properties of interfaces using structure functions,
correlation analysis, and scaling exponent estimation. This directly relates
to fractal dimension and optimal box counting parameters.

Key insight: Self-similarity scaling → fractal dimension prediction and optimal δ ranges
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


def calculate_structure_function(signal, lag, order=2):
    """
    Calculate structure function of order q at given lag.

    Structure function: S_q(τ) = <|x(t+τ) - x(t)|^q>

    Args:
        signal: 1D signal array
        lag: Lag value (integer)
        order: Order q of structure function

    Returns:
        structure_value: Structure function value
    """
    if lag >= len(signal) or lag <= 0:
        return np.nan

    diffs = signal[lag:] - signal[:-lag]
    if order == 2:
        structure_value = np.mean(diffs**2)
    else:
        structure_value = np.mean(np.abs(diffs)**order)

    return structure_value


def estimate_hurst_exponent(signal, max_lag=None, method='structure_function'):
    """
    Estimate Hurst exponent using various methods.

    Args:
        signal: 1D signal array
        max_lag: Maximum lag to consider (default: len(signal)//4)
        method: 'structure_function', 'variance_aggregation', or 'detrended_fluctuation'

    Returns:
        hurst_exponent: Estimated Hurst exponent
        r_squared: Goodness of fit
    """
    if len(signal) < 10:
        return np.nan, np.nan

    if max_lag is None:
        max_lag = min(len(signal) // 4, 50)

    if method == 'structure_function':
        return _hurst_structure_function(signal, max_lag)
    elif method == 'variance_aggregation':
        return _hurst_variance_aggregation(signal, max_lag)
    elif method == 'detrended_fluctuation':
        return _hurst_detrended_fluctuation(signal, max_lag)
    else:
        raise ValueError(f"Unknown method: {method}")


def _hurst_structure_function(signal, max_lag):
    """Estimate Hurst exponent using structure function method."""
    lags = np.arange(1, max_lag + 1)
    structure_values = []

    for lag in lags:
        sf = calculate_structure_function(signal, lag, order=2)
        if np.isfinite(sf) and sf > 0:
            structure_values.append(sf)
        else:
            structure_values.append(np.nan)

    structure_values = np.array(structure_values)
    valid_mask = np.isfinite(structure_values) & (structure_values > 0)

    if np.sum(valid_mask) < 3:
        return np.nan, np.nan

    log_lags = np.log10(lags[valid_mask])
    log_structure = np.log10(structure_values[valid_mask])

    try:
        # Linear fit: log(S(τ)) = 2H * log(τ) + const
        slope, intercept, r_value, _, _ = stats.linregress(log_lags, log_structure)
        hurst_exponent = slope / 2.0
        r_squared = r_value**2
    except:
        return np.nan, np.nan

    return hurst_exponent, r_squared


def _hurst_variance_aggregation(signal, max_lag):
    """Estimate Hurst exponent using variance aggregation method."""
    n = len(signal)
    variances = []
    block_sizes = []

    for block_size in range(2, min(max_lag + 1, n // 2)):
        n_blocks = n // block_size
        if n_blocks < 2:
            continue

        # Aggregate signal into blocks
        aggregated = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            aggregated.append(np.mean(signal[start:end]))

        if len(aggregated) > 1:
            variances.append(np.var(aggregated))
            block_sizes.append(block_size)

    if len(variances) < 3:
        return np.nan, np.nan

    variances = np.array(variances)
    block_sizes = np.array(block_sizes)

    valid_mask = (variances > 0) & np.isfinite(variances)
    if np.sum(valid_mask) < 3:
        return np.nan, np.nan

    try:
        # Fit: log(Var) = (2H - 2) * log(block_size) + const
        log_sizes = np.log10(block_sizes[valid_mask])
        log_variances = np.log10(variances[valid_mask])

        slope, intercept, r_value, _, _ = stats.linregress(log_sizes, log_variances)
        hurst_exponent = (slope + 2) / 2.0
        r_squared = r_value**2
    except:
        return np.nan, np.nan

    return hurst_exponent, r_squared


def _hurst_detrended_fluctuation(signal, max_lag):
    """Estimate Hurst exponent using detrended fluctuation analysis."""
    n = len(signal)

    # Integrate signal
    y = np.cumsum(signal - np.mean(signal))

    fluctuations = []
    scales = []

    for scale in range(4, min(max_lag + 1, n // 4)):
        n_segments = n // scale

        if n_segments < 2:
            continue

        segment_fluctuations = []

        for i in range(n_segments):
            start = i * scale
            end = start + scale
            segment = y[start:end]

            # Detrend with linear fit
            x_segment = np.arange(len(segment))
            if len(segment) > 1:
                coeffs = np.polyfit(x_segment, segment, 1)
                trend = np.polyval(coeffs, x_segment)
                detrended = segment - trend

                # Calculate RMS fluctuation
                rms = np.sqrt(np.mean(detrended**2))
                segment_fluctuations.append(rms)

        if segment_fluctuations:
            avg_fluctuation = np.mean(segment_fluctuations)
            fluctuations.append(avg_fluctuation)
            scales.append(scale)

    if len(fluctuations) < 3:
        return np.nan, np.nan

    fluctuations = np.array(fluctuations)
    scales = np.array(scales)

    valid_mask = (fluctuations > 0) & np.isfinite(fluctuations)
    if np.sum(valid_mask) < 3:
        return np.nan, np.nan

    try:
        # Fit: log(F) = H * log(scale) + const
        log_scales = np.log10(scales[valid_mask])
        log_fluctuations = np.log10(fluctuations[valid_mask])

        slope, intercept, r_value, _, _ = stats.linregress(log_scales, log_fluctuations)
        hurst_exponent = slope
        r_squared = r_value**2
    except:
        return np.nan, np.nan

    return hurst_exponent, r_squared


def detect_scaling_range(signal, method='structure_function'):
    """
    Detect the range of scales where self-similarity holds.

    Args:
        signal: 1D signal array
        method: Method for detecting scaling

    Returns:
        scaling_range: [min_scale, max_scale] where scaling is valid
        scaling_quality: Quality measure of scaling region
    """
    if len(signal) < 20:
        return [np.nan, np.nan], np.nan

    max_lag = min(len(signal) // 3, 100)
    lags = np.arange(1, max_lag + 1)

    if method == 'structure_function':
        values = []
        for lag in lags:
            sf = calculate_structure_function(signal, lag, order=2)
            values.append(sf if np.isfinite(sf) and sf > 0 else np.nan)
        values = np.array(values)
    else:
        # Use variance method
        values = []
        for lag in lags:
            if lag < len(signal):
                windowed_vars = []
                for i in range(0, len(signal) - lag, lag):
                    window = signal[i:i+lag]
                    if len(window) > 1:
                        windowed_vars.append(np.var(window))
                if windowed_vars:
                    values.append(np.mean(windowed_vars))
                else:
                    values.append(np.nan)
            else:
                values.append(np.nan)
        values = np.array(values)

    # Find longest linear region in log-log plot
    valid_mask = np.isfinite(values) & (values > 0)
    if np.sum(valid_mask) < 5:
        return [np.nan, np.nan], np.nan

    valid_lags = lags[valid_mask]
    valid_values = values[valid_mask]

    log_lags = np.log10(valid_lags)
    log_values = np.log10(valid_values)

    # Find best linear segment using sliding window
    best_r_squared = 0
    best_range = [np.nan, np.nan]
    min_points = 5

    for start in range(len(log_lags) - min_points + 1):
        for end in range(start + min_points, len(log_lags) + 1):
            if end - start >= min_points:
                x_segment = log_lags[start:end]
                y_segment = log_values[start:end]

                try:
                    slope, intercept, r_value, _, _ = stats.linregress(x_segment, y_segment)
                    r_squared = r_value**2

                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_range = [valid_lags[start], valid_lags[end-1]]
                except:
                    continue

    return best_range, best_r_squared


def extract_self_similarity_features(segments):
    """
    Extract comprehensive self-similarity features for AI parameter selection.

    Args:
        segments: Nx4 array of [x1, y1, x2, y2] segments

    Returns:
        features: Dictionary of self-similarity features
    """
    from .power_spectrum_features import segments_to_height_function

    # Convert to height function
    x, y = segments_to_height_function(segments, num_points=256)

    if len(x) < 10:
        return {
            'hurst_exponent_sf': np.nan,
            'hurst_exponent_va': np.nan,
            'hurst_exponent_dfa': np.nan,
            'hurst_consensus': np.nan,
            'hurst_confidence': np.nan,
            'scaling_range_min': np.nan,
            'scaling_range_max': np.nan,
            'scaling_quality': np.nan,
            'self_similarity_strength': np.nan,
            'fractal_character': 'insufficient_data',
            'predicted_dimension': np.nan
        }

    # Remove mean and normalize
    y_processed = y - np.mean(y)
    if np.std(y_processed) > 0:
        y_processed = y_processed / np.std(y_processed)

    # Calculate Hurst exponents using multiple methods
    hurst_sf, r2_sf = estimate_hurst_exponent(y_processed, method='structure_function')
    hurst_va, r2_va = estimate_hurst_exponent(y_processed, method='variance_aggregation')
    hurst_dfa, r2_dfa = estimate_hurst_exponent(y_processed, method='detrended_fluctuation')

    # Calculate consensus Hurst exponent
    valid_hursts = []
    weights = []

    if np.isfinite(hurst_sf) and np.isfinite(r2_sf):
        valid_hursts.append(hurst_sf)
        weights.append(r2_sf)

    if np.isfinite(hurst_va) and np.isfinite(r2_va):
        valid_hursts.append(hurst_va)
        weights.append(r2_va)

    if np.isfinite(hurst_dfa) and np.isfinite(r2_dfa):
        valid_hursts.append(hurst_dfa)
        weights.append(r2_dfa)

    if valid_hursts and weights:
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        hurst_consensus = np.average(valid_hursts, weights=weights)
        hurst_confidence = np.mean(weights)  # Average R² as confidence
    else:
        hurst_consensus = np.nan
        hurst_confidence = 0.0

    # Detect scaling range
    scaling_range, scaling_quality = detect_scaling_range(y_processed, 'structure_function')

    # Calculate self-similarity strength
    if np.isfinite(hurst_consensus) and np.isfinite(scaling_quality):
        # Strong self-similarity when H is away from 0.5 and scaling quality is high
        deviation_from_random = abs(hurst_consensus - 0.5)
        self_similarity_strength = deviation_from_random * scaling_quality
    else:
        self_similarity_strength = np.nan

    # Characterize fractal behavior
    if np.isfinite(hurst_consensus) and hurst_confidence > 0.7:
        if 0.8 < hurst_consensus < 1.2:
            fractal_character = 'strong_fractal'
        elif 0.3 < hurst_consensus < 0.7 or 1.2 < hurst_consensus < 1.7:
            fractal_character = 'moderate_fractal'
        elif 0.1 < hurst_consensus < 0.9 or 1.1 < hurst_consensus < 1.9:
            fractal_character = 'weak_fractal'
        else:
            fractal_character = 'non_fractal'
    else:
        fractal_character = 'uncertain'

    # Predict fractal dimension from Hurst exponent
    # For interface curves: D = 2 - H (for 1D curves embedded in 2D)
    if np.isfinite(hurst_consensus):
        predicted_dimension = 2.0 - hurst_consensus
        # Clamp to reasonable range for interface curves
        predicted_dimension = np.clip(predicted_dimension, 1.0, 2.0)
    else:
        predicted_dimension = np.nan

    return {
        'hurst_exponent_sf': hurst_sf,
        'hurst_exponent_va': hurst_va,
        'hurst_exponent_dfa': hurst_dfa,
        'hurst_consensus': hurst_consensus,
        'hurst_confidence': hurst_confidence,
        'scaling_range_min': scaling_range[0],
        'scaling_range_max': scaling_range[1],
        'scaling_quality': scaling_quality,
        'self_similarity_strength': self_similarity_strength,
        'fractal_character': fractal_character,
        'predicted_dimension': predicted_dimension
    }


def suggest_parameters_from_self_similarity(self_similarity_features, domain_scale=1.0):
    """
    Suggest box counting parameters based on self-similarity analysis.

    Key relationships:
    - hurst_exponent → predicted fractal dimension and scaling behavior
    - scaling_range → optimal δ₀ and δ_max
    - self_similarity_strength → delta_factor and num_steps
    - predicted_dimension → validation target

    Args:
        self_similarity_features: Features from extract_self_similarity_features()
        domain_scale: Characteristic domain scale for normalization

    Returns:
        suggested_params: Dictionary with initial_delta, delta_factor, num_steps
    """
    # Extract key features
    hurst_consensus = self_similarity_features.get('hurst_consensus', np.nan)
    hurst_confidence = self_similarity_features.get('hurst_confidence', 0.0)
    scaling_range_min = self_similarity_features.get('scaling_range_min', np.nan)
    scaling_range_max = self_similarity_features.get('scaling_range_max', np.nan)
    scaling_quality = self_similarity_features.get('scaling_quality', 0.0)
    self_similarity_strength = self_similarity_features.get('self_similarity_strength', 0.0)
    predicted_dimension = self_similarity_features.get('predicted_dimension', np.nan)

    # Handle NaN and invalid values
    if not np.isfinite(hurst_consensus):
        hurst_consensus = 0.8  # Default for moderate fractal
    if not np.isfinite(scaling_range_min):
        scaling_range_min = domain_scale * 0.01
    if not np.isfinite(scaling_range_max):
        scaling_range_max = domain_scale * 0.1
    if not np.isfinite(scaling_quality):
        scaling_quality = 0.5
    if not np.isfinite(self_similarity_strength):
        self_similarity_strength = 0.3

    # Parameter 1: Initial delta (from scaling range)
    # Start at the smaller end of the valid scaling range
    if scaling_range_min > 0 and scaling_range_max > scaling_range_min:
        # Use geometric mean of scaling range as starting point
        initial_delta = np.sqrt(scaling_range_min * scaling_range_max)
        # Adjust toward smaller end for better resolution
        initial_delta = initial_delta * 0.7
    else:
        # Fallback based on Hurst exponent
        if hurst_consensus > 0.8:
            initial_delta = domain_scale * 0.02  # Persistent signal → smaller boxes
        elif hurst_consensus < 0.3:
            initial_delta = domain_scale * 0.05  # Anti-persistent → larger boxes ok
        else:
            initial_delta = domain_scale * 0.03  # Moderate

    # Clamp to reasonable range
    min_delta = domain_scale * 0.001
    max_delta = domain_scale * 0.2
    initial_delta = np.clip(initial_delta, min_delta, max_delta)

    # Parameter 2: Delta factor (from self-similarity strength and Hurst)
    # Strong self-similarity allows larger scaling factors
    # Hurst close to 1 indicates strong persistence → good scaling

    base_factor = 1.5  # Default

    # Adjust for Hurst exponent
    if 0.7 <= hurst_consensus <= 1.3:
        hurst_factor = 1.2  # Good fractal scaling
    elif 0.5 <= hurst_consensus <= 1.5:
        hurst_factor = 1.1  # Moderate scaling
    else:
        hurst_factor = 0.9  # Weak scaling

    # Adjust for self-similarity strength
    if self_similarity_strength > 0.5:
        similarity_factor = 1.2  # Strong self-similarity
    elif self_similarity_strength > 0.2:
        similarity_factor = 1.1  # Moderate
    else:
        similarity_factor = 0.9  # Weak

    # Adjust for scaling quality
    if scaling_quality > 0.9:
        quality_factor = 1.1  # High quality scaling
    elif scaling_quality > 0.7:
        quality_factor = 1.0  # Good quality
    else:
        quality_factor = 0.95  # Lower quality

    delta_factor = base_factor * hurst_factor * similarity_factor * quality_factor
    delta_factor = np.clip(delta_factor, 1.1, 2.0)

    # Parameter 3: Number of steps (from scaling range and confidence)
    base_steps = 12

    # Calculate how many decades the scaling range spans
    if scaling_range_max > scaling_range_min > 0:
        range_decades = np.log10(scaling_range_max / scaling_range_min)
        # More decades → more steps needed
        if range_decades > 2.0:
            range_factor = 1.4
        elif range_decades > 1.0:
            range_factor = 1.2
        elif range_decades > 0.5:
            range_factor = 1.0
        else:
            range_factor = 0.8
    else:
        range_factor = 1.0

    # Confidence adjustment
    if hurst_confidence > 0.8:
        confidence_factor = 1.2  # High confidence → more steps for precision
    elif hurst_confidence > 0.6:
        confidence_factor = 1.1  # Moderate confidence
    else:
        confidence_factor = 0.9  # Low confidence → fewer steps

    # Self-similarity strength adjustment
    if self_similarity_strength > 0.4:
        strength_factor = 1.1  # Strong → more steps for better fit
    else:
        strength_factor = 1.0

    steps_factor = range_factor * confidence_factor * strength_factor
    num_steps = int(base_steps * steps_factor)
    num_steps = np.clip(num_steps, 8, 30)

    return {
        'initial_delta': initial_delta,
        'delta_factor': delta_factor,
        'num_steps': num_steps,
        'method': 'self_similarity_analysis',
        'confidence': calculate_self_similarity_confidence(self_similarity_features),
        'self_similarity_insights': {
            'hurst_exponent': hurst_consensus,
            'predicted_dimension': predicted_dimension,
            'scaling_range_decades': np.log10(scaling_range_max / scaling_range_min) if scaling_range_max > scaling_range_min > 0 else np.nan,
            'fractal_character': self_similarity_features.get('fractal_character', 'unknown'),
            'self_similarity_strength': self_similarity_strength
        }
    }


def calculate_self_similarity_confidence(self_similarity_features):
    """
    Calculate confidence in self-similarity-based parameter suggestions.

    High confidence when:
    - Consistent Hurst exponents across methods
    - Good scaling quality
    - Clear fractal character

    Args:
        self_similarity_features: Features from extract_self_similarity_features()

    Returns:
        confidence: Float 0-1 indicating confidence level
    """
    confidence = 1.0

    # Check Hurst exponent consistency
    hurst_sf = self_similarity_features.get('hurst_exponent_sf', np.nan)
    hurst_va = self_similarity_features.get('hurst_exponent_va', np.nan)
    hurst_dfa = self_similarity_features.get('hurst_exponent_dfa', np.nan)

    valid_hursts = [h for h in [hurst_sf, hurst_va, hurst_dfa] if np.isfinite(h)]

    if len(valid_hursts) < 2:
        confidence *= 0.6  # Few valid estimates
    elif len(valid_hursts) >= 2:
        hurst_std = np.std(valid_hursts)
        if hurst_std < 0.1:
            confidence *= 1.0  # Very consistent
        elif hurst_std < 0.2:
            confidence *= 0.9  # Reasonably consistent
        elif hurst_std < 0.3:
            confidence *= 0.8  # Moderately consistent
        else:
            confidence *= 0.6  # Inconsistent

    # Check scaling quality
    scaling_quality = self_similarity_features.get('scaling_quality', 0.0)
    if scaling_quality > 0.9:
        confidence *= 1.0  # Excellent scaling
    elif scaling_quality > 0.8:
        confidence *= 0.95  # Good scaling
    elif scaling_quality > 0.7:
        confidence *= 0.9  # Acceptable scaling
    else:
        confidence *= 0.7  # Poor scaling

    # Check Hurst confidence
    hurst_confidence = self_similarity_features.get('hurst_confidence', 0.0)
    confidence *= hurst_confidence  # Direct scaling

    # Check fractal character
    fractal_character = self_similarity_features.get('fractal_character', 'uncertain')
    if fractal_character == 'strong_fractal':
        confidence *= 1.0
    elif fractal_character == 'moderate_fractal':
        confidence *= 0.9
    elif fractal_character == 'weak_fractal':
        confidence *= 0.8
    else:
        confidence *= 0.6  # Non-fractal or uncertain

    # Check self-similarity strength
    similarity_strength = self_similarity_features.get('self_similarity_strength', 0.0)
    if np.isfinite(similarity_strength):
        if similarity_strength > 0.5:
            confidence *= 1.0  # Strong self-similarity
        elif similarity_strength > 0.3:
            confidence *= 0.95  # Moderate
        else:
            confidence *= 0.9  # Weak

    return np.clip(confidence, 0.0, 1.0)


def analyze_fractal_properties_from_self_similarity(self_similarity_features):
    """
    Comprehensive analysis of fractal properties from self-similarity features.

    Args:
        self_similarity_features: Features from extract_self_similarity_features()

    Returns:
        fractal_analysis: Dictionary with comprehensive fractal insights
    """
    hurst_consensus = self_similarity_features.get('hurst_consensus', np.nan)
    predicted_dimension = self_similarity_features.get('predicted_dimension', np.nan)
    fractal_character = self_similarity_features.get('fractal_character', 'unknown')
    scaling_range_min = self_similarity_features.get('scaling_range_min', np.nan)
    scaling_range_max = self_similarity_features.get('scaling_range_max', np.nan)

    # Classify fractal type
    if np.isfinite(hurst_consensus):
        if 0.95 <= hurst_consensus <= 1.05:
            fractal_type = 'brownian_motion'
            box_counting_behavior = 'standard_scaling'
        elif hurst_consensus > 1.05:
            fractal_type = 'persistent_fractal'
            box_counting_behavior = 'smooth_scaling'
        elif hurst_consensus < 0.95:
            fractal_type = 'antipersistent_fractal'
            box_counting_behavior = 'rough_scaling'
        else:
            fractal_type = 'unknown'
            box_counting_behavior = 'uncertain'
    else:
        fractal_type = 'unknown'
        box_counting_behavior = 'uncertain'

    # Estimate multifractal properties
    hurst_sf = self_similarity_features.get('hurst_exponent_sf', np.nan)
    hurst_va = self_similarity_features.get('hurst_exponent_va', np.nan)
    hurst_dfa = self_similarity_features.get('hurst_exponent_dfa', np.nan)

    valid_hursts = [h for h in [hurst_sf, hurst_va, hurst_dfa] if np.isfinite(h)]
    if len(valid_hursts) >= 2:
        hurst_variation = np.std(valid_hursts)
        if hurst_variation < 0.05:
            multifractal_character = 'monofractal'
        elif hurst_variation < 0.15:
            multifractal_character = 'weak_multifractal'
        else:
            multifractal_character = 'strong_multifractal'
    else:
        multifractal_character = 'unknown'

    # Box counting optimization suggestions
    if np.isfinite(scaling_range_min) and np.isfinite(scaling_range_max):
        optimal_box_range = [scaling_range_min * 0.5, scaling_range_max * 2.0]
        scaling_decades = np.log10(scaling_range_max / scaling_range_min)
    else:
        optimal_box_range = [np.nan, np.nan]
        scaling_decades = np.nan

    return {
        'fractal_type': fractal_type,
        'multifractal_character': multifractal_character,
        'box_counting_behavior': box_counting_behavior,
        'optimal_box_range': optimal_box_range,
        'scaling_decades': scaling_decades,
        'dimension_prediction': predicted_dimension,
        'hurst_variation': np.std(valid_hursts) if len(valid_hursts) >= 2 else np.nan,
        'analysis_confidence': self_similarity_features.get('hurst_confidence', 0.0),
        'fractal_insights': {
            'persistence': 'high' if hurst_consensus > 1.0 else 'low' if hurst_consensus < 0.5 else 'moderate',
            'roughness': 'high' if hurst_consensus < 0.5 else 'low' if hurst_consensus > 1.0 else 'moderate',
            'scaling_strength': self_similarity_features.get('self_similarity_strength', 0.0)
        }
    }