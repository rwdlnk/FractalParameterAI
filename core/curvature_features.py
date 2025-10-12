#!/usr/bin/env python3
"""
Curvature Statistics for Local Geometric Complexity Analysis

Analyzes local curvature properties of interfaces to predict optimal box
counting parameters. High curvature regions require finer resolution, while
smooth regions can use coarser boxes.

Key insight: Curvature distribution → optimal δ₀ and step progression
"""

import numpy as np
from scipy import ndimage
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


def calculate_discrete_curvature(x, y, method='finite_difference'):
    """
    Calculate discrete curvature for a parametric curve.

    Args:
        x, y: Arrays of curve coordinates
        method: 'finite_difference', 'three_point', or 'fit_circle'

    Returns:
        curvature: Array of curvature values
    """
    if len(x) < 3 or len(y) < 3:
        return np.array([])

    if method == 'finite_difference':
        # First and second derivatives using finite differences
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2)

        # Avoid division by zero
        denominator[denominator < 1e-10] = 1e-10
        curvature = numerator / denominator

    elif method == 'three_point':
        # Three-point curvature calculation
        curvature = np.zeros(len(x))

        for i in range(1, len(x) - 1):
            # Use three consecutive points
            x1, y1 = x[i-1], y[i-1]
            x2, y2 = x[i], y[i]
            x3, y3 = x[i+1], y[i+1]

            # Calculate area of triangle
            area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

            # Calculate side lengths
            a = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            b = np.sqrt((x3-x2)**2 + (y3-y2)**2)
            c = np.sqrt((x3-x1)**2 + (y3-y1)**2)

            # Curvature = 4*Area / (a*b*c)
            if a > 0 and b > 0 and c > 0:
                curvature[i] = 4 * area / (a * b * c)

        # Handle endpoints
        curvature[0] = curvature[1] if len(curvature) > 1 else 0
        curvature[-1] = curvature[-2] if len(curvature) > 1 else 0

    elif method == 'fit_circle':
        # Fit local circles and use 1/radius as curvature
        curvature = np.zeros(len(x))
        window_size = 5

        for i in range(len(x)):
            start = max(0, i - window_size//2)
            end = min(len(x), i + window_size//2 + 1)

            if end - start >= 3:
                x_local = x[start:end]
                y_local = y[start:end]

                # Fit circle using least squares
                try:
                    # Set up system: (x-a)² + (y-b)² = r²
                    # Expand: x² + y² - 2ax - 2by + a² + b² - r² = 0
                    A = np.column_stack([2*x_local, 2*y_local, np.ones(len(x_local))])
                    b_vec = x_local**2 + y_local**2

                    if np.linalg.matrix_rank(A) >= 3:
                        params = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                        a, b, c = params
                        radius = np.sqrt(a**2 + b**2 + c)

                        if radius > 1e-10:
                            curvature[i] = 1.0 / radius
                except:
                    pass

    else:
        raise ValueError(f"Unknown curvature method: {method}")

    return curvature


def extract_curvature_features(segments, method='finite_difference'):
    """
    Extract comprehensive curvature-based features for AI parameter selection.

    Args:
        segments: Nx4 array of [x1, y1, x2, y2] segments
        method: Curvature calculation method

    Returns:
        features: Dictionary of curvature-based features
    """
    if len(segments) == 0:
        return {
            'mean_curvature': np.nan,
            'curvature_variance': np.nan,
            'max_curvature': np.nan,
            'curvature_range': np.nan,
            'high_curvature_fraction': np.nan,
            'curvature_skewness': np.nan,
            'curvature_kurtosis': np.nan,
            'curvature_percentiles': np.array([]),
            'local_complexity_variation': np.nan,
            'smooth_region_fraction': np.nan
        }

    # Extract points from segments
    points = []
    for seg in segments:
        points.extend([(seg[0], seg[1]), (seg[2], seg[3])])

    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for point in points:
        if point not in seen:
            unique_points.append(point)
            seen.add(point)

    if len(unique_points) < 3:
        return {
            'mean_curvature': np.nan,
            'curvature_variance': np.nan,
            'max_curvature': np.nan,
            'curvature_range': np.nan,
            'high_curvature_fraction': np.nan,
            'curvature_skewness': np.nan,
            'curvature_kurtosis': np.nan,
            'curvature_percentiles': np.array([]),
            'local_complexity_variation': np.nan,
            'smooth_region_fraction': np.nan
        }

    # Sort points by x-coordinate (assuming roughly x-ordered interface)
    unique_points.sort(key=lambda p: p[0])
    x = np.array([p[0] for p in unique_points])
    y = np.array([p[1] for p in unique_points])

    # Calculate curvature
    curvature = calculate_discrete_curvature(x, y, method)

    if len(curvature) == 0:
        return {
            'mean_curvature': np.nan,
            'curvature_variance': np.nan,
            'max_curvature': np.nan,
            'curvature_range': np.nan,
            'high_curvature_fraction': np.nan,
            'curvature_skewness': np.nan,
            'curvature_kurtosis': np.nan,
            'curvature_percentiles': np.array([]),
            'local_complexity_variation': np.nan,
            'smooth_region_fraction': np.nan
        }

    # Remove infinite and NaN values
    finite_curvature = curvature[np.isfinite(curvature)]
    if len(finite_curvature) == 0:
        return {
            'mean_curvature': np.nan,
            'curvature_variance': np.nan,
            'max_curvature': np.nan,
            'curvature_range': np.nan,
            'high_curvature_fraction': np.nan,
            'curvature_skewness': np.nan,
            'curvature_kurtosis': np.nan,
            'curvature_percentiles': np.array([]),
            'local_complexity_variation': np.nan,
            'smooth_region_fraction': np.nan
        }

    # Feature 1: Basic statistics
    mean_curvature = np.mean(finite_curvature)
    curvature_variance = np.var(finite_curvature)
    max_curvature = np.max(finite_curvature)
    curvature_range = max_curvature - np.min(finite_curvature)

    # Feature 2: High curvature fraction
    # Define high curvature as > 75th percentile
    curvature_75th = np.percentile(finite_curvature, 75)
    high_curvature_fraction = np.sum(finite_curvature > curvature_75th) / len(finite_curvature)

    # Feature 3: Distribution shape
    curvature_skewness = skew(finite_curvature) if len(finite_curvature) > 2 else np.nan
    curvature_kurtosis_val = kurtosis(finite_curvature) if len(finite_curvature) > 3 else np.nan

    # Feature 4: Percentiles for distribution characterization
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    curvature_percentiles = np.percentile(finite_curvature, percentiles)

    # Feature 5: Local complexity variation
    # Measure how much curvature varies locally
    if len(finite_curvature) > 5:
        # Calculate local standard deviation using sliding window
        window_size = min(5, len(finite_curvature) // 3)
        local_stds = []

        for i in range(len(finite_curvature) - window_size + 1):
            window = finite_curvature[i:i + window_size]
            local_stds.append(np.std(window))

        local_complexity_variation = np.mean(local_stds) if local_stds else np.nan
    else:
        local_complexity_variation = curvature_variance

    # Feature 6: Smooth region fraction
    # Define smooth regions as having low curvature (< 25th percentile)
    curvature_25th = np.percentile(finite_curvature, 25)
    smooth_region_fraction = np.sum(finite_curvature < curvature_25th) / len(finite_curvature)

    return {
        'mean_curvature': mean_curvature,
        'curvature_variance': curvature_variance,
        'max_curvature': max_curvature,
        'curvature_range': curvature_range,
        'high_curvature_fraction': high_curvature_fraction,
        'curvature_skewness': curvature_skewness,
        'curvature_kurtosis': curvature_kurtosis_val,
        'curvature_percentiles': curvature_percentiles,
        'local_complexity_variation': local_complexity_variation,
        'smooth_region_fraction': smooth_region_fraction
    }


def suggest_parameters_from_curvature(curvature_features, domain_scale=1.0):
    """
    Suggest box counting parameters based on curvature analysis.

    Key relationships:
    - high mean_curvature → smaller initial_delta
    - high curvature_variance → more steps needed
    - high_curvature_fraction → smaller delta_factor for finer resolution
    - smooth_region_fraction → can use larger initial boxes

    Args:
        curvature_features: Features from extract_curvature_features()
        domain_scale: Characteristic domain scale for normalization

    Returns:
        suggested_params: Dictionary with initial_delta, delta_factor, num_steps
    """
    # Extract key features
    mean_curvature = curvature_features.get('mean_curvature', 0.0)
    curvature_variance = curvature_features.get('curvature_variance', 0.0)
    max_curvature = curvature_features.get('max_curvature', 0.0)
    high_curvature_fraction = curvature_features.get('high_curvature_fraction', 0.0)
    smooth_region_fraction = curvature_features.get('smooth_region_fraction', 0.5)
    local_complexity_variation = curvature_features.get('local_complexity_variation', 0.0)
    curvature_percentiles = curvature_features.get('curvature_percentiles', np.array([]))

    # Handle NaN and invalid values
    if not np.isfinite(mean_curvature):
        mean_curvature = 1.0 / domain_scale  # Default curvature
    if not np.isfinite(curvature_variance):
        curvature_variance = 0.1
    if not np.isfinite(max_curvature):
        max_curvature = 2.0 / domain_scale
    if not np.isfinite(high_curvature_fraction):
        high_curvature_fraction = 0.25
    if not np.isfinite(smooth_region_fraction):
        smooth_region_fraction = 0.5
    if not np.isfinite(local_complexity_variation):
        local_complexity_variation = 0.1

    # Parameter 1: Initial delta (from curvature characteristics)
    # High curvature requires smaller boxes for accurate capture
    # Use characteristic radius of curvature as guide

    if mean_curvature > 0:
        # Characteristic radius of curvature
        characteristic_radius = 1.0 / mean_curvature

        # Initial box should be smaller than radius for good resolution
        initial_delta = characteristic_radius * 0.5

        # Adjust based on smooth regions
        if smooth_region_fraction > 0.7:
            initial_delta *= 2.0  # Can use larger boxes in smooth regions
        elif smooth_region_fraction < 0.3:
            initial_delta *= 0.5  # Need smaller boxes if mostly curved

    else:
        # No significant curvature, use moderate box size
        initial_delta = domain_scale * 0.05

    # Clamp to reasonable range
    min_delta = domain_scale * 0.001  # 0.1% of domain
    max_delta = domain_scale * 0.2    # 20% of domain
    initial_delta = np.clip(initial_delta, min_delta, max_delta)

    # Parameter 2: Delta factor (from curvature variation)
    # High variation in curvature requires more gradual scaling

    # Base factor from curvature distribution
    if max_curvature > 0 and mean_curvature > 0:
        curvature_ratio = max_curvature / mean_curvature
        if curvature_ratio > 10:
            base_factor = 1.3  # High variation → smaller steps
        elif curvature_ratio > 5:
            base_factor = 1.4  # Moderate variation
        elif curvature_ratio > 2:
            base_factor = 1.5  # Low variation
        else:
            base_factor = 1.7  # Very uniform → larger steps ok
    else:
        base_factor = 1.5  # Default

    # Adjust for high curvature fraction
    if high_curvature_fraction > 0.5:
        delta_factor = base_factor * 0.9  # Lots of high curvature → smaller factor
    elif high_curvature_fraction < 0.2:
        delta_factor = base_factor * 1.1  # Little high curvature → larger factor ok
    else:
        delta_factor = base_factor

    # Adjust for local complexity variation
    if local_complexity_variation > curvature_variance * 0.5:
        delta_factor *= 0.95  # High local variation → slightly smaller factor

    # Clamp factor
    delta_factor = np.clip(delta_factor, 1.1, 2.0)

    # Parameter 3: Number of steps (from curvature complexity)
    base_steps = 12

    # Variance adjustment (high variance → more steps to capture all scales)
    if curvature_variance > mean_curvature**2:
        variance_factor = 1.4  # High variance
    elif curvature_variance > mean_curvature**2 * 0.5:
        variance_factor = 1.2  # Moderate variance
    else:
        variance_factor = 1.0  # Low variance

    # High curvature fraction adjustment
    if high_curvature_fraction > 0.4:
        hcf_factor = 1.3  # Lots of complex regions
    elif high_curvature_fraction > 0.2:
        hcf_factor = 1.1  # Some complex regions
    else:
        hcf_factor = 0.9  # Mostly simple

    # Local complexity adjustment
    if local_complexity_variation > curvature_variance:
        complexity_factor = 1.2  # High local variation
    else:
        complexity_factor = 1.0

    # Combine factors
    steps_factor = variance_factor * hcf_factor * complexity_factor
    num_steps = int(base_steps * steps_factor)
    num_steps = np.clip(num_steps, 8, 25)  # Reasonable range

    return {
        'initial_delta': initial_delta,
        'delta_factor': delta_factor,
        'num_steps': num_steps,
        'method': 'curvature_analysis',
        'confidence': calculate_curvature_confidence(curvature_features),
        'curvature_insights': {
            'mean_curvature': mean_curvature,
            'characteristic_radius': 1.0/mean_curvature if mean_curvature > 0 else np.inf,
            'curvature_ratio': max_curvature/mean_curvature if mean_curvature > 0 else np.inf,
            'high_curvature_fraction': high_curvature_fraction,
            'smooth_region_fraction': smooth_region_fraction
        }
    }


def calculate_curvature_confidence(curvature_features):
    """
    Calculate confidence in curvature-based parameter suggestions.

    High confidence when:
    - Meaningful curvature values computed
    - Good distribution of curvature
    - Reasonable statistics

    Args:
        curvature_features: Features from extract_curvature_features()

    Returns:
        confidence: Float 0-1 indicating confidence level
    """
    confidence = 1.0

    # Check for NaN features
    feature_values = [
        curvature_features.get('mean_curvature', np.nan),
        curvature_features.get('curvature_variance', np.nan),
        curvature_features.get('max_curvature', np.nan),
        curvature_features.get('high_curvature_fraction', np.nan),
        curvature_features.get('smooth_region_fraction', np.nan),
        curvature_features.get('local_complexity_variation', np.nan)
    ]

    nan_count = sum(1 for v in feature_values if not np.isfinite(v))
    confidence *= (1.0 - 0.15 * nan_count)  # -15% per NaN feature

    # Check curvature range reasonableness
    mean_curvature = curvature_features.get('mean_curvature', np.nan)
    max_curvature = curvature_features.get('max_curvature', np.nan)

    if np.isfinite(mean_curvature) and np.isfinite(max_curvature):
        if mean_curvature > 0:
            curvature_ratio = max_curvature / mean_curvature
            if 1.0 <= curvature_ratio <= 100:
                confidence *= 1.0  # Reasonable ratio
            elif curvature_ratio > 1000:
                confidence *= 0.6  # Very high ratio, possibly noisy
            else:
                confidence *= 0.8  # Unusual ratio

    # Check percentiles availability
    percentiles = curvature_features.get('curvature_percentiles', np.array([]))
    if len(percentiles) < 5:
        confidence *= 0.7  # Incomplete percentile information

    # Check distribution reasonableness
    high_frac = curvature_features.get('high_curvature_fraction', np.nan)
    smooth_frac = curvature_features.get('smooth_region_fraction', np.nan)

    if np.isfinite(high_frac) and np.isfinite(smooth_frac):
        total_frac = high_frac + smooth_frac
        if 0.3 <= total_frac <= 1.0:
            confidence *= 1.0  # Reasonable distribution
        else:
            confidence *= 0.8  # Unusual distribution

    return np.clip(confidence, 0.0, 1.0)


def analyze_geometric_complexity_from_curvature(curvature_features):
    """
    Analyze geometric complexity characteristics from curvature distribution.

    Args:
        curvature_features: Features from extract_curvature_features()

    Returns:
        complexity_analysis: Dictionary with geometric complexity insights
    """
    mean_curvature = curvature_features.get('mean_curvature', np.nan)
    curvature_variance = curvature_features.get('curvature_variance', np.nan)
    max_curvature = curvature_features.get('max_curvature', np.nan)
    skewness = curvature_features.get('curvature_skewness', np.nan)
    kurtosis_val = curvature_features.get('curvature_kurtosis', np.nan)
    smooth_fraction = curvature_features.get('smooth_region_fraction', np.nan)

    # Classify geometric complexity
    if not np.isfinite(mean_curvature):
        complexity_class = 'unknown'
        characteristic_scale = np.nan
    elif mean_curvature < 0.1:
        complexity_class = 'very_smooth'
        characteristic_scale = np.inf
    elif mean_curvature < 1.0:
        complexity_class = 'smooth'
        characteristic_scale = 1.0 / mean_curvature
    elif mean_curvature < 10.0:
        complexity_class = 'moderate_complexity'
        characteristic_scale = 1.0 / mean_curvature
    elif mean_curvature < 100.0:
        complexity_class = 'high_complexity'
        characteristic_scale = 1.0 / mean_curvature
    else:
        complexity_class = 'very_high_complexity'
        characteristic_scale = 1.0 / mean_curvature

    # Analyze distribution characteristics
    if np.isfinite(skewness):
        if skewness > 2.0:
            distribution_type = 'highly_skewed_right'
        elif skewness > 0.5:
            distribution_type = 'skewed_right'
        elif skewness > -0.5:
            distribution_type = 'symmetric'
        elif skewness > -2.0:
            distribution_type = 'skewed_left'
        else:
            distribution_type = 'highly_skewed_left'
    else:
        distribution_type = 'unknown'

    # Box counting implications
    if np.isfinite(mean_curvature) and mean_curvature > 0:
        suggested_min_box_size = min(0.1 / mean_curvature, 1.0)
        suggested_max_box_size = min(1.0 / mean_curvature, 10.0)
    else:
        suggested_min_box_size = 0.01
        suggested_max_box_size = 1.0

    return {
        'complexity_class': complexity_class,
        'characteristic_scale': characteristic_scale,
        'distribution_type': distribution_type,
        'uniformity': 1.0 / (1.0 + curvature_variance) if np.isfinite(curvature_variance) else np.nan,
        'suggested_min_box_size': suggested_min_box_size,
        'suggested_max_box_size': suggested_max_box_size,
        'geometric_insights': {
            'smooth_dominated': smooth_fraction > 0.6 if np.isfinite(smooth_fraction) else False,
            'complex_dominated': smooth_fraction < 0.3 if np.isfinite(smooth_fraction) else False,
            'mixed_complexity': 0.3 <= smooth_fraction <= 0.6 if np.isfinite(smooth_fraction) else True
        }
    }