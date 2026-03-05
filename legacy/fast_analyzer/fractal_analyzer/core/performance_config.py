"""
Performance configuration for fractal analysis.

Provides performance modes and configuration settings for balancing
speed vs accuracy in fractal dimension calculations.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Any


class PerformanceMode(Enum):
    """Performance modes for fractal analysis."""
    FAST = auto()       # Fast but less accurate
    BALANCED = auto()   # Balance of speed and accuracy
    ACCURATE = auto()   # Slow but most accurate
    RESEARCH = auto()   # Research-quality with full detail


@dataclass
class PerformanceConfig:
    """Configuration settings for performance modes."""

    # Box counting parameters
    initial_delta_factor: float = 0.02
    delta_factor: float = 1.5
    num_steps: int = 25

    # Quality thresholds
    min_r_squared: float = 0.95
    min_scaling_decades: float = 1.0

    # Optimization flags
    grid_optimization: bool = True
    boundary_artifact_removal: bool = False

    # Timeout settings
    timeout_seconds: Optional[float] = None

    @classmethod
    def from_mode(cls, mode: PerformanceMode) -> 'PerformanceConfig':
        """Create configuration from performance mode."""

        if mode == PerformanceMode.FAST:
            return cls(
                initial_delta_factor=0.05,
                delta_factor=2.0,
                num_steps=15,
                min_r_squared=0.90,
                min_scaling_decades=0.8,
                grid_optimization=True,
                boundary_artifact_removal=False,
                timeout_seconds=60.0
            )

        elif mode == PerformanceMode.BALANCED:
            return cls(
                initial_delta_factor=0.02,
                delta_factor=1.5,
                num_steps=25,
                min_r_squared=0.95,
                min_scaling_decades=1.0,
                grid_optimization=True,
                boundary_artifact_removal=True,
                timeout_seconds=180.0
            )

        elif mode == PerformanceMode.ACCURATE:
            return cls(
                initial_delta_factor=0.01,
                delta_factor=1.3,
                num_steps=35,
                min_r_squared=0.98,
                min_scaling_decades=1.2,
                grid_optimization=True,
                boundary_artifact_removal=True,
                timeout_seconds=600.0
            )

        elif mode == PerformanceMode.RESEARCH:
            return cls(
                initial_delta_factor=0.005,
                delta_factor=1.2,
                num_steps=50,
                min_r_squared=0.99,
                min_scaling_decades=1.5,
                grid_optimization=True,
                boundary_artifact_removal=True,
                timeout_seconds=None
            )

        else:
            return cls()  # Default configuration

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'initial_delta_factor': self.initial_delta_factor,
            'delta_factor': self.delta_factor,
            'num_steps': self.num_steps,
            'min_r_squared': self.min_r_squared,
            'min_scaling_decades': self.min_scaling_decades,
            'grid_optimization': self.grid_optimization,
            'boundary_artifact_removal': self.boundary_artifact_removal,
            'timeout_seconds': self.timeout_seconds
        }


# Default configurations for each mode
DEFAULT_CONFIGS = {
    PerformanceMode.FAST: PerformanceConfig.from_mode(PerformanceMode.FAST),
    PerformanceMode.BALANCED: PerformanceConfig.from_mode(PerformanceMode.BALANCED),
    PerformanceMode.ACCURATE: PerformanceConfig.from_mode(PerformanceMode.ACCURATE),
    PerformanceMode.RESEARCH: PerformanceConfig.from_mode(PerformanceMode.RESEARCH),
}


def get_performance_config(mode: PerformanceMode) -> PerformanceConfig:
    """Get performance configuration for the specified mode."""
    return DEFAULT_CONFIGS.get(mode, PerformanceConfig())