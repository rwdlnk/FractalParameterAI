# optimal_scaling.py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class OptimalScalingRegionSelector:
    """
    Implements the methodology for determining the optimal scaling region
    in box-counting fractal dimension calculations.
    
    As described in the paper "Optimal Scaling Region Selection for Box-Counting 
    Fractal Dimension Calculation", this class analyzes local slopes in the log-log plot
    to identify regions of consistent scaling behavior.
    """
    
    def __init__(self, window_width=5, sigma_thresh=0.05, min_region_length=5):
        """
        Initialize the selector with parameters for optimal scaling region selection.
        
        Args:
            window_width (int): Width of the window for calculating moving averages of local slopes
            sigma_thresh (float): Threshold for standard deviation to identify stable regions
            min_region_length (int): Minimum length of candidate scaling regions
        """
        self.window_width = window_width
        self.sigma_thresh = sigma_thresh
        self.min_region_length = min_region_length
    
    def select_optimal_region(self, box_sizes, box_counts, debug=False):
        """
        Select the optimal scaling region for fractal dimension calculation.
        
        Args:
            box_sizes (array-like): Array of box sizes
            box_counts (array-like): Array of box counts
            debug (bool): Whether to print debug information
        
        Returns:
            tuple: (optimal_region, optimal_dimension, standard_error, r_squared, local_slopes)
                optimal_region is a tuple of (start_idx, end_idx)
        """
        # Convert input to numpy arrays if they aren't already
        box_sizes = np.array(box_sizes)
        box_counts = np.array(box_counts)
        
        # Calculate log values
        log_box_sizes = np.log(1/box_sizes)
        log_box_counts = np.log(box_counts)
        
        # Calculate local slopes
        local_slopes = self._calculate_local_slopes(log_box_sizes, log_box_counts)
        
        if debug:
            print("Local slopes:")
            for i, slope in enumerate(local_slopes):
                print(f"  {i}: {slope:.4f}")
        
        # Calculate moving average and standard deviation
        moving_avg_slopes, std_dev_slopes = self._calculate_moving_statistics(local_slopes)
        
        # If not enough data for moving statistics
        if moving_avg_slopes is None:
            if debug:
                print("Not enough data points for moving statistics. Using entire range.")
            
            # Fallback to using the entire range
            slope, intercept, r_value, p_value, std_error = stats.linregress(
                log_box_sizes, log_box_counts)
            return None, slope, std_error, r_value**2, local_slopes
        
        if debug:
            print("\nMoving average slopes:")
            for i, avg in enumerate(moving_avg_slopes):
                print(f"  {i}: {avg:.4f} (std={std_dev_slopes[i]:.4f})")
        
        # Identify candidate regions
        candidate_regions = self._identify_candidate_regions(log_box_sizes, std_dev_slopes, debug)
        
        # If no candidate regions found, use the entire range
        if not candidate_regions:
            if debug:
                print("No candidate regions found. Using entire range.")
            
            slope, intercept, r_value, p_value, std_error = stats.linregress(
                log_box_sizes, log_box_counts)
            return None, slope, std_error, r_value**2, local_slopes
        
        # Evaluate candidate regions
        optimal_region, optimal_dimension, optimal_std_error, optimal_r_squared = (
            self._evaluate_candidate_regions(log_box_sizes, log_box_counts, candidate_regions, debug)
        )
        
        if debug:
            print(f"\nSelected optimal region: {optimal_region}")
            print(f"Optimal dimension: {optimal_dimension:.6f} ± {optimal_std_error:.6f}")
            print(f"R-squared: {optimal_r_squared:.6f}")
        
        return (optimal_region, optimal_dimension, optimal_std_error, 
                optimal_r_squared, local_slopes)
    
    def _calculate_local_slopes(self, log_box_sizes, log_box_counts):
        """
        Calculate local slopes between consecutive points in the log-log plot.
        
        Args:
            log_box_sizes (np.array): Log of 1/box_sizes
            log_box_counts (np.array): Log of box_counts
        
        Returns:
            np.array: Array of local slopes
        """
        local_slopes = np.zeros(len(log_box_sizes) - 1)
        for i in range(len(log_box_sizes) - 1):
            local_slopes[i] = ((log_box_counts[i+1] - log_box_counts[i]) / 
                              (log_box_sizes[i+1] - log_box_sizes[i]))
        return local_slopes
    
    def _calculate_moving_statistics(self, local_slopes):
        """
        Calculate moving average and standard deviation of local slopes.
        
        Args:
            local_slopes (np.array): Array of local slopes
        
        Returns:
            tuple: (moving_avg_slopes, std_dev_slopes)
        """
        w = self.window_width
        
        # Need at least window_width points for moving average
        if len(local_slopes) < w:
            return None, None
        
        # Calculate moving average and standard deviation
        moving_avg_slopes = np.zeros(len(local_slopes) - w + 1)
        std_dev_slopes = np.zeros(len(local_slopes) - w + 1)
        
        for i in range(len(moving_avg_slopes)):
            window = local_slopes[i:i+w]
            moving_avg_slopes[i] = np.mean(window)
            std_dev_slopes[i] = np.std(window, ddof=1)  # Use sample standard deviation
        
        return moving_avg_slopes, std_dev_slopes
    
    def _identify_candidate_regions(self, log_box_sizes, std_dev_slopes, debug=False):
        """
        Identify candidate scaling regions where std dev of slopes is below threshold.
        
        Args:
            log_box_sizes (np.array): Log of 1/box_sizes
            std_dev_slopes (np.array): Standard deviation of slopes
            debug (bool): Whether to print debug information
        
        Returns:
            list: List of candidate regions as (start_idx, end_idx) tuples
        """
        half_w = self.window_width // 2
        candidate_regions = []
        
        if debug:
            print("\nIdentifying candidate regions:")
            print(f"  Threshold: {self.sigma_thresh}")
            print(f"  Min region length: {self.min_region_length}")
        
        # Map std_dev_slopes indices back to original log_box_sizes indices
        start_idx = -1
        for i in range(len(std_dev_slopes)):
            # Adjusted index in the original log_box_sizes array
            orig_idx = i + half_w
            
            if std_dev_slopes[i] < self.sigma_thresh:
                if start_idx == -1:
                    start_idx = orig_idx
                    if debug:
                        print(f"  Found potential region start at index {start_idx}")
            else:
                if start_idx != -1:
                    end_idx = orig_idx
                    region_length = end_idx - start_idx + 1
                    if region_length >= self.min_region_length:
                        candidate_regions.append((start_idx, end_idx))
                        if debug:
                            print(f"  Added candidate region: {start_idx}-{end_idx} (length {region_length})")
                    elif debug:
                        print(f"  Rejected region {start_idx}-{orig_idx}: too short ({region_length} < {self.min_region_length})")
                    start_idx = -1
        
        # Check if we ended while still in a candidate region
        if start_idx != -1:
            end_idx = len(log_box_sizes) - half_w - 1
            region_length = end_idx - start_idx + 1
            if region_length >= self.min_region_length:
                candidate_regions.append((start_idx, end_idx))
                if debug:
                    print(f"  Added final candidate region: {start_idx}-{end_idx} (length {region_length})")
            elif debug:
                print(f"  Rejected final region {start_idx}-{end_idx}: too short ({region_length} < {self.min_region_length})")
        
        if debug:
            print(f"  Found {len(candidate_regions)} candidate regions")
        
        return candidate_regions
    
    def _evaluate_candidate_regions(self, log_box_sizes, log_box_counts, candidate_regions, debug=False):
        """
        Evaluate candidate regions based on length, R², and standard error.
        
        Args:
            log_box_sizes (np.array): Log of 1/box_sizes
            log_box_counts (np.array): Log of box_counts
            candidate_regions (list): List of candidate regions as (start_idx, end_idx) tuples
            debug (bool): Whether to print debug information
        
        Returns:
            tuple: (optimal_region, optimal_dimension, optimal_std_error, r_squared)
        """
        best_score = -1
        optimal_region = None
        optimal_dimension = None
        optimal_std_error = None
        optimal_r_squared = None
        
        if debug:
            print("\nEvaluating candidate regions:")
        
        for start, end in candidate_regions:
            x = log_box_sizes[start:end+1]
            y = log_box_counts[start:end+1]
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_error = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # Calculate region length in log space
            region_length = x[-1] - x[0]
            
            # Calculate score based on equation in the paper:
            # Score = (L_cand * R²_cand) / SE_cand
            score = region_length * r_squared / std_error
            
            if debug:
                print(f"  Region {start}-{end}: D={slope:.6f}, SE={std_error:.6f}, R²={r_squared:.6f}, Score={score:.6f}")
            
            if score > best_score:
                best_score = score
                optimal_region = (start, end)
                optimal_dimension = slope  # Convert slope to fractal dimension
                optimal_std_error = std_error
                optimal_r_squared = r_squared
                
                if debug:
                    print(f"    (New best score)")
        
        return optimal_region, optimal_dimension, optimal_std_error, optimal_r_squared
    
    def plot_analysis(self, box_sizes, box_counts, optimal_region=None, filename=None):
        """
        Create a detailed plot of the box-counting analysis including log-log plot,
        local slopes, and the optimal scaling region.
        
        Args:
            box_sizes (array-like): Array of box sizes
            box_counts (array-like): Array of box counts
            optimal_region (tuple): Optimal scaling region as (start_idx, end_idx)
            filename (str): Output filename for the plot
        
        Returns:
            tuple: (fig, (ax1, ax2)) - Figure and axes objects
        """
        # Convert input to numpy arrays if they aren't already
        box_sizes = np.array(box_sizes)
        box_counts = np.array(box_counts)
        
        # Calculate log values
        log_box_sizes = np.log(1/box_sizes)
        log_box_counts = np.log(box_counts)
        
        # Calculate local slopes
        local_slopes = self._calculate_local_slopes(log_box_sizes, log_box_counts)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot the log-log plot on the first subplot
        ax1.plot(log_box_sizes, log_box_counts, 'o-', color='blue', alpha=0.7)
        ax1.set_xlabel('log(1/ε)')
        ax1.set_ylabel('log(N(ε))')
        ax1.set_title('Log-Log Plot of Box Counting')
        ax1.grid(True, alpha=0.3)
        
        # Highlight the optimal region if provided
        if optimal_region is not None:
            start, end = optimal_region
            ax1.plot(log_box_sizes[start:end+1], log_box_counts[start:end+1], 'ro-', linewidth=2)
            
            # Calculate and plot the regression line for the optimal region
            slope, intercept, r_value, p_value, std_error = stats.linregress(
                log_box_sizes[start:end+1], log_box_counts[start:end+1]
            )
            x_range = np.array([log_box_sizes[start], log_box_sizes[end]])
            y_range = slope * x_range + intercept
            ax1.plot(x_range, y_range, 'r--', linewidth=1.5)
            
            # Add text with dimension and R²
            dimension_text = f"D = {slope:.4f} ± {std_error:.4f}\nR² = {r_value**2:.4f}"
            ax1.text(0.05, 0.95, dimension_text, transform=ax1.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        # Plot the local slopes on the second subplot
        x_centers = (log_box_sizes[:-1] + log_box_sizes[1:]) / 2
        ax2.plot(x_centers, local_slopes, 'o-', color='green', alpha=0.7)
        ax2.set_xlabel('log(1/ε)')
        ax2.set_ylabel('Local Slope')
        ax2.set_title('Local Slopes')
        ax2.grid(True, alpha=0.3)
        
        # Calculate moving average and standard deviation of local slopes
        moving_avg, std_dev = self._calculate_moving_statistics(local_slopes)

        if moving_avg is not None:
            # Plot moving average with error band
            half_w = self.window_width // 2
            # Make sure x_moving and moving_avg have the same length
            x_moving = x_centers[half_w:-(half_w-1) if half_w > 0 else None]
    
            # Verify lengths match
            if len(x_moving) > len(moving_avg):
                x_moving = x_moving[:len(moving_avg)]
            elif len(moving_avg) > len(x_moving):
                moving_avg = moving_avg[:len(x_moving)]

            # Also adjust std_dev to match the length of moving_avg
            if len(std_dev) != len(moving_avg):
                if len(std_dev) > len(moving_avg):
                    std_dev = std_dev[:len(moving_avg)]
                else:
                    # Use padding to extend std_dev to match moving_avg length
                    try:
                        # Use numpy's pad function
                        std_dev = np.pad(std_dev, (0, len(moving_avg) - len(std_dev)), 'edge')
                    except Exception as e:
                        # Simple padding by repeating the last value
                        last_value = std_dev[-1] if len(std_dev) > 0 else 0
                        std_dev = np.array(list(std_dev) + [last_value] * (len(moving_avg) - len(std_dev)))
         
            ax2.plot(x_moving, moving_avg, 'b-', linewidth=2, alpha=0.7, label='Moving Avg')
            ax2.fill_between(x_moving, moving_avg - std_dev, moving_avg + std_dev, 
                       color='blue', alpha=0.2, label='±1σ')
            
            # Add threshold line
            ax2.axhline(y=self.sigma_thresh, color='r', linestyle='--', 
                       alpha=0.5, label=f'σ threshold: {self.sigma_thresh:.2f}')
            
            # Highlight the optimal region in the local slopes plot
            if optimal_region is not None:
                # Adjust indices for the local slopes array (which is one shorter)
                start_slope = max(0, optimal_region[0] - 1)
                end_slope = min(len(local_slopes) - 1, optimal_region[1] - 1)
                ax2.plot(x_centers[start_slope:end_slope+1], local_slopes[start_slope:end_slope+1], 
                       'ro-', linewidth=2)
                
                # Add horizontal line at the average slope in the optimal region
                avg_slope = np.mean(local_slopes[start_slope:end_slope+1])
                ax2.axhline(y=avg_slope, color='r', linestyle='--', alpha=0.7)
                ax2.text(x_centers[start_slope], avg_slope*1.02, 
                       f"Avg slope: {avg_slope:.4f}", color='r')
        
        ax2.legend()
        plt.tight_layout()
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")
        
        return fig, (ax1, ax2)

