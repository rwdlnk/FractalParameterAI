import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk
from scipy import stats


class KochCurveAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Koch Curve Box-Counting Analyzer")
        self.root.geometry("1400x900")
        
        # Default parameters
        self.iterations = tk.IntVar(value=3)
        self.start_x = tk.DoubleVar(value=0.0)
        self.start_y = tk.DoubleVar(value=0.0)
        self.end_x = tk.DoubleVar(value=3.0)
        self.end_y = tk.DoubleVar(value=0.0)
        self.initial_delta = tk.DoubleVar(value=0.3)
        self.delta_factor = tk.DoubleVar(value=1.5)
        self.num_steps = tk.IntVar(value=10)
        
        self.setup_gui()
        self.update_analysis()
    
    def generate_koch_curve(self, p1, p2, iterations):
        """Generate Koch curve segments recursively."""
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
        segments.extend(self.generate_koch_curve(p1, p1_3, iterations - 1))
        segments.extend(self.generate_koch_curve(p1_3, peak, iterations - 1))
        segments.extend(self.generate_koch_curve(peak, p2_3, iterations - 1))
        segments.extend(self.generate_koch_curve(p2_3, p2, iterations - 1))
        
        return segments
    
    def segment_intersects_box(self, seg, box_min, box_max):
        """Check if segment intersects box."""
        x1, y1, x2, y2 = seg
        xmin, ymin = box_min
        xmax, ymax = box_max
        
        # Quick rejection
        if (x1 < xmin and x2 < xmin) or (x1 > xmax and x2 > xmax):
            return False
        if (y1 < ymin and y2 < ymin) or (y1 > ymax and y2 > ymax):
            return False
        
        # Check endpoints
        if xmin <= x1 <= xmax and ymin <= y1 <= ymax:
            return True
        if xmin <= x2 <= xmax and ymin <= y2 <= ymax:
            return True
        
        # Parametric intersection tests
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > 1e-10:
            t = (xmin - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if ymin <= y <= ymax:
                    return True
            t = (xmax - x1) / dx
            if 0 <= t <= 1:
                y = y1 + t * dy
                if ymin <= y <= ymax:
                    return True
        
        if abs(dy) > 1e-10:
            t = (ymin - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if xmin <= x <= xmax:
                    return True
            t = (ymax - y1) / dy
            if 0 <= t <= 1:
                x = x1 + t * dx
                if xmin <= x <= xmax:
                    return True
        
        return False
    
    def count_boxes(self, segments, delta, domain_x, domain_y):
        """Count boxes intersecting the curve."""
        xmin, xmax = domain_x
        ymin, ymax = domain_y
        
        nx = int(np.ceil((xmax - xmin) / delta))
        ny = int(np.ceil((ymax - ymin) / delta))
        
        boxes = set()
        
        for seg in segments:
            for i in range(nx):
                for j in range(ny):
                    box_min = [xmin + i * delta, ymin + j * delta]
                    box_max = [xmin + (i + 1) * delta, ymin + (j + 1) * delta]
                    
                    if self.segment_intersects_box(seg, box_min, box_max):
                        boxes.add((i, j))
        
        return len(boxes)
    
    def perform_box_counting(self, segments, domain_x, domain_y):
        """Perform box-counting analysis."""
        # Generate deltas
        deltas = []
        delta = self.initial_delta.get()
        for _ in range(self.num_steps.get()):
            deltas.append(delta)
            delta = delta / self.delta_factor.get()
        
        deltas = np.array(deltas)
        
        # Count boxes
        n_boxes = np.array([
            self.count_boxes(segments, d, domain_x, domain_y) 
            for d in deltas
        ])
        
        # Calculate logs
        log_inv_delta = np.log(1 / deltas)
        log_n_boxes = np.log(n_boxes)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_inv_delta, log_n_boxes
        )
        
        return {
            'deltas': deltas,
            'n_boxes': n_boxes,
            'log_inv_delta': log_inv_delta,
            'log_n_boxes': log_n_boxes,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'std_err': std_err
        }
    
    def setup_gui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Koch Curve Box-Counting Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=10)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Koch curve parameters
        koch_frame = ttk.Frame(params_frame)
        koch_frame.grid(row=0, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(koch_frame, text="Koch Curve Generation", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=6, sticky=tk.W)
        
        # Iterations
        ttk.Label(koch_frame, text="Iterations:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.iter_label = ttk.Label(koch_frame, text=str(self.iterations.get()))
        self.iter_label.grid(row=1, column=1, sticky=tk.W)
        iter_scale = ttk.Scale(koch_frame, from_=0, to=6, variable=self.iterations,
                              command=self.on_param_change, orient=tk.HORIZONTAL, length=200)
        iter_scale.grid(row=1, column=2, padx=5)
        self.segments_label = ttk.Label(koch_frame, text=f"Segments: {4**self.iterations.get()}")
        self.segments_label.grid(row=1, column=3, padx=10)
        
        # Start/End points
        ttk.Label(koch_frame, text="Start X:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(koch_frame, textvariable=self.start_x, width=8).grid(row=2, column=1)
        ttk.Label(koch_frame, text="Start Y:").grid(row=2, column=2, sticky=tk.W, padx=5)
        ttk.Entry(koch_frame, textvariable=self.start_y, width=8).grid(row=2, column=3)
        
        ttk.Label(koch_frame, text="End X:").grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Entry(koch_frame, textvariable=self.end_x, width=8).grid(row=3, column=1)
        ttk.Label(koch_frame, text="End Y:").grid(row=3, column=2, sticky=tk.W, padx=5)
        ttk.Entry(koch_frame, textvariable=self.end_y, width=8).grid(row=3, column=3)
        
        # Box-counting parameters
        box_frame = ttk.Frame(params_frame)
        box_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(box_frame, text="Box-Counting Parameters", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=6, sticky=tk.W)
        
        # Initial delta
        ttk.Label(box_frame, text="Initial δ:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.delta_label = ttk.Label(box_frame, text=f"{self.initial_delta.get():.3f}")
        self.delta_label.grid(row=1, column=1, sticky=tk.W)
        delta_scale = ttk.Scale(box_frame, from_=0.1, to=1.0, variable=self.initial_delta,
                               command=self.on_param_change, orient=tk.HORIZONTAL, length=200)
        delta_scale.grid(row=1, column=2, padx=5)
        
        # Delta factor
        ttk.Label(box_frame, text="δ Factor:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.factor_label = ttk.Label(box_frame, text=f"{self.delta_factor.get():.2f}")
        self.factor_label.grid(row=2, column=1, sticky=tk.W)
        factor_scale = ttk.Scale(box_frame, from_=1.2, to=2.5, variable=self.delta_factor,
                                command=self.on_param_change, orient=tk.HORIZONTAL, length=200)
        factor_scale.grid(row=2, column=2, padx=5)
        
        # Num steps
        ttk.Label(box_frame, text="Steps:").grid(row=3, column=0, sticky=tk.W, padx=5)
        self.steps_label = ttk.Label(box_frame, text=str(self.num_steps.get()))
        self.steps_label.grid(row=3, column=1, sticky=tk.W)
        steps_scale = ttk.Scale(box_frame, from_=5, to=15, variable=self.num_steps,
                               command=self.on_param_change, orient=tk.HORIZONTAL, length=200)
        steps_scale.grid(row=3, column=2, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(params_frame)
        button_frame.grid(row=2, column=0, columnspan=6, pady=10)
        
        ttk.Button(button_frame, text="Reset Default", 
                  command=self.reset_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="- Iteration", 
                  command=self.decrease_iteration).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="+ Iteration", 
                  command=self.increase_iteration).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Update", 
                  command=self.update_analysis).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.measured_dim_label = ttk.Label(results_frame, text="Measured D: --", 
                                           font=('Arial', 12, 'bold'))
        self.measured_dim_label.grid(row=0, column=0, padx=20)
        
        theoretical = np.log(4) / np.log(3)
        self.theoretical_label = ttk.Label(results_frame, 
                                          text=f"Theoretical D: {theoretical:.6f}", 
                                          font=('Arial', 12, 'bold'))
        self.theoretical_label.grid(row=0, column=1, padx=20)
        
        self.error_label = ttk.Label(results_frame, text="Error: --", 
                                    font=('Arial', 12, 'bold'))
        self.error_label.grid(row=0, column=2, padx=20)
        
        self.r2_label = ttk.Label(results_frame, text="R²: --", 
                                 font=('Arial', 12, 'bold'))
        self.r2_label.grid(row=0, column=3, padx=20)
        
        # Plot frame
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(3, weight=1)
        
        # Create figure with two subplots
        self.fig = Figure(figsize=(14, 5))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_param_change(self, event=None):
        """Update labels when sliders change."""
        self.iter_label.config(text=str(int(self.iterations.get())))
        self.segments_label.config(text=f"Segments: {4**int(self.iterations.get())}")
        self.delta_label.config(text=f"{self.initial_delta.get():.3f}")
        self.factor_label.config(text=f"{self.delta_factor.get():.2f}")
        self.steps_label.config(text=str(int(self.num_steps.get())))
    
    def reset_defaults(self):
        """Reset to default parameters."""
        self.iterations.set(3)
        self.start_x.set(0.0)
        self.start_y.set(0.0)
        self.end_x.set(3.0)
        self.end_y.set(0.0)
        self.initial_delta.set(0.3)
        self.delta_factor.set(1.5)
        self.num_steps.set(10)
        self.update_analysis()
    
    def decrease_iteration(self):
        """Decrease iteration count."""
        current = self.iterations.get()
        if current > 0:
            self.iterations.set(current - 1)
            self.update_analysis()
    
    def increase_iteration(self):
        """Increase iteration count."""
        current = self.iterations.get()
        if current < 6:
            self.iterations.set(current + 1)
            self.update_analysis()
    
    def update_analysis(self):
        """Update the analysis and plots."""
        # Generate Koch curve
        p1 = [self.start_x.get(), self.start_y.get()]
        p2 = [self.end_x.get(), self.end_y.get()]
        segments = self.generate_koch_curve(p1, p2, int(self.iterations.get()))
        
        # Calculate domain
        all_x = []
        all_y = []
        for seg in segments:
            all_x.extend([seg[0], seg[2]])
            all_y.extend([seg[1], seg[3]])
        
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        
        padding = 0.1
        x_pad = (xmax - xmin) * padding
        y_pad = max((ymax - ymin) * padding, 0.5)
        
        domain_x = [xmin - x_pad, xmax + x_pad]
        domain_y = [ymin - y_pad, ymax + y_pad]
        
        # Perform box-counting
        results = self.perform_box_counting(segments, domain_x, domain_y)
        
        # Update results labels
        theoretical = np.log(4) / np.log(3)
        error_pct = abs(results['slope'] - theoretical) * 100 / theoretical
        
        self.measured_dim_label.config(text=f"Measured D: {results['slope']:.6f}")
        self.error_label.config(text=f"Error: {error_pct:.2f}%")
        self.r2_label.config(text=f"R²: {results['r_squared']:.8f}")
        
        # Update plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot 1: Koch curve
        for seg in segments:
            self.ax1.plot([seg[0], seg[2]], [seg[1], seg[3]], 'b-', linewidth=1)
        
        self.ax1.set_xlabel('X', fontweight='bold')
        self.ax1.set_ylabel('Y', fontweight='bold')
        self.ax1.set_title(f'Koch Curve (Iteration {int(self.iterations.get())})', 
                          fontweight='bold', fontsize=12)
        self.ax1.set_xlim(domain_x)
        self.ax1.set_ylim(domain_y)
        self.ax1.set_aspect('equal', adjustable='box')
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-log plot
        self.ax2.scatter(results['log_inv_delta'], results['log_n_boxes'], 
                        s=80, c='blue', alpha=0.6, edgecolors='darkblue', 
                        linewidth=2, label='Data points', zorder=3)
        
        # Fitted line
        fitted_y = results['slope'] * results['log_inv_delta'] + results['intercept']
        self.ax2.plot(results['log_inv_delta'], fitted_y, 'r-', linewidth=2, 
                     label=f'Fit: D = {results["slope"]:.6f}', zorder=2)
        
        self.ax2.set_xlabel('log(1/δ)', fontweight='bold')
        self.ax2.set_ylabel('log(N)', fontweight='bold')
        self.ax2.set_title('Log-Log Plot: Box-Counting', fontweight='bold', fontsize=12)
        self.ax2.grid(True, alpha=0.3, linestyle='--')
        self.ax2.legend()
        
        # Add text box with stats
        textstr = f'R² = {results["r_squared"]:.8f}\nDimension = {results["slope"]:.6f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.ax2.text(0.05, 0.95, textstr, transform=self.ax2.transAxes, 
                     fontsize=10, verticalalignment='top', bbox=props)
        
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = KochCurveAnalyzer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
