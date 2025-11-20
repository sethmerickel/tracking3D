"""
Analysis and visualization module for missile tracking results.
Compares estimated trajectory to truth and plots error with confidence envelopes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple


class TrackingAnalyzer:
    """Analyzes tracking performance by comparing estimates to truth."""
    
    def __init__(self, truth_df: pd.DataFrame, estimates_df: pd.DataFrame):
        """
        Initialize analyzer with truth and estimates.
        
        Args:
            truth_df: Truth trajectory DataFrame
            estimates_df: Estimated trajectory DataFrame
        """
        self.truth_df = truth_df
        self.estimates_df = estimates_df
        
        # Merge DataFrames on time
        self.merged_df = pd.merge(
            truth_df,
            estimates_df,
            on='time_s',
            how='inner'
        )
    
    def compute_position_errors(self) -> pd.DataFrame:
        """
        Compute position errors (estimate - truth).
        
        Returns:
            DataFrame with position errors and standard deviations
        """
        errors = []
        
        for _, row in self.merged_df.iterrows():
            # Truth position
            truth_pos = np.array([row['pos_x_km'], row['pos_y_km'], row['pos_z_km']])
            
            # Estimated position
            est_pos = np.array([row['est_pos_x_km'], row['est_pos_y_km'], row['est_pos_z_km']])
            
            # Error
            error = est_pos - truth_pos
            error_mag = np.linalg.norm(error)
            
            # Standard deviations (1-sigma from covariance diagonal)
            std_x = np.sqrt(row['pos_cov_xx'])
            std_y = np.sqrt(row['pos_cov_yy'])
            std_z = np.sqrt(row['pos_cov_zz'])
            
            errors.append({
                'time_s': row['time_s'],
                'error_x_km': error[0],
                'error_y_km': error[1],
                'error_z_km': error[2],
                'error_mag_km': error_mag,
                'std_x_km': std_x,
                'std_y_km': std_y,
                'std_z_km': std_z,
            })
        
        return pd.DataFrame(errors)
    
    def compute_velocity_errors(self) -> pd.DataFrame:
        """
        Compute velocity errors (estimate - truth).
        
        Returns:
            DataFrame with velocity errors
        """
        errors = []
        
        for _, row in self.merged_df.iterrows():
            # Truth velocity
            truth_vel = np.array([row['vel_x_km_s'], row['vel_y_km_s'], row['vel_z_km_s']])
            
            # Estimated velocity
            est_vel = np.array([row['est_vel_x_km_s'], row['est_vel_y_km_s'], row['est_vel_z_km_s']])
            
            # Error
            error = est_vel - truth_vel
            error_mag = np.linalg.norm(error)
            
            errors.append({
                'time_s': row['time_s'],
                'error_x_km_s': error[0],
                'error_y_km_s': error[1],
                'error_z_km_s': error[2],
                'error_mag_km_s': error_mag,
            })
        
        return pd.DataFrame(errors)
    
    def plot_position_errors(self, figsize: Tuple[int, int] = (14, 10)) -> None:
        """
        Plot position errors with confidence envelopes.
        
        Args:
            figsize: Figure size (width, height)
        """
        errors_df = self.compute_position_errors()
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Missile Tracking Position Errors with Confidence Envelopes', fontsize=14)
        
        components = ['x', 'y', 'z']
        
        for idx, component in enumerate(components):
            error_col = f'error_{component}_km'
            std_col = f'std_{component}_km'
            
            # Error vs time
            ax = axes[idx, 0]
            ax.plot(errors_df['time_s'], errors_df[error_col], 'b-', linewidth=2, label='Error')
            ax.fill_between(
                errors_df['time_s'],
                -errors_df[std_col],
                errors_df[std_col],
                alpha=0.3,
                label='±1σ (68%)'
            )
            ax.fill_between(
                errors_df['time_s'],
                -2*errors_df[std_col],
                2*errors_df[std_col],
                alpha=0.15,
                label='±2σ (95%)'
            )
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{component.upper()} Error (km)')
            ax.set_title(f'{component.upper()} Position Error')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Error histogram
            ax = axes[idx, 1]
            ax.hist(errors_df[error_col], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=errors_df[error_col].mean(), color='r', linestyle='--', 
                      linewidth=2, label=f'Mean: {errors_df[error_col].mean():.4f}')
            ax.axvline(x=errors_df[error_col].std(), color='g', linestyle='--',
                      linewidth=2, label=f'Std: {errors_df[error_col].std():.4f}')
            ax.set_xlabel(f'{component.upper()} Error (km)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{component.upper()} Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/sethmerickel/Projects/tracking3D/position_errors.png', dpi=150)
        plt.show()
    
    def plot_trajectory_3d(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot 3D trajectory: truth vs estimate.
        
        Args:
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Truth trajectory
        ax.plot(
            self.merged_df['pos_x_km'],
            self.merged_df['pos_y_km'],
            self.merged_df['pos_z_km'],
            'b-', linewidth=2, label='Truth'
        )
        
        # Estimated trajectory
        ax.plot(
            self.merged_df['est_pos_x_km'],
            self.merged_df['est_pos_y_km'],
            self.merged_df['est_pos_z_km'],
            'r--', linewidth=2, label='Estimate'
        )
        
        # Start and end points
        ax.scatter(
            self.merged_df['pos_x_km'].iloc[0],
            self.merged_df['pos_y_km'].iloc[0],
            self.merged_df['pos_z_km'].iloc[0],
            c='blue', s=100, marker='o', label='Truth Start'
        )
        ax.scatter(
            self.merged_df['est_pos_x_km'].iloc[0],
            self.merged_df['est_pos_y_km'].iloc[0],
            self.merged_df['est_pos_z_km'].iloc[0],
            c='red', s=100, marker='s', label='Est Start'
        )
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Missile Trajectory: Truth vs Estimate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/sethmerickel/Projects/tracking3D/trajectory_3d.png', dpi=150)
        plt.show()
    
    def plot_magnitude_errors(self, figsize: Tuple[int, int] = (12, 5)) -> None:
        """
        Plot magnitude of position error over time.
        
        Args:
            figsize: Figure size
        """
        errors_df = self.compute_position_errors()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Magnitude vs time
        ax1.plot(errors_df['time_s'], errors_df['error_mag_km'], 'b-', linewidth=2)
        ax1.fill_between(
            errors_df['time_s'],
            0,
            errors_df['error_mag_km'],
            alpha=0.3
        )
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position Error Magnitude (km)')
        ax1.set_title('Overall Position Error vs Time')
        ax1.grid(True, alpha=0.3)
        
        # Error statistics
        ax2.axis('off')
        stats_text = f"""
Position Error Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━
Mean Error: {errors_df['error_mag_km'].mean():.6f} km
Max Error: {errors_df['error_mag_km'].max():.6f} km
Min Error: {errors_df['error_mag_km'].min():.6f} km
Std Dev: {errors_df['error_mag_km'].std():.6f} km
RMS Error: {np.sqrt((errors_df['error_mag_km']**2).mean()):.6f} km
        """
        ax2.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=11,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('/Users/sethmerickel/Projects/tracking3D/magnitude_errors.png', dpi=150)
        plt.show()


def run_full_analysis(truth_df: pd.DataFrame, estimates_df: pd.DataFrame) -> None:
    """
    Run complete analysis and generate all plots.
    
    Args:
        truth_df: Truth trajectory DataFrame
        estimates_df: Estimated trajectory DataFrame
    """
    analyzer = TrackingAnalyzer(truth_df, estimates_df)
    
    print("Generating position error plots...")
    analyzer.plot_position_errors()
    
    print("Generating 3D trajectory plot...")
    analyzer.plot_trajectory_3d()
    
    print("Generating magnitude error plot...")
    analyzer.plot_magnitude_errors()
    
    print("Analysis complete!")
