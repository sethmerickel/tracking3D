"""
Quick test to verify the improved tracking system works correctly.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/sethmerickel/Projects/tracking3D')

from simulation import Satellite, MissileSimulator, simulate_measurements
from tracker import MissileTracker


def quick_test():
    """Run a quick test of the tracking system."""
    print("Quick Tracking Test")
    print("=" * 50)
    
    # Simple 2-satellite setup
    satellites = [
        Satellite(sat_id=0, altitude_km=1000.0, true_anomaly_deg=0.0),
        Satellite(sat_id=1, altitude_km=1000.0, true_anomaly_deg=180.0),
    ]
    
    # Simple missile trajectory
    missile = MissileSimulator(
        initial_pos=np.array([6500.0, 0.0, 200.0]),
        initial_vel=np.array([0.0, 8.0, 3.0])
    )
    
    # Short simulation
    times = np.arange(0, 100, 1.0)
    meas_dfs, truth_df = simulate_measurements(satellites, missile, times, 0.001)
    
    print(f"Generated {len(truth_df)} truth samples")
    print(f"Generated measurements from {len(meas_dfs)} satellites\n")
    
    # Initialize tracker
    initial_state = np.array([6500.0, 0.0, 200.0, 0.0, 8.0, 3.0, 0.0, 0.0, -0.00981])
    
    tracker = MissileTracker(
        dt=1.0,
        process_noise_std=np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.002, 0.002, 0.002]),
        measurement_noise_std=0.001,
        initial_state=initial_state
    )
    
    # Run tracker
    estimates_df = tracker.process_measurements(meas_dfs)
    
    print(f"Generated {len(estimates_df)} estimates\n")
    
    # Compute sample error
    if len(estimates_df) > 0:
        est = estimates_df.iloc[-1]
        truth = truth_df.iloc[-1]
        
        error = np.array([
            est['est_pos_x_km'] - truth['pos_x_km'],
            est['est_pos_y_km'] - truth['pos_y_km'],
            est['est_pos_z_km'] - truth['pos_z_km']
        ])
        error_mag = np.linalg.norm(error)
        
        print(f"Final position error: {error_mag:.4f} km")
        print(f"Error components: [{error[0]:.4f}, {error[1]:.4f}, {error[2]:.4f}] km")
        print("\nTest PASSED!" if error_mag < 5.0 else "Test FAILED - error too large")
    
    return estimates_df, truth_df


if __name__ == "__main__":
    quick_test()
