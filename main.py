"""
Main script for 3D missile tracking simulation and analysis.
"""

import numpy as np
from simulation import Satellite, MissileSimulator, simulate_measurements
from tracker import MissileTracker
from analysis import run_full_analysis


def main():
    """Run complete missile tracking simulation and analysis."""
    
    print("=" * 60)
    print("3D MISSILE TRACKING SIMULATION")
    print("=" * 60)
    
    # ========== Setup ==========
    print("\n[1/5] Setting up simulation parameters...")
    
    # Create satellites in LEO (1000 km altitude)
    num_satellites = 3
    satellites = []
    for i in range(num_satellites):
        # Equally spaced around the equator
        true_anomaly = (360.0 / num_satellites) * i
        sat = Satellite(sat_id=i, altitude_km=1000.0, true_anomaly_deg=true_anomaly)
        satellites.append(sat)
        print(f"  Satellite {i}: True anomaly = {true_anomaly:.1f}°, "
              f"Orbital period = {sat.orbital_period_h:.2f} hours")
    
    # Create missile trajectory
    # Initial position: 200 km altitude, flying north
    initial_pos = np.array([6500.0, 0.0, 200.0])  # km
    initial_vel = np.array([0.0, 8.0, 3.0])  # km/s (northbound with vertical component)
    
    missile = MissileSimulator(initial_pos, initial_vel)
    print(f"  Missile initial position: {initial_pos} km")
    print(f"  Missile initial velocity: {initial_vel} km/s")
    
    # Simulation time
    duration_s = 600.0  # 10 minutes
    dt = 1.0  # 1 second
    time_array_s = np.arange(0, duration_s, dt)
    print(f"  Simulation duration: {duration_s:.0f} seconds, dt = {dt:.1f} s")
    
    # ========== Simulation ==========
    print("\n[2/5] Running simulation...")
    
    np.random.seed(42)  # For reproducibility
    measurements_dfs, truth_df = simulate_measurements(
        satellites,
        missile,
        time_array_s,
        measurement_noise_std=0.005
    )
    
    print(f"  Generated {len(truth_df)} truth samples")
    for sat_id, meas_df in measurements_dfs.items():
        print(f"  Satellite {sat_id}: {len(meas_df)} measurements")
    
    # Print some truth data
    print("\n  Truth data sample:")
    print(truth_df[['time_s', 'pos_x_km', 'pos_y_km', 'pos_z_km']].head())
    
    # ========== Tracking ==========
    print("\n[3/5] Running tracker...")
    
    # Initialize tracker
    process_noise_std = np.array([
        0.5,    # Position noise (km)
        0.5,
        0.5,
        0.05,   # Velocity noise (km/s)
        0.05,
        0.05,
        0.01,   # Acceleration noise (km/s^2)
        0.01,
        0.01
    ])
    
    tracker = MissileTracker(
        dt=dt,
        process_noise_std=process_noise_std,
        measurement_noise_std=0.005
    )
    
    # Process measurements
    estimates_df = tracker.process_measurements(measurements_dfs)
    
    print(f"  Generated {len(estimates_df)} estimates")
    print("\n  Estimate data sample:")
    print(estimates_df[['time_s', 'est_pos_x_km', 'est_pos_y_km', 'est_pos_z_km']].head())
    
    # ========== Analysis ==========
    print("\n[4/5] Running analysis...")
    
    run_full_analysis(truth_df, estimates_df)
    
    # ========== Save Results ==========
    print("\n[5/5] Saving results...")
    
    truth_df.to_csv('/Users/sethmerickel/Projects/tracking3D/truth_trajectory.csv', index=False)
    estimates_df.to_csv('/Users/sethmerickel/Projects/tracking3D/estimates_trajectory.csv', index=False)
    
    for sat_id, meas_df in measurements_dfs.items():
        meas_df.to_csv(f'/Users/sethmerickel/Projects/tracking3D/measurements_sat{sat_id}.csv', index=False)
    
    print("  Saved CSV files to /Users/sethmerickel/Projects/tracking3D/")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
