"""
3D Missile Tracking Application

A complete simulation and tracking system for 3D missile trajectories using 
line-of-sight measurements from Low Earth Orbit (LEO) satellites.

Architecture
============

The application is organized into three main modules:

1. simulation.py
   - Satellite class: Models circular LEO orbits around Earth
   - MissileSimulator class: Simulates ballistic missile trajectory with gravity
   - simulate_measurements(): Generates synthetic satellite measurements and truth data
   
2. tracker.py
   - ExtendedKalmanFilter class: EKF with constant acceleration motion model
   - MissileTracker class: High-level tracker that processes satellite measurements
   
3. analysis.py
   - TrackingAnalyzer class: Compares estimates to truth and computes statistics
   - Visualization functions: Generates plots with error envelopes

4. main.py
   - Complete workflow demonstrating simulation → tracking → analysis

Getting Started
===============

Requirements:
- numpy
- pandas
- matplotlib

Install with:
    pip install numpy pandas matplotlib

Running the Simulation
======================

python main.py

This will:
1. Create 3 satellites in polar orbit at 1000 km altitude
2. Simulate a ballistic missile trajectory
3. Generate line-of-sight measurements from each satellite
4. Run an Extended Kalman Filter to estimate the missile trajectory
5. Analyze errors and generate plots

Output
======

The simulation produces:
- CSV files with truth, estimates, and measurements
- PNG plots showing:
  * Position errors in X, Y, Z with confidence envelopes (±1σ, ±2σ)
  * Error distributions
  * 3D trajectory visualization
  * Overall error statistics

Key Features
============

Simulation:
- Realistic circular LEO orbits using Kepler's third law
- Ballistic missile trajectory with gravity
- Satellite line-of-sight measurements (unit vectors)
- Measurement noise simulation

Tracking:
- Extended Kalman Filter with constant acceleration model
- Multi-satellite fusion
- Position, velocity, and acceleration estimation
- Covariance matrix propagation

Analysis:
- Position error vs time with predicted confidence intervals
- 3D trajectory comparison
- Statistical error analysis
- Publication-ready plots

State Vector
============

The tracker estimates a 9-element state vector:
    [x, y, z, vx, vy, vz, ax, ay, az]

Where:
- (x, y, z): Position in km
- (vx, vy, vz): Velocity in km/s
- (ax, ay, az): Acceleration in km/s²

Measurements
============

Each satellite produces:
- Satellite position (x, y, z) in ECI coordinates
- Line-of-sight unit vector from satellite to missile
- Time tag

The LOS vectors are the primary measurements used by the tracker.

Configuration
=============

Key parameters in main.py:

Satellite Configuration:
- num_satellites: Number of LEO satellites
- altitude_km: Orbital altitude (default 1000 km)
- Initial true anomalies equally spaced around equator

Missile Configuration:
- initial_pos: Starting position (km)
- initial_vel: Starting velocity (km/s)
- Gravity: 0.00981 km/s² (acceleration magnitude)

Tracker Configuration:
- process_noise_std: Process noise for each state element
- measurement_noise_std: Measurement noise standard deviation
- dt: Time step

Simulation Configuration:
- duration_s: Total simulation time
- dt: Integration time step

Extending the Application
==========================

To add features:

1. Different motion models: Modify ExtendedKalmanFilter.state_transition_matrix()
2. More realistic sensors: Add measurement bias, scale factors, etc. in simulate_measurements()
3. Additional satellites: Simply increase num_satellites in main()
4. Atmospheric effects: Add drag model to MissileSimulator.get_acceleration()
5. Earth rotation: Use ECEF coordinates instead of ECI

References
==========

- Welch, G., & Bishop, G. (2006). An Introduction to the Kalman Filter
- Curtis, H. D. (2013). Orbital Mechanics for Engineering Students
- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with Applications 
  to Tracking and Navigation
"""
