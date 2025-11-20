"""
3D Missile Tracking Application

A complete simulation and tracking system for 3D missile trajectories using 
line-of-sight measurements from Low Earth Orbit (LEO) satellites.

Table of Contents
=================
1. Architecture
2. Mathematical Theory
3. Orbital Mechanics
4. Tracking Theory
5. Configuration
6. References

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

Mathematical Theory
===================

State Vector Definition
-----------------------

The system state is represented as a 9-dimensional vector:

    x(t) = [x, y, z, vx, vy, vz, ax, ay, az]^T

Where:
- Position: r = [x, y, z]^T in km (ECI coordinates)
- Velocity: v = [vx, vy, vz]^T in km/s
- Acceleration: a = [ax, ay, az]^T in km/s^2

Motion Model
------------

We use a constant acceleration (nearly-constant-velocity) model:

State Transition Matrix:

    F(t) = | I3   F_pos   F_acc  |
           | 0    I3      F_vel  |
           | 0    0       I3     |

Where:

    F_pos = Δt * I3 + 0.5*Δt² * I3  (position update matrix)
    F_vel = Δt * I3                  (velocity update matrix)

Explicitly:

    x(k+1) = x(k) + vx(k)*Δt + 0.5*ax(k)*Δt²
    y(k+1) = y(k) + vy(k)*Δt + 0.5*ay(k)*Δt²
    z(k+1) = z(k) + vz(k)*Δt + 0.5*az(k)*Δt²
    
    vx(k+1) = vx(k) + ax(k)*Δt
    vy(k+1) = vy(k) + ay(k)*Δt
    vz(k+1) = vz(k) + az(k)*Δt
    
    ax(k+1) = ax(k)
    ay(k+1) = ay(k)
    az(k+1) = az(k)

Process Noise Model
-------------------

Process noise accounts for modeling uncertainties:

    x(k+1) = F(k)*x(k) + w(k)

Where w(k) ~ N(0, Q), and Q is the process noise covariance matrix:

    Q = diag(σ²_px, σ²_py, σ²_pz, σ²_vx, σ²_vy, σ²_vz, σ²_ax, σ²_ay, σ²_az)

Typical values:
- σ_p ≈ 0.5-1.0 km (position uncertainty)
- σ_v ≈ 0.05-0.1 km/s (velocity uncertainty)
- σ_a ≈ 0.001-0.01 km/s² (acceleration uncertainty)

Measurement Model
-----------------

Line-of-Sight (LOS) Measurement:

Given satellite position r_sat = [x_sat, y_sat, z_sat]^T and missile position r = [x, y, z]^T,
the LOS vector is:

    los_vec = r - r_sat

The LOS unit vector is:

    u = los_vec / ||los_vec||

Where ||los_vec|| is the Euclidean norm:

    ||los_vec|| = √[(x - x_sat)² + (y - y_sat)² + (z - z_sat)²]

The measurement is one component of the unit vector:

    z_i = u_i + v_i    (i ∈ {x, y, z})

Where v_i ~ N(0, σ²_m) is measurement noise.

Measurement Jacobian
---------------------

For the i-th component of the LOS unit vector, the Jacobian with respect to position is:

    H_i = ∂u_i/∂r = (δ_ij/||los_vec|| - u_i*u_j/||los_vec||)

Explicitly:

    H_i = [H_ix, H_iy, H_iz, 0, 0, 0, 0, 0, 0]

Where:

    H_ix = (δ_i,x - u_i*u_x) / ||los_vec||
    H_iy = (δ_i,y - u_i*u_y) / ||los_vec||
    H_iz = (δ_i,z - u_i*u_z) / ||los_vec||

Extended Kalman Filter
----------------------

The EKF estimates the state in two steps:

**Prediction Step:**

State prediction:
    x̂⁻(k+1) = F(k)*x̂(k)

Covariance prediction:
    P⁻(k+1) = F(k)*P(k)*F(k)^T + Q(k)

Where:
- x̂ is the state estimate
- P is the error covariance matrix
- Superscript - denotes a priori (predicted) values

**Update Step (for each LOS measurement component):**

Innovation:
    y(k) = z(k) - H(k)*x̂⁻(k)

Innovation covariance:
    S(k) = H(k)*P⁻(k)*H(k)^T + R(k)

Kalman gain:
    K(k) = P⁻(k)*H(k)^T / S(k)

State update:
    x̂(k) = x̂⁻(k) + K(k)*y(k)

Covariance update (Joseph form for numerical stability):
    P(k) = [I - K(k)*H(k)]*P⁻(k)

Where R(k) = σ²_m is the measurement noise variance.

Missile Ballistics Model
------------------------

The missile follows a ballistic trajectory under constant gravitational acceleration:

    r(t) = r₀ + v₀*t + 0.5*a_grav*t²
    v(t) = v₀ + a_grav*t
    a(t) = [0, 0, -g]

Where:
- r₀ = initial position
- v₀ = initial velocity
- g = 0.00981 km/s² (gravitational acceleration magnitude)
- a_grav = [0, 0, -g] (gravitational acceleration vector in ECI)

Note: This simplified model assumes:
- Constant gravitational acceleration (valid for short trajectories)
- No atmospheric drag
- No Coriolis effects
- Earth is non-rotating (true ECI frame)

Orbital Mechanics
=================

Circular Orbit Model
--------------------

Satellites are modeled in circular orbits using Kepler's equations.

Orbital radius:
    r_orb = R_Earth + h

Where:
- R_Earth = 6371 km (Earth's mean radius)
- h = orbital altitude (e.g., 1000 km for LEO)

Mean motion (angular velocity):
    n = √(μ/r_orb³)

Where μ = 398600.4418 km³/s² is Earth's gravitational parameter.

Orbital period:
    T = 2π/n

Position in ECI (equatorial plane):
    x(t) = r_orb*cos(θ(t))
    y(t) = r_orb*sin(θ(t))
    z(t) = 0

True anomaly:
    θ(t) = θ₀ + n*t

Velocity in ECI:
    vx(t) = -r_orb*n*sin(θ(t))
    vy(t) = r_orb*n*cos(θ(t))
    vz(t) = 0

Orbital speed:
    v_orb = r_orb*n = √(μ/r_orb)

Configuration
=============

Key parameters in main.py:

Satellite Configuration:
- num_satellites: Number of LEO satellites (default: 5)
- altitude_km: Orbital altitude (default: 1000 km)
- Initial true anomalies equally spaced around equator
- Orbital period at 1000 km: ~105 minutes

Missile Configuration:
- initial_pos: Starting position (km) [6500, 0, 200]
- initial_vel: Starting velocity (km/s) [0, 8, 3]
- Gravity: 0.00981 km/s² (constant downward acceleration)

Tracker Configuration:
- process_noise_std: Process noise for each state element
  * Position: 1.0 km
  * Velocity: 0.1 km/s
  * Acceleration: 0.002 km/s²
- measurement_noise_std: 0.001 (LOS unit vector components)
- dt: Time step (1 second)

Simulation Configuration:
- duration_s: Total simulation time (600 seconds = 10 minutes)
- dt: Integration time step (1 second)

Observability Analysis
======================

Single Satellite vs Multiple Satellites
----------------------------------------

With a single satellite, the LOS vector provides 2 degrees of freedom (DOF):
- Range is unobservable
- Only direction is measured

With two satellites, we get 4 DOF:
- Both range and cross-range components constrained
- Position becomes observable

With three or more satellites, full 3D position is observable:
- Better geometry improves estimation accuracy
- Redundancy provides robustness

Theoretical Prediction Errors
------------------------------

For a constellation of N satellites with measurement noise σ_m and geometry factor G:

Position standard deviation (1-sigma):
    σ_pos ≈ G*σ_m*r_range

Where:
- G = geometry factor (typically 0.5-2.0)
- r_range = distance from satellite to missile (~7000 km)

Typical numbers:
- With σ_m = 0.001 (0.1% of unit vector)
- σ_pos ≈ 0.5-1.0*7 km ≈ 3.5-7 km (single satellite)
- σ_pos ≈ 1.0-2.0 km (three satellites with good geometry)
- σ_pos ≈ 0.1-0.5 km (five satellites with excellent geometry)

Extending the Application
==========================

To add features:

1. Different motion models: Modify ExtendedKalmanFilter.state_transition_matrix()
   - Higher-order polynomials
   - Turn-rate models
   - Maneuver detection

2. More realistic sensors:
   - Add measurement bias
   - Time-correlated noise
   - Intermittent measurements
   - Sensor dropouts

3. Atmospheric effects: Add drag model to MissileSimulator.get_acceleration()
   - Ballistic coefficient C_B
   - Atmospheric density ρ(altitude)
   - Drag acceleration: a_drag = -0.5*ρ*C_B*v*||v||

4. Earth rotation: Use ECEF coordinates instead of ECI
   - Need transformation matrices

5. Advanced filtering:
   - Unscented Kalman Filter (UKF)
   - Particle Filter
   - Multiple Model Adaptive Estimation (MMAE)

6. Sensor fusion:
   - Range measurements
   - Doppler velocity measurements
   - Angle measurements

References
==========

- Welch, G., & Bishop, G. (2006). An Introduction to the Kalman Filter
  https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

- Curtis, H. D. (2013). Orbital Mechanics for Engineering Students
  3rd Edition, Butterworth-Heinemann

- Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). Estimation with Applications 
  to Tracking and Navigation: Algorithms and Software
  John Wiley & Sons

- Bierman, G. J. (1977). Factorization Methods for Discrete Sequential Estimation
  Academic Press

- Simon, D. (2006). Optimal State Estimation: Kalman, H-infinity, and Nonlinear Approaches
  John Wiley & Sons

- Vallado, D. A., Crawford, P., Hujsak, R., & Kelso, T. S. (2006). 
  Revisiting Spacetrack Report #3. AIAA/AAS Astrodynamics Specialist Conference

Mathematical Notation Reference
===============================

Common Symbols:
- x, y, z : Position components (km)
- vx, vy, vz : Velocity components (km/s)
- ax, ay, az : Acceleration components (km/s²)
- r : Position vector (km)
- v : Velocity vector (km/s)
- a : Acceleration vector (km/s²)
- t, Δt : Time and time step (seconds)
- θ : True anomaly (radians)
- n : Mean motion / angular velocity (rad/s)
- T : Orbital period (seconds)
- μ : Gravitational parameter (km³/s²)
- σ : Standard deviation
- P : Covariance matrix
- Q : Process noise covariance
- R : Measurement noise covariance
- F : State transition matrix
- H : Measurement Jacobian matrix
- K : Kalman gain
- x̂ : State estimate
- z : Measurement
- w : Process noise
- v : Measurement noise

"""

