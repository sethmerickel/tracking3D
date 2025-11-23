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

$$\mathbf{x}(t) = [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T$$

Where:
- Position: $\mathbf{r} = [x, y, z]^T$ in km (ECI coordinates)
- Velocity: $\mathbf{v} = [v_x, v_y, v_z]^T$ in km/s
- Acceleration: $\mathbf{a} = [a_x, a_y, a_z]^T$ in km/s²

Motion Model
------------

We use a constant acceleration (nearly-constant-velocity) model:

State Transition Matrix:

$$\mathbf{F}(t) = \begin{bmatrix} \mathbf{I}_3 & \mathbf{F}_{pos} & \mathbf{F}_{acc} \\ \mathbf{0} & \mathbf{I}_3 & \mathbf{F}_{vel} \\ \mathbf{0} & \mathbf{0} & \mathbf{I}_3 \end{bmatrix}$$

Where:

$$\mathbf{F}_{pos} = \Delta t \cdot \mathbf{I}_3 + 0.5\Delta t^2 \cdot \mathbf{I}_3 \quad \text{(position update)}$$

$$\mathbf{F}_{vel} = \Delta t \cdot \mathbf{I}_3 \quad \text{(velocity update)}$$

Explicit state equations:

$$x(k+1) = x(k) + v_x(k)\Delta t + \frac{1}{2}a_x(k)\Delta t^2$$

$$y(k+1) = y(k) + v_y(k)\Delta t + \frac{1}{2}a_y(k)\Delta t^2$$

$$z(k+1) = z(k) + v_z(k)\Delta t + \frac{1}{2}a_z(k)\Delta t^2$$

$$v_x(k+1) = v_x(k) + a_x(k)\Delta t$$

$$v_y(k+1) = v_y(k) + a_y(k)\Delta t$$

$$v_z(k+1) = v_z(k) + a_z(k)\Delta t$$

$$a_x(k+1) = a_x(k), \quad a_y(k+1) = a_y(k), \quad a_z(k+1) = a_z(k)$$

Process Noise Model
-------------------

Process noise accounts for modeling uncertainties:

$$\mathbf{x}(k+1) = \mathbf{F}(k)\mathbf{x}(k) + \mathbf{w}(k)$$

Where $\mathbf{w}(k) \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$, and $\mathbf{Q}$ is the process noise covariance matrix:

$$\mathbf{Q} = \text{diag}(\sigma_x^2, \sigma_y^2, \sigma_z^2, \sigma_{v_x}^2, \sigma_{v_y}^2, \sigma_{v_z}^2, \sigma_{a_x}^2, \sigma_{a_y}^2, \sigma_{a_z}^2)$$

Typical values:
- $\sigma_p \approx 0.5\text{-}1.0$ km (position uncertainty)
- $\sigma_v \approx 0.05\text{-}0.1$ km/s (velocity uncertainty)
- $\sigma_a \approx 0.001\text{-}0.01$ km/s² (acceleration uncertainty)

Measurement Model
-----------------

Line-of-Sight (LOS) Measurement:

Given satellite position $\mathbf{r}_{sat} = [x_{sat}, y_{sat}, z_{sat}]^T$ and missile position $\mathbf{r} = [x, y, z]^T$,
the LOS vector is:

$$\mathbf{los}_{vec} = \mathbf{r} - \mathbf{r}_{sat}$$

The LOS unit vector is:

$$\mathbf{u} = \frac{\mathbf{los}_{vec}}{||\mathbf{los}_{vec}||}$$

Where $||\mathbf{los}_{vec}||$ is the Euclidean norm:

$$||\mathbf{los}_{vec}|| = \sqrt{(x - x_{sat})^2 + (y - y_{sat})^2 + (z - z_{sat})^2}$$

The measurement is each component of the unit vector:

$$z_i = u_i + v_i \quad (i \in \{x, y, z\})$$

Where $v_i \sim \mathcal{N}(0, \sigma_m^2)$ is measurement noise.

Measurement Jacobian
---------------------

Detailed Derivation

The LOS unit vector is defined as:

$$\mathbf{u} = \frac{\mathbf{los}_{vec}}{r} = \frac{\mathbf{los}_{vec}}{||\mathbf{los}_{vec}||}$$

where $r = ||\mathbf{los}_{vec}|| = \sqrt{\mathbf{los}_{vec}^T \mathbf{los}_{vec}}$.

The i-th component is:

$$u_i = \frac{\text{los}_{vec,i}}{r}$$

To find the Jacobian, we take the partial derivative with respect to the j-th position component:

$$\frac{\partial u_i}{\partial r_j} = \frac{\partial}{\partial r_j}\left(\frac{\text{los}_{vec,i}}{r}\right)$$

Using the quotient rule:

$$\frac{\partial u_i}{\partial r_j} = \frac{\frac{\partial \text{los}_{vec,i}}{\partial r_j} \cdot r - \text{los}_{vec,i} \cdot \frac{\partial r}{\partial r_j}}{r^2}$$

Now we compute each term:

1. Since $\text{los}_{vec,i} = r_i - r_{sat,i}$:
$$\frac{\partial \text{los}_{vec,i}}{\partial r_j} = \delta_{i,j}$$

2. The derivative of the norm:
$$\frac{\partial r}{\partial r_j} = \frac{\partial}{\partial r_j}\sqrt{\mathbf{los}_{vec}^T \mathbf{los}_{vec}} = \frac{\text{los}_{vec,j}}{r}$$

Substituting back:

$$\frac{\partial u_i}{\partial r_j} = \frac{\delta_{i,j} \cdot r - \text{los}_{vec,i} \cdot \frac{\text{los}_{vec,j}}{r}}{r^2}$$

$$= \frac{\delta_{i,j}}{r} - \frac{\text{los}_{vec,i} \cdot \text{los}_{vec,j}}{r^3}$$

Since $u_i = \frac{\text{los}_{vec,i}}{r}$ and $u_j = \frac{\text{los}_{vec,j}}{r}$:

$$\frac{\partial u_i}{\partial r_j} = \frac{\delta_{i,j}}{r} - \frac{u_i u_j}{r}$$

$$= \frac{1}{r}\left(\delta_{i,j} - u_i u_j\right)$$

Final Jacobian Form

For the i-th component of the LOS unit vector, the Jacobian with respect to position is:

$$\frac{\partial u_i}{\partial \mathbf{r}} = \frac{1}{r}\left(\delta_{ij} - u_i u_j\right)$$

where $r = ||\mathbf{los}_{vec}||$ is the range.

The full measurement Jacobian row (accounting for 9-element state) is:

$$\mathbf{H}_i = \left[\frac{1}{r}(\delta_{i,x} - u_i u_x), \frac{1}{r}(\delta_{i,y} - u_i u_y), \frac{1}{r}(\delta_{i,z} - u_i u_z), 0, 0, 0, 0, 0, 0\right]$$

Or equivalently:

$$\mathbf{H}_i = [H_{i,x}, H_{i,y}, H_{i,z}, 0, 0, 0, 0, 0, 0]$$

Where:

$$H_{i,x} = \frac{\delta_{i,x} - u_i u_x}{r}$$

$$H_{i,y} = \frac{\delta_{i,y} - u_i u_y}{r}$$

$$H_{i,z} = \frac{\delta_{i,z} - u_i u_z}{r}$$

Note that $\delta_{i,j}$ is the Kronecker delta function: $\delta_{i,j} = 1$ if $i=j$, and $0$ otherwise.

Physical Interpretation

The Jacobian has an important geometric interpretation:

- The term $\frac{\delta_{i,j}}{r}$ represents the direct sensitivity of the unit vector component to position changes
- The term $-\frac{u_i u_j}{r}$ is the correction due to the normalization constraint
- When the missile moves perpendicular to the LOS direction, the unit vector changes more
- When the missile moves along the LOS direction (radial), the unit vector is less sensitive
- The factor $\frac{1}{r}$ shows that sensitivity decreases with increasing range

Example: For the x-component ($u_x$) derivative with respect to x position:

$$H_{x,x} = \frac{1 - u_x^2}{r} = \frac{1}{r}(1 - u_x^2) = \frac{1}{r}(u_y^2 + u_z^2)$$

This shows that the sensitivity is maximum when the LOS is perpendicular to the x-axis (maximum $u_y^2 + u_z^2$)
and zero when the LOS is aligned with the x-axis ($u_x = \pm 1$).

Derivation of Range Derivative

The derivative of the range (norm) is a key component. Starting with:

$$r = ||\mathbf{los}_{vec}|| = \sqrt{\sum_{i=1}^{3} \text{los}_{vec,i}^2}$$

We can also express this as:

$$r^2 = \mathbf{los}_{vec}^T \mathbf{los}_{vec}$$

Taking the differential:

$$d(r^2) = 2\mathbf{los}_{vec}^T d\mathbf{los}_{vec}$$

$$2r \, dr = 2\mathbf{los}_{vec}^T d\mathbf{los}_{vec}$$

For the partial derivative with respect to $r_j$:

$$2r \frac{\partial r}{\partial r_j} = 2 \sum_{i=1}^{3} \text{los}_{vec,i} \frac{\partial \text{los}_{vec,i}}{\partial r_j}$$

Since $\text{los}_{vec,i} = r_i - r_{sat,i}$:

$$2r \frac{\partial r}{\partial r_j} = 2 \sum_{i=1}^{3} \text{los}_{vec,i} \delta_{i,j} = 2\text{los}_{vec,j}$$

Therefore:

$$\frac{\partial r}{\partial r_j} = \frac{\text{los}_{vec,j}}{r}$$

Matrix Form of Jacobian

When processing three components of the LOS vector simultaneously, the full measurement Jacobian is a $3 \times 9$ matrix:

$$\mathbf{H} = \begin{bmatrix} H_{x,x} & H_{x,y} & H_{x,z} & 0 & 0 & 0 & 0 & 0 & 0 \\ H_{y,x} & H_{y,y} & H_{y,z} & 0 & 0 & 0 & 0 & 0 & 0 \\ H_{z,x} & H_{z,y} & H_{z,z} & 0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix}$$

The measurement residuals are:

$$\mathbf{y} = \begin{bmatrix} z_x - u_x \\ z_y - u_y \\ z_z - u_z \end{bmatrix}$$

where $(z_x, z_y, z_z)$ are the measured LOS components and $(u_x, u_y, u_z)$ are the predicted unit vector components.

Properties of the Jacobian Matrix

1. **Rank Deficiency**: The Jacobian is rank-2 (full row rank), not rank-3, because the LOS components satisfy the unit norm constraint:

$$u_x^2 + u_y^2 + u_z^2 = 1$$

This means the three measurements are not independent; they satisfy one constraint equation.

2. **Normalization Correction**: The term $-\frac{u_i u_j}{r}$ in the Jacobian automatically accounts for this constraint. When you differentiate a unit vector, you must correct for the change in magnitude.

3. **Symmetry**: The Jacobian is symmetric in the sense that:

$$H_{i,j} = \frac{1}{r}(\delta_{i,j} - u_i u_j) = H_{j,i}$$

4. **Singular Case**: The Jacobian becomes singular (rank-0) only when $r = 0$, i.e., when the satellite and missile are at the same location. This is physically unrealistic and handled as a special case.

5. **Condition Number**: The Jacobian is better conditioned when the LOS direction is perpendicular to the observation coordinate axis (e.g., when the missile is far from the x-axis for the x-component measurement).

Derivation Summary Table

For quick reference, here are the key derivatives for each LOS component:

| Component | $\frac{\partial u_x}{\partial \mathbf{r}}$ | $\frac{\partial u_y}{\partial \mathbf{r}}$ | $\frac{\partial u_z}{\partial \mathbf{r}}$ |
|-----------|-------|-------|-------|
| w.r.t. $r_x$ | $\frac{1-u_x^2}{r}$ | $\frac{-u_x u_y}{r}$ | $\frac{-u_x u_z}{r}$ |
| w.r.t. $r_y$ | $\frac{-u_x u_y}{r}$ | $\frac{1-u_y^2}{r}$ | $\frac{-u_y u_z}{r}$ |
| w.r.t. $r_z$ | $\frac{-u_x u_z}{r}$ | $\frac{-u_y u_z}{r}$ | $\frac{1-u_z^2}{r}$ |

Notice that the diagonal elements have the form $\frac{1-u_i^2}{r}$ while the off-diagonal elements have the form $\frac{-u_i u_j}{r}$.

Unit Vector Constraint Analysis

A critical property of the LOS measurement is the unit vector constraint:

$$g(\mathbf{u}) = u_x^2 + u_y^2 + u_z^2 - 1 = 0$$

This constraint means that the three LOS components are not independent. To understand the implications, consider the gradient of this constraint:

$$\nabla g = \begin{bmatrix} 2u_x \\ 2u_y \\ 2u_z \end{bmatrix} = 2\mathbf{u}$$

The constraint defines a 2-dimensional surface (unit sphere) in 3-dimensional measurement space. When we differentiate the constraint with respect to position, we get:

$$\frac{\partial g}{\partial r_j} = 2 \sum_{i=1}^{3} u_i \frac{\partial u_i}{\partial r_j} = 0$$

Substituting our Jacobian:

$$2 \sum_{i=1}^{3} u_i \cdot \frac{1}{r}(\delta_{i,j} - u_i u_j) = 0$$

$$\frac{2}{r} \sum_{i=1}^{3} u_i \delta_{i,j} - \frac{2}{r} \sum_{i=1}^{3} u_i^2 u_j = 0$$

$$\frac{2}{r} u_j - \frac{2}{r} u_j \sum_{i=1}^{3} u_i^2 = 0$$

$$\frac{2}{r} u_j - \frac{2}{r} u_j \cdot 1 = 0$$

This confirms that the constraint is satisfied by our Jacobian, ensuring consistency.

Sensitivity Analysis: Radial vs Transverse Motion

A key insight is understanding how the Jacobian responds to motion in different directions:

**Transverse Motion** (perpendicular to LOS):
When the missile moves perpendicular to the LOS direction, the change in the unit vector is large. 
For example, if motion is in the y-direction when $u_y \approx 0$:

$$\Delta u_x \approx \frac{\Delta y}{r} \quad \text{(large sensitivity)}$$

**Radial Motion** (along LOS):
When the missile moves along the LOS direction, the change in the unit vector is small.
For motion in the x-direction when $u_x \approx 1$:

$$\Delta u_x \approx \frac{\Delta x}{r}(1 - u_x^2) = \frac{\Delta x}{r}(1 - 1) = 0 \quad \text{(no sensitivity)}$$

This fundamental property explains why:
- Range is unobservable from LOS-only measurements
- Transverse position is well-observed
- A single satellite provides 2 DOF of observability (azimuth and elevation)
- Multiple satellites are needed to estimate range and 3D position

Numerical Implementation Notes

When implementing the Jacobian in code, consider these practical issues:

1. **Zero Range Protection**: Check for $r < \epsilon$ (typically $\epsilon = 10^{-6}$ km) to avoid division by zero.

2. **Numerical Stability**: Use the factorization:
$$\mathbf{H}_i = \frac{1}{r}\left[\delta_{i,j} \mathbf{I}_3 - \mathbf{u} \otimes \mathbf{u}\right]$$

where $\otimes$ denotes outer product. This avoids computing unit vector components multiple times.

3. **Efficient Computation**: 
$$\mathbf{u} \otimes \mathbf{u} = \begin{bmatrix} u_x^2 & u_x u_y & u_x u_z \\ u_y u_x & u_y^2 & u_y u_z \\ u_z u_x & u_z u_y & u_z^2 \end{bmatrix}$$

4. **Rank Deficiency Handling**: Since the measurement is rank-deficient (rank 2), the innovation covariance $\mathbf{S}$ will be singular. Use pseudoinverse or process all three components sequentially rather than simultaneously.

Higher-Order Derivatives (Optional)

If implementing a second-order filter (e.g., for nonlinearity correction), the Hessian is needed:

$$\mathbf{H}_{i,jk} = \frac{\partial^2 u_i}{\partial r_j \partial r_k}$$

Starting with:
$$\frac{\partial u_i}{\partial r_j} = \frac{1}{r}(\delta_{i,j} - u_i u_j)$$

Taking another derivative:
$$\frac{\partial^2 u_i}{\partial r_k \partial r_j} = \frac{\partial}{\partial r_k}\left[\frac{1}{r}(\delta_{i,j} - u_i u_j)\right]$$

$$= -\frac{1}{r^2}\frac{\text{los}_{vec,k}}{r}(\delta_{i,j} - u_i u_j) + \frac{1}{r}\left[-\frac{\partial u_i}{\partial r_k}u_j - u_i\frac{\partial u_j}{\partial r_k}\right]$$

This becomes algebraically complex and is typically not needed for standard EKF implementations.

Extended Kalman Filter
----------------------

The EKF estimates the state in two steps:

**Prediction Step:**

State prediction:
$$\hat{\mathbf{x}}^-(k+1) = \mathbf{F}(k)\hat{\mathbf{x}}(k)$$

Covariance prediction:
$$\mathbf{P}^-(k+1) = \mathbf{F}(k)\mathbf{P}(k)\mathbf{F}(k)^T + \mathbf{Q}(k)$$

Where:
- $\hat{\mathbf{x}}$ is the state estimate
- $\mathbf{P}$ is the error covariance matrix
- Superscript $-$ denotes a priori (predicted) values

**Update Step (for each LOS measurement component):**

Innovation (measurement residual):
$$\mathbf{y}(k) = z(k) - \mathbf{H}(k)\hat{\mathbf{x}}^-(k)$$

Innovation covariance:
$$\mathbf{S}(k) = \mathbf{H}(k)\mathbf{P}^-(k)\mathbf{H}(k)^T + R(k)$$

Kalman gain:
$$\mathbf{K}(k) = \mathbf{P}^-(k)\mathbf{H}(k)^T / \mathbf{S}(k)$$

State update:
$$\hat{\mathbf{x}}(k) = \hat{\mathbf{x}}^-(k) + \mathbf{K}(k)\mathbf{y}(k)$$

Covariance update (Joseph form for numerical stability):
$$\mathbf{P}(k) = [\mathbf{I} - \mathbf{K}(k)\mathbf{H}(k)]\mathbf{P}^-(k)$$

Where $R(k) = \sigma_m^2$ is the measurement noise variance.

Missile Ballistics Model
------------------------

The missile follows a ballistic trajectory under constant gravitational acceleration:

$$\mathbf{r}(t) = \mathbf{r}_0 + \mathbf{v}_0 t + \frac{1}{2}\mathbf{a}_{grav}t^2$$

$$\mathbf{v}(t) = \mathbf{v}_0 + \mathbf{a}_{grav}t$$

$$\mathbf{a}(t) = [0, 0, -g]$$

Where:
- $\mathbf{r}_0$ = initial position
- $\mathbf{v}_0$ = initial velocity
- $g = 0.00981$ km/s² (gravitational acceleration magnitude)
- $\mathbf{a}_{grav} = [0, 0, -g]$ (gravitational acceleration vector in ECI)

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
$$r_{orb} = R_{Earth} + h$$

Where:
- $R_{Earth} = 6371$ km (Earth's mean radius)
- $h$ = orbital altitude (e.g., 1000 km for LEO)

Mean motion (angular velocity):
$$n = \sqrt{\frac{\mu}{r_{orb}^3}}$$

Where $\mu = 398600.4418$ km³/s² is Earth's gravitational parameter.

Orbital period:
$$T = \frac{2\pi}{n}$$

Position in ECI (equatorial plane):
$$x(t) = r_{orb}\cos(\theta(t))$$
$$y(t) = r_{orb}\sin(\theta(t))$$
$$z(t) = 0$$

True anomaly:
$$\theta(t) = \theta_0 + nt$$

Velocity in ECI:
$$v_x(t) = -r_{orb}n\sin(\theta(t))$$
$$v_y(t) = r_{orb}n\cos(\theta(t))$$
$$v_z(t) = 0$$

Orbital speed:
$$v_{orb} = r_{orb}n = \sqrt{\frac{\mu}{r_{orb}}}$$

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

For a constellation of N satellites with measurement noise $\sigma_m$ and geometry factor $G$:

Position standard deviation (1-sigma):
$$\sigma_{pos} \approx G \cdot \sigma_m \cdot r_{range}$$

Where:
- $G$ = geometry factor (typically 0.5-2.0)
- $r_{range}$ = distance from satellite to missile (~7000 km)

Typical numbers:
- With $\sigma_m = 0.001$ (0.1% of unit vector)
- $\sigma_{pos} \approx 0.5\text{-}1.0 \times 7$ km $\approx 3.5\text{-}7$ km (single satellite)
- $\sigma_{pos} \approx 1.0\text{-}2.0$ km (three satellites with good geometry)
- $\sigma_{pos} \approx 0.1\text{-}0.5$ km (five satellites with excellent geometry)

Extending the Application
==========================

To add features:

1. Different motion models: Modify ExtendedKalmanFilter.state_transition_matrix()
   - Higher-order polynomials
   - Turn-rate models with state: $[\mathbf{r}, \mathbf{v}, \mathbf{a}, \omega]$
   - Maneuver detection with adaptive noise

2. More realistic sensors:
   - Add measurement bias: $z_i = u_i + b_i + v_i$
   - Time-correlated noise with colored noise model
   - Intermittent measurements and sensor dropouts
   - Measurement outlier rejection

3. Atmospheric effects: Add drag model to MissileSimulator.get_acceleration()
   - Ballistic coefficient $C_B$
   - Atmospheric density $\rho(altitude)$
   - Drag acceleration: $\mathbf{a}_{drag} = -\frac{1}{2}\rho C_B \mathbf{v}||\mathbf{v}||$

4. Earth rotation: Use ECEF coordinates instead of ECI
   - Need rotation transformation matrices
   - Coriolis acceleration effects

5. Advanced filtering:
   - Unscented Kalman Filter (UKF) for better nonlinearity handling
   - Particle Filter for multi-modal distributions
   - Multiple Model Adaptive Estimation (MMAE) for maneuvers

6. Sensor fusion:
   - Range measurements: $z_r = ||\mathbf{los}_{vec}|| + v_r$
   - Doppler velocity measurements: $z_d = \frac{d||\mathbf{los}_{vec}||}{dt} + v_d$
   - Angle measurements in spherical coordinates

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

