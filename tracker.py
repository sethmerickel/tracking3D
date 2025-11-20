"""
Missile tracking module using line-of-sight measurements from satellites.
Implements an extended Kalman filter for tracking position, velocity, and acceleration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class ExtendedKalmanFilter:
    """Extended Kalman Filter for missile tracking with LOS measurements."""
    
    def __init__(
        self,
        dt: float,
        process_noise_std: np.ndarray,
        measurement_noise_std: float,
        initial_state: Optional[np.ndarray] = None
    ):
        """
        Initialize EKF.
        
        State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        
        Args:
            dt: Time step (seconds)
            process_noise_std: Standard deviation of process noise for each state element
            measurement_noise_std: Standard deviation of LOS measurement noise
            initial_state: Initial state estimate (9-element vector)
        """
        self.dt = dt
        self.state_dim = 9
        self.meas_dim = 1  # Scalar LOS measurement
        
        # State: [x, y, z, vx, vy, vz, ax, ay, az]
        if initial_state is None:
            self.state = np.zeros(self.state_dim)
        else:
            self.state = initial_state.copy()
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 100.0  # Large initial uncertainty
        
        # Process noise covariance
        self.Q = np.diag(process_noise_std**2)
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise_std**2]])
    
    def state_transition_matrix(self) -> np.ndarray:
        """
        Get state transition matrix F for constant acceleration model.
        
        State: [x, y, z, vx, vy, vz, ax, ay, az]
        """
        F = np.eye(self.state_dim)
        
        # Position update: x = x + vx*dt + 0.5*ax*dt^2
        F[0, 3] = self.dt
        F[0, 6] = 0.5 * self.dt**2
        F[1, 4] = self.dt
        F[1, 7] = 0.5 * self.dt**2
        F[2, 5] = self.dt
        F[2, 8] = 0.5 * self.dt**2
        
        # Velocity update: v = v + a*dt
        F[3, 6] = self.dt
        F[4, 7] = self.dt
        F[5, 8] = self.dt
        
        # Acceleration remains constant
        
        return F
    
    def measurement_jacobian(self, sat_pos: np.ndarray) -> np.ndarray:
        """
        Get measurement Jacobian H for LOS measurement from satellite.
        
        Measurement: unit vector from satellite to missile
        
        Args:
            sat_pos: Satellite position [x, y, z] in km
            
        Returns:
            Jacobian matrix (1 x 9)
        """
        # Extract missile position from state
        missile_pos = self.state[:3]
        
        # Vector from satellite to missile
        los_vector = missile_pos - sat_pos
        los_distance = np.linalg.norm(los_vector)
        
        if los_distance < 1e-6:
            return np.zeros((1, self.state_dim))
        
        # We measure one component of the LOS unit vector
        # H is the Jacobian of measurement with respect to state
        # For simplicity, we'll use a pseudo-measurement approach
        H = np.zeros((1, self.state_dim))
        
        # Partial derivative w.r.t. position components
        H[0, 0] = los_vector[0] / (los_distance**2)
        H[0, 1] = los_vector[1] / (los_distance**2)
        H[0, 2] = los_vector[2] / (los_distance**2)
        
        return H
    
    def predict(self) -> None:
        """Predict step of EKF."""
        F = self.state_transition_matrix()
        
        # State prediction
        self.state = F @ self.state
        
        # Covariance prediction
        self.P[:] = F @ self.P @ F.T + self.Q 
    def update(self, measurement: float, sat_pos: np.ndarray) -> None:
        """
        Update step of EKF with LOS measurement.
        
        Args:
            measurement: Scalar LOS measurement
            sat_pos: Satellite position [x, y, z]
        """
        H = self.measurement_jacobian(sat_pos)
        
        # Innovation
        innovation = measurement - (H @ self.state)[0]
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T / S[0, 0]
        
        # State update
        self.state = self.state + K.flatten() * innovation
        
        # Covariance update
        self.P[:] = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def get_position(self) -> np.ndarray:
        """Get estimated missile position."""
        return self.state[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get estimated missile velocity."""
        return self.state[3:6].copy()
    
    def get_acceleration(self) -> np.ndarray:
        """Get estimated missile acceleration."""
        return self.state[6:9].copy()
    
    def get_position_covariance(self) -> np.ndarray:
        """Get position covariance (3x3 submatrix)."""
        return self.P[:3, :3].copy()


class MissileTracker:
    """Main tracker class that processes satellite measurements."""
    
    def __init__(
        self,
        dt: float,
        process_noise_std: Optional[np.ndarray] = None,
        measurement_noise_std: float = 0.001,
        initial_state: Optional[np.ndarray] = None
    ):
        """
        Initialize tracker.
        
        Args:
            dt: Time step (seconds)
            process_noise_std: Process noise standard deviations (9-element array)
            measurement_noise_std: LOS measurement noise standard deviation
            initial_state: Initial state estimate
        """
        if process_noise_std is None:
            # Default process noise for position, velocity, acceleration
            process_noise_std = np.array([
                0.1,    # Position noise (km)
                0.1,
                0.1,
                0.01,   # Velocity noise (km/s)
                0.01,
                0.01,
                0.001,  # Acceleration noise (km/s^2)
                0.001,
                0.001
            ])
        
        self.dt = dt
        self.ekf = ExtendedKalmanFilter(dt, process_noise_std, measurement_noise_std, initial_state)
        self.estimates = []
    
    def process_measurements(
        self,
        measurements_dfs: Dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Process satellite measurements and produce missile trajectory estimate.
        
        Args:
            measurements_dfs: Dictionary mapping satellite ID to measurement DataFrames
            
        Returns:
            DataFrame with estimated trajectory (time, position, velocity, acceleration, covariance)
        """
        # Get unique times from first satellite's measurements
        first_sat_id = next(iter(measurements_dfs.keys()))
        times = measurements_dfs[first_sat_id]['time_s'].unique()
        times = np.sort(times)
        
        estimates = []
        
        for t in times:
            # Predict
            self.ekf.predict()
            
            # Update with measurements from each satellite
            for sat_id, df in measurements_dfs.items():
                # Find measurement at this time
                meas_at_t = df[df['time_s'] == t]
                
                if len(meas_at_t) > 0:
                    los_x = meas_at_t['los_x'].values[0]
                    los_y = meas_at_t['los_y'].values[0]
                    los_z = meas_at_t['los_z'].values[0]
                    sat_pos = np.array([
                        meas_at_t['sat_pos_x_km'].values[0],
                        meas_at_t['sat_pos_y_km'].values[0],
                        meas_at_t['sat_pos_z_km'].values[0]
                    ])
                    
                    # Use one LOS component as measurement
                    measurement = los_x
                    
                    # Update
                    self.ekf.update(measurement, sat_pos)
            
            # Store estimate
            pos = self.ekf.get_position()
            vel = self.ekf.get_velocity()
            acc = self.ekf.get_acceleration()
            pos_cov = self.ekf.get_position_covariance()
            
            estimates.append({
                'time_s': t,
                'est_pos_x_km': pos[0],
                'est_pos_y_km': pos[1],
                'est_pos_z_km': pos[2],
                'est_vel_x_km_s': vel[0],
                'est_vel_y_km_s': vel[1],
                'est_vel_z_km_s': vel[2],
                'est_acc_x_km_s2': acc[0],
                'est_acc_y_km_s2': acc[1],
                'est_acc_z_km_s2': acc[2],
                'pos_cov_xx': pos_cov[0, 0],
                'pos_cov_yy': pos_cov[1, 1],
                'pos_cov_zz': pos_cov[2, 2],
            })
        
        return pd.DataFrame(estimates)
