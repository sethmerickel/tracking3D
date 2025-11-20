"""
Missile tracking simulation using LEO satellites.
This module simulates satellite orbits and missile trajectories with LOS measurements.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


class Satellite:
    """Represents a LEO satellite in a circular orbit."""
    
    def __init__(self, sat_id: int, altitude_km: float = 1000.0, true_anomaly_deg: float = 0.0):
        """
        Initialize a satellite in a circular orbit.
        
        Args:
            sat_id: Satellite identifier
            altitude_km: Altitude above Earth surface (km)
            true_anomaly_deg: Initial true anomaly (degrees)
        """
        self.sat_id = sat_id
        self.earth_radius_km = 6371.0
        self.orbital_radius_km = self.earth_radius_km + altitude_km
        self.altitude_km = altitude_km
        self.true_anomaly_deg = true_anomaly_deg
        
        # Orbital period (hours) using Kepler's third law
        mu_earth = 398600.4418  # km^3/s^2
        self.orbital_period_s = 2 * np.pi * np.sqrt((self.orbital_radius_km * 1000)**3 / mu_earth)
        self.orbital_period_h = self.orbital_period_s / 3600.0
        self.mean_motion_rad_s = 2 * np.pi / self.orbital_period_s
    
    def get_position_eci(self, time_s: float) -> np.ndarray:
        """
        Get satellite position in ECI coordinates at given time.
        Assumes circular orbit in equatorial plane.
        
        Args:
            time_s: Time in seconds
            
        Returns:
            Position vector [x, y, z] in km
        """
        # True anomaly at time (radiians)
        true_anomaly_rad = self.true_anomaly_deg * np.pi / 180.0 + self.mean_motion_rad_s * time_s
        
        # Position in orbital plane
        x = self.orbital_radius_km * np.cos(true_anomaly_rad)
        y = self.orbital_radius_km * np.sin(true_anomaly_rad)
        z = 0.0  # Equatorial plane
        
        return np.array([x, y, z])
    
    def get_velocity_eci(self, time_s: float) -> np.ndarray:
        """
        Get satellite velocity in ECI coordinates at given time.
        
        Args:
            time_s: Time in seconds
            
        Returns:
            Velocity vector [vx, vy, vz] in km/s
        """
        true_anomaly_rad = self.true_anomaly_deg * np.pi / 180.0 + self.mean_motion_rad_s * time_s
        
        orbital_speed = self.orbital_radius_km * self.mean_motion_rad_s
        vx = -orbital_speed * np.sin(true_anomaly_rad)
        vy = orbital_speed * np.cos(true_anomaly_rad)
        vz = 0.0
        
        return np.array([vx, vy, vz])


class MissileSimulator:
    """Simulates a ballistic missile trajectory."""
    
    def __init__(self, initial_pos: np.ndarray, initial_vel: np.ndarray):
        """
        Initialize missile with position and velocity.
        
        Args:
            initial_pos: Initial position [x, y, z] in km
            initial_vel: Initial velocity [vx, vy, vz] in km/s
        """
        self.initial_pos = initial_pos.copy()
        self.initial_vel = initial_vel.copy()
        self.gravity = 0.00981  # km/s^2 (downward, along -z for simplicity)
    
    def get_position(self, time_s: float) -> np.ndarray:
        """
        Get missile position at given time under constant acceleration (gravity).
        
        Args:
            time_s: Time in seconds
            
        Returns:
            Position vector [x, y, z] in km
        """
        pos = self.initial_pos + self.initial_vel * time_s
        pos[2] -= 0.5 * self.gravity * time_s**2  # Gravity in -z direction
        return pos
    
    def get_velocity(self, time_s: float) -> np.ndarray:
        """
        Get missile velocity at given time.
        
        Args:
            time_s: Time in seconds
            
        Returns:
            Velocity vector [vx, vy, vz] in km/s
        """
        vel = self.initial_vel.copy()
        vel[2] -= self.gravity * time_s  # Gravity acceleration
        return vel
    
    def get_acceleration(self, time_s: float) -> np.ndarray:
        """
        Get missile acceleration at given time (constant gravity).
        
        Args:
            time_s: Time in seconds
            
        Returns:
            Acceleration vector [ax, ay, az] in km/s^2
        """
        return np.array([0.0, 0.0, -self.gravity])


def simulate_measurements(
    satellites: List[Satellite],
    missile: MissileSimulator,
    time_array_s: np.ndarray,
    measurement_noise_std: float = 0.001
) -> Tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Simulate satellite measurements and missile truth trajectory.
    
    Args:
        satellites: List of Satellite objects
        missile: MissileSimulator object
        time_array_s: Array of times to simulate (seconds)
        measurement_noise_std: Standard deviation of LOS vector measurement noise
        
    Returns:
        Tuple of (measurements_dict, truth_dataframe)
        measurements_dict: Dictionary with DataFrames for each satellite
        truth_dataframe: DataFrame with true missile state
    """
    measurements_dict: dict[int, list[dict[str,float]]] = {}
    truth_data: List[dict[str,float]] = []
    
    for t in time_array_s:
        # Get true missile state
        missile_pos = missile.get_position(t)
        missile_vel = missile.get_velocity(t)
        missile_acc = missile.get_acceleration(t)
        
        truth_data.append({
            'time_s': t,
            'pos_x_km': missile_pos[0],
            'pos_y_km': missile_pos[1],
            'pos_z_km': missile_pos[2],
            'vel_x_km_s': missile_vel[0],
            'vel_y_km_s': missile_vel[1],
            'vel_z_km_s': missile_vel[2],
            'acc_x_km_s2': missile_acc[0],
            'acc_y_km_s2': missile_acc[1],
            'acc_z_km_s2': missile_acc[2],
        })
        
        # Get measurements from each satellite
        for sat in satellites:
            sat_pos = sat.get_position_eci(t)
            
            # Line-of-sight vector from satellite to missile
            los_vector = missile_pos - sat_pos
            los_distance = np.linalg.norm(los_vector)
            los_unit_vector = los_vector / los_distance if los_distance > 0 else np.array([0, 0, 0])
            
            # Add measurement noise
            noise = np.random.normal(0, measurement_noise_std, 3)
            los_measured = los_unit_vector + noise
            los_measured /= np.linalg.norm(los_measured)  # Normalize back to unit vector
            
            # Store measurement
            if sat.sat_id not in measurements_dict:
                measurements_dict[sat.sat_id] = []
            
            measurements_dict[sat.sat_id].append({
                'time_s': t,
                'sat_pos_x_km': sat_pos[0],
                'sat_pos_y_km': sat_pos[1],
                'sat_pos_z_km': sat_pos[2],
                'los_x': los_measured[0],
                'los_y': los_measured[1],
                'los_z': los_measured[2],
            })
    
    # Convert to DataFrames
    truth_df = pd.DataFrame(truth_data)
    measurements_dfs = {sat_id: pd.DataFrame(meas) for sat_id, meas in measurements_dict.items()}
    
    return measurements_dfs, truth_df
