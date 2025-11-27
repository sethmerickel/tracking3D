"""
Missile tracking simulation using LEO satellites.
This module simulates satellite orbits and missile trajectories with LOS measurements,
including Earth occlusion effects.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class Satellite:
    """Models a satellite in circular LEO orbit."""
    
    def __init__(self, sat_id: int, altitude_km: float, longitude_deg: float, inclination_deg: float = 90.0):
        """
        Initialize satellite in circular orbit.
        
        Args:
            sat_id: Satellite identifier
            altitude_km: Altitude above Earth surface (km)
            longitude_deg: Initial longitude (degrees)
            inclination_deg: Orbit inclination (degrees), default 90° for polar orbit
        """
        self.sat_id = sat_id
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.longitude_deg = longitude_deg
        
        # Orbital parameters
        self.earth_radius_km = 6371.0
        self.orbital_radius_km = self.earth_radius_km + altitude_km
        
        # Mean motion (radians per second)
        # n = sqrt(mu / a^3), where mu = 398600.4418 km^3/s^2
        self.mu = 398600.4418  # Earth's gravitational parameter
        self.mean_motion = np.sqrt(self.mu / self.orbital_radius_km**3)
        
        # Orbital period in seconds
        self.period_s = 2 * np.pi / self.mean_motion
    
    def get_position_eci(self, time_s: float) -> np.ndarray:
        """
        Get satellite position in ECI coordinates at given time.
        
        Args:
            time_s: Time in seconds
            
        Returns:
            Position vector [x, y, z] in km (ECI frame)
        """
        # Mean anomaly at this time (radians)
        mean_anomaly = self.mean_motion * time_s
        
        # For circular orbits, true anomaly = mean anomaly
        true_anomaly = mean_anomaly
        
        # Position in orbital plane
        x_orb = self.orbital_radius_km * np.cos(true_anomaly)
        y_orb = self.orbital_radius_km * np.sin(true_anomaly)
        z_orb = 0.0
        
        # Convert inclination to radians
        inc_rad = np.radians(self.inclination_deg)
        
        # Convert initial longitude to radians
        lon_rad = np.radians(self.longitude_deg)
        
        # Rotation matrices
        # First rotate by inclination (around x-axis in orbital plane)
        x1 = x_orb
        y1 = y_orb * np.cos(inc_rad)
        z1 = y_orb * np.sin(inc_rad)
        
        # Then rotate by longitude (around z-axis in ECI frame)
        x_eci = x1 * np.cos(lon_rad) - y1 * np.sin(lon_rad)
        y_eci = x1 * np.sin(lon_rad) + y1 * np.cos(lon_rad)
        z_eci = z1
        
        return np.array([x_eci, y_eci, z_eci])


def is_los_occluded(sat_pos: np.ndarray, missile_pos: np.ndarray, earth_radius_km: float = 6371.0) -> bool:
    """
    Check if Earth occludes the line-of-sight between satellite and missile.
    
    Uses the closest point of approach method to determine if Earth blocks the LOS vector.
    
    Args:
        sat_pos: Satellite position [x, y, z] in km
        missile_pos: Missile position [x, y, z] in km
        earth_radius_km: Earth's radius (km)
        
    Returns:
        True if Earth occludes LOS, False otherwise
    """
    # Vector from satellite to missile
    los_vec = missile_pos - sat_pos
    los_distance = np.linalg.norm(los_vec)
    
    if los_distance < 1e-6:
        return False
    
    # Vector from Earth center to satellite
    sat_to_earth = -sat_pos  # Vector from sat to Earth center
    
    # Project satellite-to-earth vector onto LOS vector
    # This gives the parameter t of closest approach on the LOS line
    los_unit = los_vec / los_distance
    t = np.dot(sat_to_earth, los_unit) / los_distance
    
    # Closest point on LOS line to Earth center
    if t < 0:
        # Closest point is behind satellite, so satellite is between Earth and missile
        closest_point = sat_pos
    elif t > 1:
        # Closest point is beyond missile
        closest_point = missile_pos
    else:
        # Closest point is between satellite and missile
        closest_point = sat_pos + t * los_vec
    
    # Distance from Earth center to closest point on LOS line
    dist_to_los = np.linalg.norm(closest_point)
    
    # Check if LOS line intersects Earth
    # Account for Earth's radius plus small margin for tangent rays
    if dist_to_los < earth_radius_km + 1e-3:
        return True
    
    # Also check if missile is below Earth surface (shouldn't happen in normal trajectory)
    if np.linalg.norm(missile_pos) < earth_radius_km:
        return True
    
    return False


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
    
    def simulate_trajectory(self, times: np.ndarray) -> np.ndarray:
        """
        Simulate missile trajectory over time steps.
        
        Args:
            times: Array of time values in seconds
            
        Returns:
            Array of shape (len(times), 6) with [x, y, z, vx, vy, vz] at each time
        """
        trajectory = np.zeros((len(times), 6))
        for i, t in enumerate(times):
            trajectory[i, :3] = self.get_position(t)
            trajectory[i, 3:] = self.get_velocity(t)
        return trajectory


def simulate_measurements(
    num_satellites: int = 5,
    duration_s: float = 5400.0,
    dt_s: float = 1.0,
    measurement_noise_los: float = 0.001,
    include_occlusion: bool = False
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Simulate satellite measurements and missile truth trajectory.
    
    Args:
        num_satellites: Number of satellites
        duration_s: Simulation duration in seconds
        dt_s: Time step in seconds
        measurement_noise_los: Measurement noise on LOS components
        include_occlusion: Whether to include Earth occlusion
        
    Returns:
        Tuple of (measurements_dict, truth_df)
    """
    # Initialize satellites in cluster around 0° longitude
    satellites = []
    arc_span_deg = 120.0  # Spread satellites over 120° arc
    for i in range(num_satellites):
        longitude = -arc_span_deg/2 + (i / (num_satellites - 1)) * arc_span_deg
        sat = Satellite(
            sat_id=i,
            altitude_km=1000.0,
            longitude_deg=longitude,
            inclination_deg=90.0
        )
        satellites.append(sat)
    
    # Time vector
    times = np.arange(0, duration_s + dt_s, dt_s)
    
    # Missile trajectory (ballistic)
    # Initialize missile with launch position and velocity
    initial_pos = np.array([0.0, 0.0, 100.0])  # Launch from 100 km altitude
    initial_vel = np.array([7.0, 0.0, 2.0])    # Initial velocity in km/s
    missile_sim = MissileSimulator(initial_pos, initial_vel)
    truth_trajectory = missile_sim.simulate_trajectory(times)
    
    # Satellite measurements
    measurements = {}
    for sat in satellites:
        measurements[sat.sat_id] = []
    
    # Generate measurements at each time step
    for t in times:
        # Get missile position at this time
        missile_idx = int(np.round(t / dt_s))
        if missile_idx >= len(truth_trajectory):
            break
            
        missile_pos = truth_trajectory[missile_idx]
        
        # Get measurements from each satellite
        for sat in satellites:
            # Get satellite position at this time (KEY FIX: pass time to get updated position)
            sat_pos = sat.get_position_eci(t)
            
            # Line of sight vector
            los_vec = missile_pos[0:3] - sat_pos
            range_km = np.linalg.norm(los_vec)
            
            # Unit LOS vector
            if range_km > 0.1:
                los_unit = los_vec / range_km
            else:
                los_unit = np.array([0, 0, 0])
            
            # Check Earth occlusion
            if include_occlusion and is_los_occluded(sat_pos, missile_pos):
                continue
            
            # Add measurement noise
            los_noisy = los_unit + np.random.normal(0, measurement_noise_los, 3)
            
            measurements[sat.sat_id].append({
                'time_s': t,
                'sat_pos_x_km': sat_pos[0],
                'sat_pos_y_km': sat_pos[1],
                'sat_pos_z_km': sat_pos[2],
                'los_x': los_noisy[0],
                'los_y': los_noisy[1],
                'los_z': los_noisy[2],
            })
    
    # Convert to DataFrames
    for sat_id in measurements:
        measurements[sat_id] = pd.DataFrame(measurements[sat_id])
    
    # Truth trajectory DataFrame
    truth_df = pd.DataFrame({
        'time_s': times[:len(truth_trajectory)],
        'pos_x_km': truth_trajectory[:, 0],
        'pos_y_km': truth_trajectory[:, 1],
        'pos_z_km': truth_trajectory[:, 2],
        'vel_x_km_s': truth_trajectory[:, 3],
        'vel_y_km_s': truth_trajectory[:, 4],
        'vel_z_km_s': truth_trajectory[:, 5],
    })
    
    return measurements, truth_df
