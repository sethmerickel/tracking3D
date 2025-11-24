#!/usr/bin/env python3
"""
Diagnostic script to check satellite movement in the simulation data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_satellite_movement(output_dir='output'):
    """Analyze satellite positions over time to verify movement."""
    
    output_path = Path(output_dir)
    
    print("Satellite Movement Analysis")
    print("=" * 60)
    
    # Find all measurement files
    meas_files = sorted(output_path.glob('measurements_sat*.csv'))
    
    if not meas_files:
        print("No measurement files found!")
        return
    
    for meas_file in meas_files:
        sat_id = int(meas_file.stem.replace('measurements_sat', ''))
        df = pd.read_csv(meas_file)
        
        print(f"\nSatellite {sat_id}:")
        print(f"  Total measurements: {len(df)}")
        
        if len(df) > 0:
            # Get first and last positions
            first_pos = df.iloc[0][['sat_pos_x_km', 'sat_pos_y_km', 'sat_pos_z_km']].values
            last_pos = df.iloc[-1][['sat_pos_x_km', 'sat_pos_y_km', 'sat_pos_z_km']].values
            
            first_r = np.linalg.norm(first_pos)
            last_r = np.linalg.norm(last_pos)
            
            displacement = np.linalg.norm(last_pos - first_pos)
            
            print(f"  Time range: {df['time_s'].min():.1f}s to {df['time_s'].max():.1f}s")
            print(f"  First position: ({first_pos[0]:.1f}, {first_pos[1]:.1f}, {first_pos[2]:.1f})")
            print(f"  Last position:  ({last_pos[0]:.1f}, {last_pos[1]:.1f}, {last_pos[2]:.1f})")
            print(f"  Distance from Earth center: {first_r:.1f} km → {last_r:.1f} km")
            print(f"  Total displacement: {displacement:.1f} km")
            
            # Check if satellite moved
            if displacement < 1:
                print(f"  ⚠️  WARNING: Satellite barely moved ({displacement:.2f} km)")
            else:
                print(f"  ✓ Satellite moved {displacement:.1f} km during simulation")


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
    analyze_satellite_movement(output_dir)
