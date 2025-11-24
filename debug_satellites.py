#!/usr/bin/env python3
"""
Debug script to verify satellite positions are changing across frames.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def debug_satellite_positions(output_dir='output'):
    """Check if satellites are actually moving in the data."""
    
    output_path = Path(output_dir)
    
    print("Satellite Position Debug")
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
        
        if len(df) >= 3:
            # Sample positions at different times
            indices = [0, len(df)//2, -1]
            print(f"  Sample positions:")
            
            for idx in indices:
                row = df.iloc[idx]
                time = row['time_s']
                pos_x = row['sat_pos_x_km']
                pos_y = row['sat_pos_y_km']
                pos_z = row['sat_pos_z_km']
                radius = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
                
                print(f"    t={time:6.1f}s: ({pos_x:7.1f}, {pos_y:7.1f}, {pos_z:7.1f}) r={radius:7.1f}")
        
        # Check for actual position changes
        if len(df) > 1:
            first_pos = df.iloc[0][['sat_pos_x_km', 'sat_pos_y_km', 'sat_pos_z_km']].values
            last_pos = df.iloc[-1][['sat_pos_x_km', 'sat_pos_y_km', 'sat_pos_z_km']].values
            displacement = np.linalg.norm(last_pos - first_pos)
            
            if displacement > 10:
                print(f"  ✓ Satellite moved {displacement:.1f} km")
            else:
                print(f"  ⚠️  Satellite barely moved: {displacement:.2f} km")


if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'
    debug_satellite_positions(output_dir)
