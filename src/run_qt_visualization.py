#!/usr/bin/env python3
"""
Launch the Qt-based interactive 3D visualization of missile tracking.

Usage:
    python run_qt_visualization.py <path_to_output_directory>
    
Example:
    python run_qt_visualization.py output/
"""

import sys
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from interactive_viz_qt import InteractiveTrajectoryWindow


def main():
    """Load data and launch Qt visualization window."""
    
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = 'output'
    
    output_path = Path(output_dir)
    
    # Load data files
    print(f"Loading data from {output_dir}/...")
    
    try:
        truth_df = pd.read_csv(output_path / 'truth_trajectory.csv')
        estimates_df = pd.read_csv(output_path / 'estimates_trajectory.csv')
        print(f"  ✓ Loaded truth and estimates ({len(truth_df)} samples)")
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files in {output_dir}")
        print(f"  {e}")
        sys.exit(1)
    
    # Load measurement files
    measurements_dfs = {}
    sat_count = 0
    for sat_file in output_path.glob('measurements_sat*.csv'):
        sat_id = int(sat_file.stem.replace('measurements_sat', ''))
        measurements_dfs[sat_id] = pd.read_csv(sat_file)
        sat_count += 1
    
    print(f"  ✓ Loaded {sat_count} satellite measurement files")
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create visualization window
    print("Launching Qt 3D visualization...")
    window = InteractiveTrajectoryWindow(truth_df, estimates_df, measurements_dfs)
    window.show()
    
    print("✓ Ready! Use mouse to rotate, scroll to zoom, slider to navigate time")
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
