"""
QT 3D VISUALIZATION GUIDE
=========================

A high-performance, interactive 3D visualization of missile tracking using
PyQt5 and OpenGL. This replaces the Plotly web-based visualization with a
native desktop application.

Features
========

✓ Real-time 3D rendering with OpenGL
✓ Textured Earth sphere at origin
✓ True missile trajectory (yellow line)
✓ Estimated missile trajectory (orange dashed line)
✓ Satellites orbiting Earth (cyan dots)
✓ Line-of-sight vectors (lime green lines)
✓ Play/Pause animation controls
✓ Time slider for frame-by-frame navigation
✓ Mouse rotation, zoom, pan controls
✓ Full-screen support
✓ High performance (60+ FPS)

Installation
============

1. Make sure uv is installed:
   curl -LsSf https://astral.sh/uv/install.sh | sh

2. Install dependencies:
   uv pip install PyQt5 PyOpenGL

   Or install all:
   uv pip install -r requirements.txt

3. Run simulation:
   python main.py

4. Launch visualization:
   python run_qt_visualization.py output/

Usage
=====

Mouse Controls:
- Left-click + drag: Rotate view
- Scroll wheel: Zoom in/out
- Right-click + drag: Pan (optional)

Keyboard Shortcuts:
- Space: Play/Pause
- Left arrow: Previous frame
- Right arrow: Next frame
- Home: Jump to start
- End: Jump to end

UI Controls:
- Play button (▶/⏸): Start/stop animation
- Time slider: Jump to specific time
- Time display: Shows current time in seconds

Keyboard Shortcuts (Not Yet Implemented - Future)
- R: Reset view
- F: Fit to view
- H: Show help

Performance Tips
================

1. For large simulations (1000+ frames):
   - Use frame subsampling if needed
   - Close other applications
   - Reduce window size if performance is slow

2. For smooth animation:
   - Recommended minimum: 60 FPS
   - Most systems will achieve 30-60 FPS
   - Adjust update interval in code if needed

3. For accurate visualization:
   - Use high-quality graphics settings
   - Full-screen or maximized window recommended
   - External monitor provides best experience

Troubleshooting
===============

Issue: "ModuleNotFoundError: No module named 'PyQt5'"
Fix: Install PyQt5:
  uv pip install PyQt5

Issue: "ModuleNotFoundError: No module named 'OpenGL'"
Fix: Install PyOpenGL:
  uv pip install PyOpenGL

Issue: "RuntimeError: Could not find any Qt installation"
Fix on macOS:
  brew install python3 qt5
  Then reinstall PyQt5:
  uv pip install --upgrade PyQt5

Issue: Window is black or not rendering
Fix:
  1. Try clicking in the window to focus it
  2. Try resizing the window
  3. Update GPU drivers
  4. Try running with software rendering:
     export QT_XCB_GL_INTEGRATION=xcb_glx
     python run_qt_visualization.py output/

Issue: Animation is very slow
Fix:
  1. Close other applications
  2. Reduce window size
  3. Increase animation interval (edit run_qt_visualization.py)
  4. Try in full-screen mode

File Structure
==============

The visualization loads data from the output directory:

output/
├── truth_trajectory.csv          # Required
├── estimates_trajectory.csv      # Required
├── measurements_sat0.csv         # Required
├── measurements_sat1.csv         # Required
└── measurements_sat2.csv         # etc.

All files must be present in the output directory.

Command Line Usage
==================

Basic usage:
  python run_qt_visualization.py output/

Custom output directory:
  python run_qt_visualization.py /path/to/output/

From project root:
  python run_qt_visualization.py

Advanced Usage
==============

Modify animation speed:
Open interactive_viz_qt.py and change:
  self.timer.start(100)  # 100ms = 10 FPS
  
To 50 for faster (20 FPS):
  self.timer.start(50)

Modify initial camera position:
Edit initializeGL() method in Earth3DVisualizer class

View Technical Details
======================

The Qt visualization provides better control than web-based visualization:

Advantages:
✓ Native performance (not limited by browser)
✓ Direct hardware acceleration
✓ Better control over rendering
✓ Can run offline without internet
✓ Can capture screenshots/video

Disadvantages vs Plotly:
✗ Requires additional dependencies (Qt, OpenGL)
✗ Cannot share directly (requires code/binary)
✗ Platform-specific rendering

Exporting Results
=================

To save a screenshot:
1. Press Print Screen or Cmd+Shift+3 (macOS)
2. Or implement screenshot functionality (future)

To record video:
1. Use screen recording software (ffmpeg, OBS, etc.)
2. Or implement video export (future)

Architecture
============

Components:

1. Earth3DVisualizer (QGLWidget)
   - OpenGL rendering engine
   - Handles mouse/keyboard input
   - Manages 3D transformations

2. InteractiveTrajectoryWindow (QMainWindow)
   - Main application window
   - Controls playback (play/pause, slider)
   - Loads and manages data

3. run_qt_visualization.py
   - Entry point
   - Loads CSV data
   - Launches Qt application

Data Flow:

CSV Files
   ↓
run_qt_visualization.py (loads CSV)
   ↓
InteractiveTrajectoryWindow (prepares data)
   ↓
Earth3DVisualizer (renders 3D scene)
   ↓
Qt Event Loop (handles user input)

Future Enhancements
===================

Planned features:
- [ ] Keyboard shortcuts for playback control
- [ ] Reset/home camera button
- [ ] Fit-to-view button
- [ ] Screenshot/video recording
- [ ] Statistics panel showing tracking metrics
- [ ] Error envelope visualization
- [ ] Measurement visibility toggle
- [ ] Trajectory history length control
- [ ] Multiple view layouts
- [ ] VR/3D stereoscopic support

Example Session
===============

1. Run simulation:
   $ python main.py
   ✓ Generates CSV and PNG files in output/

2. Launch Qt viewer:
   $ python run_qt_visualization.py output/
   ✓ Window opens with Earth and trajectories

3. Explore:
   - Click and drag to rotate view
   - Scroll to zoom in/out
   - Click Play to animate
   - Move slider to jump in time

4. Analyze:
   - Watch how estimate diverges from truth
   - See satellites orbiting Earth
   - Observe LOS vectors to missile
   - Measure error visually
"""