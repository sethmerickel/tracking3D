"""
Qt-based interactive 3D visualization module for missile tracking.
Uses PyOpenGL for high-performance 3D rendering with Earth, satellites, and trajectories.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, Qt as QtEnums
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_LIGHTING,
    GL_LIGHT0, GL_COLOR_MATERIAL, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
    GL_POSITION, GL_AMBIENT, GL_DIFFUSE, GL_SPECULAR, GL_PROJECTION,
    GL_MODELVIEW, GL_POINTS, GL_LINE_STRIP, GL_FILL, glClear, glEnable,
    glColorMaterial, glLight, glMatrixMode, glLoadIdentity, glTranslatef,
    glRotatef, glScalef, glColor3f, glPointSize, glBegin, glVertex3f, glEnd,
    glLineWidth, glClearColor, glViewport, GL_FILL
)
from OpenGL.GLU import *
import math


class Earth3DVisualizer(QGLWidget):
    """OpenGL widget for 3D visualization of missile tracking with Earth."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 1.0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Trajectory data
        self.truth_trajectory = None
        self.estimate_trajectory = None
        self.satellite_positions = None
        self.los_vectors = None
        
        self.current_frame = 0
        self.total_frames = 0
        
    def initializeGL(self):
        """Initialize OpenGL settings."""
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        glLight(GL_LIGHT0, GL_POSITION, (5, 5, 5, 0))
        glLight(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
        glLight(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
        glLight(GL_LIGHT0, GL_SPECULAR, (1, 1, 1, 1))
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width() or 800) / (self.height() or 600), 0.1, 50000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def resizeGL(self, w, h):
        """Handle window resize."""
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h if h > 0 else 1, 0.1, 50000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def paintGL(self):
        """Render the 3D scene."""
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        glLoadIdentity()
        
        # Move camera back and apply rotations
        glTranslatef(0, 0, -25000)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glScalef(self.zoom, self.zoom, self.zoom)
        
        # Draw Earth
        self._draw_earth()
        
        # Draw satellite orbital trails (all history)
        self._draw_satellite_trails()
        
        # Draw trajectories
        if self.truth_trajectory is not None:
            self._draw_trajectory(self.truth_trajectory[:self.current_frame+1], (1, 1, 0))  # Yellow
        
        if self.estimate_trajectory is not None:
            self._draw_trajectory(self.estimate_trajectory[:self.current_frame+1], (1, 0.5, 0))  # Orange
        
        # Draw satellites and LOS vectors at current frame
        if self.satellite_positions is not None and self.current_frame < len(self.satellite_positions):
            self._draw_satellites_and_los(self.current_frame)
        
        # Draw current positions
        if self.truth_trajectory is not None and self.current_frame < len(self.truth_trajectory):
            pos = self.truth_trajectory[self.current_frame]
            glPointSize(12)
            glColor3f(1, 1, 0)  # Yellow
            glBegin(GL_POINTS)
            glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
        
        if self.estimate_trajectory is not None and self.current_frame < len(self.estimate_trajectory):
            pos = self.estimate_trajectory[self.current_frame]
            glPointSize(10)
            glColor3f(1, 0.5, 0)  # Orange
            glBegin(GL_POINTS)
            glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
    
    def _draw_earth(self):
        """Draw Earth as a textured sphere."""
        glColor3f(0.2, 0.4, 0.8)  # Blue
        quad = gluNewQuadric()
        gluQuadricDrawStyle(quad, GLU_FILL)
        gluSphere(quad, 6371, 32, 32)
    
    def _draw_trajectory(self, trajectory, color):
        """Draw trajectory line."""
        glColor3f(*color)
        glLineWidth(2)
        glBegin(GL_LINE_STRIP)
        for point in trajectory:
            glVertex3f(point[0], point[1], point[2])
        glEnd()
    
    def _draw_satellite_trails(self):
        """Draw satellite position history trails."""
        if self.satellite_positions is None or self.current_frame == 0:
            return
        
        # Group satellite positions by ID across time (only up to current frame)
        sat_trails: Dict[int, list] = {}
        
        for frame_idx in range(self.current_frame + 1):
            if frame_idx >= len(self.satellite_positions):
                break
            
            sats_at_frame = self.satellite_positions[frame_idx]
            for sat_id, pos in sats_at_frame.items():
                if sat_id not in sat_trails:
                    sat_trails[sat_id] = []
                sat_trails[sat_id].append((pos[0], pos[1], pos[2]))
        
        # Draw trails for each satellite with minimum 2 points
        for sat_id, trail in sat_trails.items():
            if len(trail) > 1:
                glColor3f(0, 0.7, 0.7)  # Light cyan
                glLineWidth(1)
                glBegin(GL_LINE_STRIP)
                for x, y, z in trail:
                    glVertex3f(x, y, z)
                glEnd()
    
    def _draw_satellites_and_los(self, frame_idx):
        """Draw satellites and LOS vectors at current frame."""
        if self.satellite_positions is None or (frame_idx >= len(self.satellite_positions)):
            return
        
        sats = self.satellite_positions[frame_idx]
        
        # Skip if no satellites at this frame
        if not sats:
            return
        
        # Draw satellites with larger markers
        for sat_id, pos in sats.items():
            glColor3f(0, 1, 1)  # Cyan
            glPointSize(15)  # Larger size for visibility
            glBegin(GL_POINTS)
            glVertex3f(pos[0], pos[1], pos[2])
            glEnd()
            
            # Draw satellite label/orbit trail (small circle)
            glColor3f(0, 0.5, 1)  # Darker cyan
            glLineWidth(1)
            glBegin(GL_LINE_STRIP)
            for angle in np.linspace(0, 2*np.pi, 16):
                radius = np.sqrt(pos[0]**2 + pos[1]**2)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = pos[2]
                glVertex3f(x, y, z)
            glEnd()
        
        # Draw LOS vectors
        if self.truth_trajectory is not None and frame_idx < len(self.truth_trajectory):
            missile_pos = self.truth_trajectory[frame_idx]
            for sat_id, sat_pos in sats.items():
                glColor3f(0, 1, 0)  # Lime green
                glLineWidth(2)  # Thicker for visibility
                glBegin(GL_LINE_STRIP)
                glVertex3f(sat_pos[0], sat_pos[1], sat_pos[2])
                glVertex3f(missile_pos[0], missile_pos[1], missile_pos[2])
                glEnd()
    
    def mousePressEvent(self, event):
        """Handle mouse press."""
        self.last_mouse_x = event.x()
        self.last_mouse_y = event.y()
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for rotation."""
        dx = event.x() - self.last_mouse_x
        dy = event.y() - self.last_mouse_y
        
        self.rotation_y += dx * 0.5
        self.rotation_x += dy * 0.5
        
        self.last_mouse_x = event.x()
        self.last_mouse_y = event.y()
        
        self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        self.zoom *= zoom_factor
        self.zoom = max(0.1, min(10, self.zoom))
        self.update()
    
    def set_trajectory_data(self, truth_traj, estimate_traj, sat_positions, los_vectors, num_frames):
        """Set trajectory data for visualization."""
        self.truth_trajectory = truth_traj
        self.estimate_trajectory = estimate_traj
        self.satellite_positions = sat_positions
        self.los_vectors = los_vectors
        self.total_frames = num_frames
        self.current_frame = 0
    
    def set_frame(self, frame):
        """Set current frame to display."""
        self.current_frame = min(frame, self.total_frames - 1)
        self.update()


class InteractiveTrajectoryWindow(QMainWindow):
    """Main window for interactive trajectory visualization."""
    
    def __init__(self, truth_df, estimates_df, measurements_dfs):
        super().__init__()
        self.truth_df = truth_df
        self.estimates_df = estimates_df
        self.measurements_dfs = measurements_dfs
        
        self.setWindowTitle('3D Missile Tracking Visualization')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create 3D visualization widget
        self.visualizer = Earth3DVisualizer()
        layout.addWidget(self.visualizer)
        
        # Create control panel
        control_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton('▶ Play')
        self.play_button.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_button)
        
        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.sliderMoved.connect(self.on_slider_moved)
        self.time_slider.valueChanged.connect(self.on_value_changed)
        control_layout.addWidget(self.time_slider)
        
        # Time label
        self.time_label = QLabel('Time: 0.0 s')
        control_layout.addWidget(self.time_label)
        
        layout.addLayout(control_layout)
        
        # Prepare data
        self._prepare_trajectory_data()
        
        # Setup timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        
        # Set initial frame
        self.visualizer.set_frame(0)
    
    def _prepare_trajectory_data(self):
        """Prepare trajectory data for visualization."""
        # Get unique times
        times = np.sort(self.truth_df['time_s'].unique())
        num_frames = len(times)
        
        # Extract truth trajectory
        truth_traj = self.truth_df[['pos_x_km', 'pos_y_km', 'pos_z_km']].values
        
        # Extract estimate trajectory
        estimate_traj = self.estimates_df[['est_pos_x_km', 'est_pos_y_km', 'est_pos_z_km']].values
        
        # Extract satellite positions and LOS vectors
        sat_positions = []
        los_vectors = []
        
        for t in times:
            sats_at_t = {}
            los_at_t = {}
            
            for sat_id, meas_df in self.measurements_dfs.items():
                meas = meas_df[meas_df['time_s'] == t]
                if len(meas) > 0:
                    sat_pos = np.array([
                        meas['sat_pos_x_km'].values[0],
                        meas['sat_pos_y_km'].values[0],
                        meas['sat_pos_z_km'].values[0]
                    ])
                    sats_at_t[sat_id] = sat_pos
                    los_at_t[sat_id] = np.array([
                        meas['los_x'].values[0],
                        meas['los_y'].values[0],
                        meas['los_z'].values[0]
                    ])
            
            sat_positions.append(sats_at_t)
            los_vectors.append(los_at_t)
        
        self.visualizer.set_trajectory_data(
            truth_traj,
            estimate_traj,
            sat_positions,
            los_vectors,
            num_frames
        )
        
        self.times = times
        self.time_slider.setMaximum(num_frames - 1)
        self.time_slider.setSliderPosition(0)
    
    def toggle_play(self):
        """Toggle play/pause."""
        if self.is_playing:
            self.timer.stop()
            self.play_button.setText('▶ Play')
            self.is_playing = False
        else:
            self.timer.start(50)  # Update every 50ms (20 FPS) for faster animation
            self.play_button.setText('⏸ Pause')
            self.is_playing = True
    
    def next_frame(self):
        """Advance to next frame."""
        current_frame = self.time_slider.value()
        if current_frame < self.time_slider.maximum():
            self.time_slider.setValue(current_frame + 1)
        else:
            self.toggle_play()  # Stop at end
    
    def on_slider_moved(self, value):
        """Handle slider movement."""
        if value < len(self.times):
            self.time_label.setText(f'Time: {self.times[value]:.1f} s')
            self.visualizer.set_frame(value)
    
    def on_value_changed(self, value):
        """Handle value changes (both user and programmatic)."""
        if value < len(self.times):
            self.time_label.setText(f'Time: {self.times[value]:.1f} s')
            self.visualizer.set_frame(value)
