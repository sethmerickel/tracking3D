"""
Interactive 3D visualization module for missile tracking.
Uses Plotly for self-contained, offline 3D visualization with time controls.
Includes textured Earth visualization.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Tuple


def create_earth_sphere() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a textured Earth sphere for visualization.
    
    Returns:
        Tuple of (x, y, z, color) arrays for Earth sphere
    """
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    earth_radius = 6371  # km
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create a simple color texture (blue ocean, green/brown land)
    # This creates a simple Earth-like appearance
    color = np.zeros_like(x)
    
    # Longitude-based coloring (simplified)
    for i in range(len(u)):
        for j in range(len(v)):
            # Create land masses pattern
            lon_pattern = np.sin(u[i]) * np.cos(v[j])
            lat_pattern = np.cos(u[i]) * np.sin(v[j])
            
            # Combine patterns for land/ocean appearance
            if (lon_pattern**2 + lat_pattern**2) > 0.3:
                color[i, j] = 0.2  # Ocean (blue)
            else:
                color[i, j] = 0.5  # Land (green/brown)
    
    return x, y, z, color


class InteractiveTrajectoryVisualizer:
    """Creates interactive 3D visualization of missile tracking with satellites and LOS vectors."""
    
    def __init__(
        self,
        truth_df: pd.DataFrame,
        estimates_df: pd.DataFrame,
        measurements_dfs: Dict[int, pd.DataFrame],
        satellite_positions: Dict[int, Tuple[float, float, float]]
    ):
        """
        Initialize visualizer.
        
        Args:
            truth_df: Truth trajectory DataFrame
            estimates_df: Estimated trajectory DataFrame
            measurements_dfs: Dictionary of measurement DataFrames per satellite
            satellite_positions: Dictionary mapping sat_id to (lat, lon, alt) or position
        """
        self.truth_df = truth_df
        self.estimates_df = estimates_df
        self.measurements_dfs = measurements_dfs
        self.satellite_positions = satellite_positions
        
        # Get unique times
        self.times = np.sort(truth_df['time_s'].unique())
        self.num_frames = len(self.times)
    
    def create_interactive_3d_plot(self) -> go.Figure:
        """
        Create interactive 3D plot with textured Earth and time slider.
        
        Returns:
            Plotly figure with interactive controls
        """
        # Create figure
        fig = go.Figure()
        
        # Add textured Earth sphere
        x_earth, y_earth, z_earth, color_earth = create_earth_sphere()
        
        fig.add_trace(go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            surfacecolor=color_earth,
            colorscale=[[0, 'rgb(0, 100, 255)'],      # Blue for ocean
                       [0.5, 'rgb(34, 139, 34)'],     # Green for land
                       [1, 'rgb(139, 90, 43)']],      # Brown for mountains
            showscale=False,
            hoverinfo='skip',
            name='Earth',
            opacity=0.9
        ))
        
        # Create frames for animation
        frames = []
        
        for frame_idx, t in enumerate(self.times):
            frame_data = []
            
            # Get missile positions at this time
            truth_at_t = self.truth_df[self.truth_df['time_s'] == t]
            est_at_t = self.estimates_df[self.estimates_df['time_s'] == t]
            
            if len(truth_at_t) == 0 or len(est_at_t) == 0:
                continue
            
            # Truth trajectory line (all history up to this time)
            truth_hist = self.truth_df[self.truth_df['time_s'] <= t]
            frame_data.append(
                go.Scatter3d(
                    x=truth_hist['pos_x_km'],
                    y=truth_hist['pos_y_km'],
                    z=truth_hist['pos_z_km'],
                    mode='lines',
                    name='Truth Trajectory',
                    line=dict(color='yellow', width=3),
                    hovertemplate='Truth<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
                )
            )
            
            # Estimate trajectory line
            est_hist = self.estimates_df[self.estimates_df['time_s'] <= t]
            frame_data.append(
                go.Scatter3d(
                    x=est_hist['est_pos_x_km'],
                    y=est_hist['est_pos_y_km'],
                    z=est_hist['est_pos_z_km'],
                    mode='lines',
                    name='Estimate Trajectory',
                    line=dict(color='orange', width=2, dash='dash'),
                    hovertemplate='Estimate<br>X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<extra></extra>'
                )
            )
            
            # Current truth position
            frame_data.append(
                go.Scatter3d(
                    x=truth_at_t['pos_x_km'].values,
                    y=truth_at_t['pos_y_km'].values,
                    z=truth_at_t['pos_z_km'].values,
                    mode='markers',
                    name='Truth Position',
                    marker=dict(size=12, color='yellow', symbol='diamond', 
                               line=dict(color='white', width=2)),
                    hovertemplate='Truth Position<br>Time: ' + f'{t:.1f}s<extra></extra>'
                )
            )
            
            # Current estimate position
            frame_data.append(
                go.Scatter3d(
                    x=est_at_t['est_pos_x_km'].values,
                    y=est_at_t['est_pos_y_km'].values,
                    z=est_at_t['est_pos_z_km'].values,
                    mode='markers',
                    name='Estimate Position',
                    marker=dict(size=10, color='orange', symbol='circle',
                               line=dict(color='white', width=2)),
                    hovertemplate='Estimate Position<br>Time: ' + f'{t:.1f}s<extra></extra>'
                )
            )
            
            # Satellite positions and LOS vectors
            sat_count = 0
            for sat_id, meas_df in self.measurements_dfs.items():
                meas_at_t = meas_df[meas_df['time_s'] == t]
                
                if len(meas_at_t) > 0:
                    sat_count += 1
                    sat_pos_x = meas_at_t['sat_pos_x_km'].values[0]
                    sat_pos_y = meas_at_t['sat_pos_y_km'].values[0]
                    sat_pos_z = meas_at_t['sat_pos_z_km'].values[0]
                    
                    missile_pos_x = truth_at_t['pos_x_km'].values[0]
                    missile_pos_y = truth_at_t['pos_y_km'].values[0]
                    missile_pos_z = truth_at_t['pos_z_km'].values[0]
                    
                    # Satellite position (only add trace once per satellite in frame)
                    if sat_count <= len(self.measurements_dfs):
                        frame_data.append(
                            go.Scatter3d(
                                x=[sat_pos_x],
                                y=[sat_pos_y],
                                z=[sat_pos_z],
                                mode='markers+text',
                                name=f'Satellite {sat_id}',
                                marker=dict(size=8, color='cyan', symbol='square',
                                           line=dict(color='white', width=1)),
                                text=[f'Sat {sat_id}'],
                                textposition='top center',
                                hovertemplate=f'Satellite {sat_id}<br>Distance from Earth center: %{{customdata}}<extra></extra>',
                                customdata=[np.sqrt(sat_pos_x**2 + sat_pos_y**2 + sat_pos_z**2)]
                            )
                        )
                    
                    # Line of sight vector (from satellite to missile)
                    frame_data.append(
                        go.Scatter3d(
                            x=[sat_pos_x, missile_pos_x],
                            y=[sat_pos_y, missile_pos_y],
                            z=[sat_pos_z, missile_pos_z],
                            mode='lines',
                            name=f'LOS Sat {sat_id}',
                            line=dict(color='lime', width=1.5, dash='dot'),
                            showlegend=(sat_id == 0),  # Only show legend for first LOS
                            hoverinfo='skip'
                        )
                    )
            
            frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
        
        # Set initial frame data (include Earth)
        if len(frames) > 0:
            fig.add_traces(frames[0].data)
        
        fig.frames = frames
        
        # Add play and pause buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[None, {
                                'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 0}
                            }]
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        )
                    ],
                    x=0.0,
                    y=1.08,
                    xanchor='left',
                    yanchor='top'
                )
            ],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'y': -0.05,
                'xanchor': 'left',
                'x': 0.0,
                'len': 0.9,
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'currentvalue': {
                    'prefix': 'Time: ',
                    'visible': True,
                    'xanchor': 'center',
                    'suffix': ' s'
                },
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'method': 'animate',
                        'label': f'{self.times[i]:.1f}'
                    }
                    for i, f in enumerate(frames)
                ]
            }]
        )
        
        # Update layout with Earth-centered scene
        fig.update_layout(
            title='Satellite Tracking Visualization - Earth-Centered View',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8),
                    center=dict(x=0, y=0, z=0)
                ),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray', zeroline=False),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray', zeroline=False),
                zaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray', zeroline=False),
                bgcolor='rgba(0, 0, 0, 0.9)'  # Dark space background
            ),
            width=1400,
            height=900,
            showlegend=True,
            hovermode='closest',
            margin=dict(l=0, r=0, b=100, t=100),
            paper_bgcolor='rgb(20, 20, 40)',
            plot_bgcolor='rgba(0, 0, 0, 0.8)'
        )
        
        return fig
    
    def save_interactive_html(self, filepath: str) -> None:
        """
        Create and save interactive visualization to HTML file.
        
        Args:
            filepath: Path to save HTML file
        """
        fig = self.create_interactive_3d_plot()
        fig.write_html(filepath, config={'responsive': True})
        print(f"Interactive visualization saved to: {filepath}")
    
    def show_interactive(self) -> None:
        """Display interactive visualization."""
        fig = self.create_interactive_3d_plot()
        fig.show()
