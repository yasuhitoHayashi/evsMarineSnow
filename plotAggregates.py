import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load the centroid history data from the pickle file
centroid_output_file = 'centroid_history_results.pkl'

with open(centroid_output_file, 'rb') as f:
    centroid_history = pickle.load(f)

# Load the particle data to access their mass
particle_output_file = 'particle_tracking_results.pkl'

with open(particle_output_file, 'rb') as f:
    particle_data = pickle.load(f)

# Define a mass threshold for filtering particles
mass_threshold = 1000  # You can adjust this threshold

# Create a 3D scatter plot using Plotly
fig = go.Figure()

# Add each particle's trajectory to the plot
for particle_id, history in centroid_history.items():
    # Check if the particle's mass is above the threshold
    if particle_data[particle_id]['mass'] >= mass_threshold:
        # Unpack the history into separate lists
        times, centroids = zip(*history)
        centroids = np.array(centroids)
        
        # Add the trajectory as a line plot
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=times,
            mode='lines',
            name=f'Particle {particle_id}'
        ))

# Update plot layout
fig.update_layout(
    scene=dict(
        xaxis_title='X Coordinate',
        yaxis_title='Y Coordinate',
        zaxis_title='Time (microseconds)'
    ),
    title=f'3D Trajectory of Particle Centroids Over Time (Mass ≥ {mass_threshold})',
    showlegend=True
)

# Show the plot
fig.show()