import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the centroid history data from the pickle file
centroid_output_file = 'centroid_history_results.pkl'

with open(centroid_output_file, 'rb') as f:
    centroid_history = pickle.load(f)

# Load the particle data to access their mass and events
particle_output_file = 'particle_tracking_results.pkl'

with open(particle_output_file, 'rb') as f:
    particle_data = pickle.load(f)

# Define a mass threshold for filtering particles
mass_threshold = 1000  # You can adjust this threshold

# Define a sampling ratio for event downsampling
sampling_ratio = 0.1  # Use 10% of the events for plotting

# Flag to determine whether to plot individual events
plot_events = True  # Set to False to skip plotting events

# Create a 3D plot using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add each particle's trajectory and events to the plot
for particle_id, history in centroid_history.items():
    # Check if the particle's mass is above the threshold
    if particle_data[particle_id]['mass'] >= mass_threshold:
        # Unpack the history into separate lists
        times, centroids = zip(*history)
        centroids = np.array(centroids)

        # Convert times to milliseconds for visualization
        times = np.array(times) * 1e-3

        # Plot the trajectory as a line
        ax.plot(times, centroids[:, 0], centroids[:, 1], label=f'Particle {particle_id} Trajectory')

        # Plot individual events if the flag is set
        if plot_events:
            event_coords = np.array(particle_data[particle_id]['events'])
            event_times = event_coords[:, 2] * 1e-3  # Convert event times to milliseconds

            # Downsample the events
            num_events = len(event_coords)
            sample_size = int(num_events * sampling_ratio)
            if sample_size > 0:
                sampled_indices = np.random.choice(num_events, sample_size, replace=False)
                sampled_events = event_coords[sampled_indices]
                sampled_event_times = event_times[sampled_indices]
                
                # Scatter plot for sampled events
                ax.scatter(sampled_event_times, sampled_events[:, 0], sampled_events[:, 1], alpha=0.3, marker='.')

# Set axis labels
ax.set_ylabel('X Coordinate')
ax.set_zlabel('Y Coordinate')
ax.set_xlabel('Time (milliseconds)')

# Set axis limits to maintain aspect ratio
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1280])
ax.set_zlim([0, 720])

# Show the plot
plt.show()