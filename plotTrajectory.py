import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the particle data from the pickle file
particle_output_file = 'particle_tracking_results_suruga_test_short.pkl'
#particle_output_file = 'particle_tracking_results_recording_2024-04-25_10-43-50.pkl'

with open(particle_output_file, 'rb') as f:
    particle_data = pickle.load(f)

# Define a threshold for filtering particles based on the number of events (instead of mass)
event_threshold = 1000  # You can adjust this threshold

# Define a sampling ratio for event downsampling
sampling_ratio = 0  # Use 10% of the events for plotting

# Flag to determine whether to plot individual events
plot_events = False  # Set to False to skip plotting events

# Create a 3D plot using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add each particle's trajectory and events to the plot
for particle_id, particle_info in particle_data.items():
    # Check if the particle has a centroid history
    if 'centroid_history' in particle_info:
        centroid_history = np.array(particle_info['centroid_history'])  # (time, centroid_x, centroid_y)

        # Ensure centroid history has enough points to plot
        if len(centroid_history) > 1:
            # Plot the full trajectory of the centroid as a line
            ax.plot(centroid_history[:, 0] * 1e-3, centroid_history[:, 1], centroid_history[:, 2], label=f'Particle {particle_id} Centroid Trajectory')

    # Plot individual events if the flag is set
    if plot_events:
        events = particle_info['events']
        event_coords = np.array(events)
        event_times = event_coords[:, 2] * 1e-3  # Convert event times to milliseconds

        # Downsample the events based on the sampling ratio
        num_events = len(events)
        if sampling_ratio < 1.0:
            sample_size = int(num_events * sampling_ratio)
            if sample_size > 0:
                sampled_indices = np.random.choice(num_events, sample_size, replace=False)
                sampled_events = event_coords[sampled_indices]
                sampled_event_times = event_times[sampled_indices]
                
                # Scatter plot for sampled events
                ax.scatter(sampled_event_times, sampled_events[:, 0], sampled_events[:, 1], alpha=0.3, marker='.')
        else:
            # Plot all events if sampling_ratio is 1 or more
            ax.scatter(event_times, event_coords[:, 0], event_coords[:, 1], alpha=0.3, marker='.')

# Set axis labels
ax.set_ylabel('X Coordinate')
ax.set_zlabel('Y Coordinate')
ax.set_xlabel('Time (milliseconds)')

# Set axis limits to maintain aspect ratio (adjust to your data)
#ax.set_xlim([0, 1000])
ax.set_ylim([0, 1280])
ax.set_zlim([0, 720])

# Show the plot
plt.show()