import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the particle data to access their events
particle_output_file = 'particle_tracking_results.pkl'

with open(particle_output_file, 'rb') as f:
    particle_data = pickle.load(f)

# Define a mass threshold for filtering particles
mass_threshold = 1000  # You can adjust this threshold

# Define time bin size in microseconds (1 ms = 1000 μs)
time_bin_size = 1000

# Define window size for smoothing (e.g., 10 milliseconds)
smoothing_window_size = 10  # This should be in terms of the number of bins

# Process each particle
for particle_id, particle_info in particle_data.items():
    # Check if the particle's mass is above the threshold
    if particle_info['mass'] >= mass_threshold:
        # Extract events for the particle
        event_coords = np.array(particle_info['events'])
        
        # Convert event times to milliseconds
        event_times = event_coords[:, 2] * 1e-3
        
        # Find the range of times
        min_time, max_time = np.min(event_times), np.max(event_times)
        
        # Create time bins
        time_bins = np.arange(min_time, max_time + time_bin_size * 1e-3, time_bin_size * 1e-3)
        
        # Count events in each time bin
        event_counts, _ = np.histogram(event_times, bins=time_bins)
        
        # Apply smoothing using a moving average
        smoothed_event_counts = np.convolve(event_counts, np.ones(smoothing_window_size) / smoothing_window_size, mode='same')
        
        # Fit a polynomial to the smoothed event counts
        try:
            poly_coeff = np.polyfit(time_bins[:-1], smoothed_event_counts, deg=5)  # Polynomial of degree 5
            poly_fit = np.polyval(poly_coeff, time_bins[:-1])
        except Exception as e:
            print(f"Failed to fit a polynomial for particle {particle_id}: {e}")
            continue
        
        # Identify the peak of the polynomial fit
        peak_index = np.argmax(poly_fit)
        peak_time = time_bins[peak_index]
        
        # Highlight a 10 ms window centered around the polynomial peak
        window_start = peak_time - 5  # Start 5 ms before the peak
        window_end = peak_time + 5    # End 5 ms after the peak

        # Extract events within the highlighted window
        window_mask = (event_times >= window_start) & (event_times <= window_end)
        window_events = event_coords[window_mask]

        # Extract events for the 2 ms before and after
        pre_window_mask = (event_times >= window_start - 2) & (event_times < window_start)
        post_window_mask = (event_times > window_end) & (event_times <= window_end + 2)
        
        pre_window_events = event_coords[pre_window_mask]
        post_window_events = event_coords[post_window_mask]

        if pre_window_events.shape[0] < 2 or post_window_events.shape[0] < 2:
            print(f"Not enough events in pre/post window for particle {particle_id}")
            continue

        # Calculate centroids
        pre_centroid = pre_window_events[:, :3].mean(axis=0)
        post_centroid = post_window_events[:, :3].mean(axis=0)

        # Calculate the direction vector from pre to post centroid
        direction_vector = post_centroid - pre_centroid
        direction_vector /= np.linalg.norm(direction_vector)  # Normalize

        # Rotate the 10 ms window events to align with the direction vector
        # For simplicity, we can assume the direction is along the Z-axis
        # Compute rotation matrix to align the vector to the Z-axis
        z_vector = np.array([0, 0, 1])
        rotation_axis = np.cross(direction_vector, z_vector)
        rotation_angle = np.arccos(np.dot(direction_vector, z_vector))

        if np.linalg.norm(rotation_axis) != 0:
            # Normalize rotation axis
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Create the rotation matrix
            cos_angle = np.cos(rotation_angle)
            sin_angle = np.sin(rotation_angle)
            ux, uy, uz = rotation_axis
            rotation_matrix = np.array([
                [cos_angle + ux*ux*(1-cos_angle), ux*uy*(1-cos_angle) - uz*sin_angle, ux*uz*(1-cos_angle) + uy*sin_angle],
                [uy*ux*(1-cos_angle) + uz*sin_angle, cos_angle + uy*uy*(1-cos_angle), uy*uz*(1-cos_angle) - ux*sin_angle],
                [uz*ux*(1-cos_angle) - uy*sin_angle, uz*uy*(1-cos_angle) + ux*sin_angle, cos_angle + uz*uz*(1-cos_angle)]
            ])

            # Rotate the window events
            rotated_window_events = np.dot(window_events, rotation_matrix.T)
        else:
            # If the rotation axis is zero, no rotation is needed
            rotated_window_events = window_events

        # Project the rotated events onto the XY plane
        projected_points = rotated_window_events[:, [0, 1]]

        # Plot the smoothed event counts over time
        plt.figure(figsize=(10, 8))
        
        # First subplot: smoothed event counts and polynomial fit
        plt.subplot(3, 1, 1)
        plt.plot(time_bins[:-1], smoothed_event_counts, label='Smoothed Event Count', linewidth=2)
        plt.plot(time_bins[:-1], poly_fit, label='Polynomial Fit', linestyle='--')
        plt.axvspan(window_start, window_end, color='red', alpha=0.3, label='Peak 10 ms Window')
        plt.xlabel('Time (milliseconds)')
        plt.ylabel('Number of Events')
        plt.title(f'Event Count over Time for Particle {particle_id}')
        plt.legend()
        plt.grid(True)

        # Second subplot: 3D scatter plot of events
        ax1 = plt.subplot(3, 1, 2, projection='3d')
        ax1.scatter(window_events[:, 0], window_events[:, 1], window_events[:, 2], c='b', marker='o')
        ax1.set_title('3D Event Distribution in Peak Window')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Time (ms)')

        # Third subplot: 2D projection aligned with direction
        plt.subplot(3, 1, 3)
        plt.scatter(projected_points[:, 0], projected_points[:, 1], c='r', marker='o')
        plt.title(f'Projection Aligned with Centroid Line for Particle {particle_id}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')

        plt.tight_layout()
        plt.show()