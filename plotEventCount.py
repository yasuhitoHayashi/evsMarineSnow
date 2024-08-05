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

# Define window size for smoothing (e.g., 5 milliseconds)
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
        
        # Plot the smoothed event counts over time
        plt.figure()
        plt.plot(time_bins[:-1], smoothed_event_counts, label='Smoothed Event Count', linewidth=2)
        plt.xlabel('Time (milliseconds)')
        plt.ylabel('Number of Events')
        plt.title(f'Event Count over Time for Particle {particle_id}')
        plt.grid(True)
        plt.show()