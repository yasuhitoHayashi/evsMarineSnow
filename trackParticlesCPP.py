import pandas as pd
import numpy as np
import ctypes
import pickle

# Load the shared library
lib = ctypes.CDLL('./particle_tracking.so')

# Define the argument and return types of the C++ function
lib.track_particles.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_int),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
]

# CSV file path
file_path = 'suruga_test_short.csv'  # Replace with your actual file path

# Read CSV file
data = pd.read_csv(file_path, header=None, names=['x', 'y', 'polarity', 'time'])

# Use only data with polarity 1
data_filtered = data[data['polarity'] == 1].copy()
print(f"Number of data points after filtering: {len(data_filtered)}")

# Convert data to numpy array
events_array = data_filtered[['x', 'y', 'time']].to_numpy(dtype=np.float64).flatten()

# Prepare output arrays
num_events = len(data_filtered)
output_array = np.zeros(num_events * 5, dtype=np.float64)  # 1D array for particle info

# Assume an upper bound for the number of centroids and events
max_centroids = num_events * 5
max_event_coords = num_events * 4

centroid_array = np.zeros(max_centroids * 3, dtype=np.float64)
event_array = np.zeros(max_event_coords * 4, dtype=np.float64)

particle_count = ctypes.c_int()

# Run the C++ optimized particle tracking
lib.track_particles(events_array, num_events, 6, 2000, 10000, ctypes.byref(particle_count), output_array, centroid_array, event_array)

print(f"Number of tracked particles: {particle_count.value}")

# Process the output
particles = {}
centroid_history = {}
event_coordinates = {}

for i in range(particle_count.value):
    pid = int(output_array[i * 5])
    centroid = output_array[i * 5 + 1:i * 5 + 3].tolist()
    mass = int(output_array[i * 5 + 3])
    event_count = int(output_array[i * 5 + 4])

    particles[pid] = {
        'centroid': centroid,
        'mass': mass,
        'events': []  # Placeholder for events
    }

# Parse centroid history
i = 0
while i < max_centroids * 3 and centroid_array[i] != 0:
    pid = int(centroid_array[i])
    time = centroid_array[i + 1]
    centroid_x = centroid_array[i + 2]

    if pid not in centroid_history:
        centroid_history[pid] = []

    centroid_history[pid].append((time, centroid_x))
    i += 3

# Parse event coordinates
i = 0
while i < max_event_coords * 4 and event_array[i] != 0:
    pid = int(event_array[i])
    x = event_array[i + 1]
    y = event_array[i + 2]
    time = event_array[i + 3]

    if pid not in event_coordinates:
        event_coordinates[pid] = []

    event_coordinates[pid].append((x, y, time))
    i += 4

# Integrate parsed events into particles
for pid in particles:
    particles[pid]['events'] = event_coordinates.get(pid, [])

# Save the results in the desired format
particle_output_file = 'particle_tracking_results.pkl'
centroid_output_file = 'centroid_history_results.pkl'

with open(particle_output_file, 'wb') as f:
    pickle.dump(particles, f)
print(f"Particle tracking results saved to {particle_output_file}")

with open(centroid_output_file, 'wb') as f:
    pickle.dump(centroid_history, f)
print(f"Centroid history saved to {centroid_output_file}")
