import pandas as pd
import numpy as np
import pickle
from collections import deque
from tqdm import tqdm
from numba import njit
import argparse
import os
import cProfile
import pstats
from expelliarmus import Wizard
from pathlib import Path

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Particle tracking script.')
parser.add_argument('-i', '--input', required=True, help='Path to the input raw file.')
args = parser.parse_args()

# RAW file path
file_path = Path(args.input)

# Initialize the Wizard class and read the data
wizard = Wizard(encoding="evt3")
wizard.set_file(file_path)
data = wizard.read()

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Filter data to only include events with polarity 1
data_filtered = df[df['p'] == 1].copy()
print(f"Number of data points after filtering: {len(data_filtered)}")

# Class to manage particles
class Particle:
    def __init__(self, particle_id, x, y, time):
        self.particle_id = particle_id
        self.events = deque([(x, y, time)])  # Store all events with coordinates and time
        self.recent_events = deque([(x, y, time)])  # Separate deque for recent events
        self.mass = 1
        self.centroid = np.array([x, y], dtype=np.float64)
        self.centroid_history = [(time, self.centroid.copy())]
        self.centroid_time_window = 1000  # Time window in microseconds for centroid calculation

    def add_event(self, x, y, time):
        self.events.append((x, y, time))
        self.recent_events.append((x, y, time))  # Add to recent events
        self.mass += 1

        # Remove old events from the recent events deque
        while self.recent_events and self.recent_events[0][2] < time - self.centroid_time_window:
            self.recent_events.popleft()
        
        # Update centroid using recent events
        if self.recent_events:
            coords = np.array([event[:2] for event in self.recent_events], dtype=np.float64)
            self.centroid = coords.mean(axis=0)
            self.centroid_history.append((time, self.centroid.copy()))

    def is_active(self, current_time, m_threshold):
        # A particle is inactive if no events in 5 ms and mass < M
        if self.events and self.events[-1][2] < current_time - 5000:
            return self.mass > m_threshold
        return True

# JIT compile distance calculation
@njit
def calculate_distance_sq(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

# Particle tracking
def track_particles(data, spatial_radius=6, time_window=50000, m_threshold=500):
    particles = {}
    particle_id_counter = 0
    active_particles = []

    spatial_radius_squared = spatial_radius ** 2  # Precompute for efficiency

    for event in tqdm(data.itertuples(index=False), total=len(data), desc="Processing events"):
        x, y, time = event.x, event.y, event.t

        # Check for nearby active particles
        nearby_particle_id = None
        for particle in active_particles:
            dist_sq = calculate_distance_sq(particle.centroid[0], particle.centroid[1], x, y)
            if dist_sq <= spatial_radius_squared and particle.events[-1][2] >= time - time_window:
                nearby_particle_id = particle.particle_id
                break

        if nearby_particle_id is not None:
            # Update existing particle
            particles[nearby_particle_id].add_event(x, y, time)
        else:
            # Create a new particle
            particle_id_counter += 1
            new_particle = Particle(particle_id_counter, x, y, time)
            particles[particle_id_counter] = new_particle
            active_particles.append(new_particle)

        # Clean up inactive particles
        active_particles = [p for p in active_particles if p.is_active(time, m_threshold)]

    return particles

# Profile the script
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run particle tracking
    particles = track_particles(data_filtered)
    print(f"Number of tracked particles: {len(particles)}")

    # Output results as a dictionary
    particle_data = {pid: {'centroid': p.centroid.tolist(), 'mass': p.mass, 'events': list(p.events)} for pid, p in particles.items()}

    # Save centroid history in a dictionary
    centroid_history = {pid: p.centroid_history for pid, p in particles.items()}

    # Get the base name of the input file without the extension
    base_file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Save clustering results to a Pickle file
    particle_output_file = f'particle_tracking_results_potter_{base_file_name}.pkl'
    centroid_output_file = f'centroid_history_results_potter_{base_file_name}.pkl'

    with open(particle_output_file, 'wb') as f:
        pickle.dump(particle_data, f)
    print(f"Particle tracking results saved to {particle_output_file}")

    with open(centroid_output_file, 'wb') as f:
        pickle.dump(centroid_history, f)
    print(f"Centroid history saved to {centroid_output_file}")

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(10)