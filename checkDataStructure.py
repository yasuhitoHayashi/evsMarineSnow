import pickle

# Load the particle data from the pickle file
particle_output_file = 'particle_tracking_results_suruga_test_short.pkl'

with open(particle_output_file, 'rb') as f:
    particle_data = pickle.load(f)

# Check the structure of the first particle
for particle_id, particle_info in particle_data.items():
    print(f"Particle ID: {particle_id}")
    print(f"Particle Data: {particle_info}")
    break  # Just show the first particle's data