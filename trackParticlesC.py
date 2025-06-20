import pandas as pd
import argparse
import pickle
from particle_tracking import track_particles_cpp  # C++ モジュールのインポート

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Particle tracking script with splitting.')
parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file.')
args = parser.parse_args()

# CSV file path
file_path = args.input

# Read CSV file
data = pd.read_csv(file_path, header=None, names=['x', 'y', 'polarity', 'time'])

# Use only data with polarity 1
data_filtered = data[data['polarity'] == 1].copy()

# スタートから最初の5秒（5000000マイクロ秒）以内のデータだけをフィルタリング
start_time = data_filtered['time'].min()
time_limit = start_time + 1000000  # 5秒 = 5000000マイクロ秒
data_filtered = data_filtered[data_filtered['time'] <= time_limit]

# Convert to list of tuples for compatibility with C++ function
# Ensure that time is a float (this will create a list of (x, y, float time) tuples)
data_list = [tuple(row) for row in data_filtered[['x', 'y', 'time']].itertuples(index=False, name=None)]

print(f"Number of data points after filtering: {len(data_filtered)}")

# Set the parameters for the particle tracking function
m_sigma_x = 9.0  # トップハット判定の空間半径
m_sigma_t = 10000.0  # トップハット判定の時間半径
gaussian_threshold = 0.0  # トップハットでは未使用
m_threshold = 100  # 質量のしきい値, 100くらい

# Try to run the C++ particle tracking function and catch any exceptions
try:
    # C++の関数を呼び出す
    particles = track_particles_cpp(data_list, m_sigma_x, m_sigma_t, gaussian_threshold, m_threshold)

    # Prepare the output data for saving
    particle_output = {}
    for p in particles:
        particle_output[p.particle_id] = {
            'centroid_history': p.centroid_history,  # Centroid coordinates (x, y)
            'events': p.events       # List of events [(x, y, time), ...]
        }

    # Save the output data to a pickle file
    output_file = f'{file_path.split(".csv")[0]}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(particle_output, f)

    print(f"Particle tracking results saved to {output_file}")

except Exception as e:
    print("An error occurred during the particle tracking process.")
    print(f"Error message: {e}")