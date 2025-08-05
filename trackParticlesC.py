import pandas as pd
import argparse
import pickle
from particle_tracking import track_particles_cpp  # C++ モジュールのインポート

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Particle tracking script.')
parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file.')
# しきい値を外部指定したい場合は以下をアンコメント
# parser.add_argument('--threshold', type=float, default=0.8, help='Gaussian threshold (default: 0.8)')
args = parser.parse_args()

# CSV file path
file_path = args.input

# Read CSV file
data = pd.read_csv(file_path, header=None, names=['x', 'y', 'polarity', 'time'])

# Use only data with polarity 1
data_filtered = data[data['polarity'] == 1].copy()

start_time = data_filtered['time'].min()
time_limit = start_time + 5000000  # 500ms
data_filtered = data_filtered[data_filtered['time'] <= time_limit]

# new: (x, y, polarity, time)
data_list = [
    (int(row.x), int(row.y), 1, float(row.time))
    for row in data_filtered.itertuples(index=False)
]

print(f"Number of data points after filtering: {len(data_filtered)}")

# トップハット判定パラメータ（実際には Gaussian 判定に戻したので、ここは merge 用半径）
sigma_x = 6       # 空間半径（0.8閾値相当でスケーリング）
sigma_t = 10000 # 時間半径（µs）
m_threshold = 10            # 質量のしきい値
gaussian_threshold = 0.8    # Gaussian 閾値

try:
    # C++ の関数を呼び出し（gaussian_threshold を追加）
    particles = track_particles_cpp(
        data_list,
        sigma_x,
        sigma_t,
        gaussian_threshold,
        m_threshold
    )

    # 出力用に整形
    particle_output = {
        p.particle_id: {
            'centroid_history': p.centroid_history,
            'events': p.events
        }
        for p in particles
    }

    # pickle で保存
    output_file = file_path.rsplit('.csv', 1)[0] + '.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(particle_output, f)

    print(f"Particle tracking results saved to {output_file}")

except Exception as e:
    print("An error occurred during the particle tracking process.")
    print(f"Error message: {e}")