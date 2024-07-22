import pandas as pd
import numpy as np
import progressbar
import pickle
from pykalman import KalmanFilter

# クラスタリング結果のPickleファイルを読み込み
pkl_input_file = 'clustering_results.pkl'
with open(pkl_input_file, 'rb') as f:
    results_df = pickle.load(f)
print(f"クラスタリング結果を {pkl_input_file} から読み込みました")

# カルマンフィルタの初期化
def initialize_kalman_filter(initial_position):
    transition_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    observation_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf = KalmanFilter(transition_matrices=transition_matrix, observation_matrices=observation_matrix)
    initial_state_mean = [initial_position[0], initial_position[1], 0, 0]
    initial_state_covariance = np.eye(4) * 100
    return kf, initial_state_mean, initial_state_covariance

# 粒子ごとの座標変化を追跡するための辞書
particles = {}
particles_points = {}
kalman_filters = {}
max_distance = 10  # 粒子が次の位置に移動できる最大距離
max_time_without_update = 5  # 粒子がトラッキングされずに削除されるまでの最大時間

# 現在トラッキング中の粒子と、最近の更新時間を追跡する辞書
active_particles = {}
last_update_time = {}

# クラスタのリンク
for time_bin in progressbar.progressbar(results_df['time_bin'].unique(), prefix='粒子リンク進行中: '):
    current_clusters = results_df[results_df['time_bin'] == time_bin]
    
    for i, row in current_clusters.iterrows():
        current_position = np.array([row['centroid_x'], row['centroid_y']])
        assigned = False
        
        # 現在トラッキング中の粒子をチェック
        for particle_id, (kf, state_mean, state_covariance) in kalman_filters.items():
            # カルマンフィルタで予測
            predicted_state_mean, predicted_state_covariance = kf.filter_update(
                state_mean, state_covariance
            )
            predicted_position = predicted_state_mean[:2]
            
            if np.linalg.norm(predicted_position - current_position) < max_distance:
                # 同一粒子としてリンク
                particles[particle_id].append((row['centroid_time'], row['centroid_x'], row['centroid_y'], row['cluster_id']))
                particles_points[particle_id].append(row['cluster_points'])
                # カルマンフィルタを更新
                state_mean, state_covariance = kf.filter_update(
                    predicted_state_mean, predicted_state_covariance, current_position
                )
                kalman_filters[particle_id] = (kf, state_mean, state_covariance)
                active_particles[particle_id] = current_position
                last_update_time[particle_id] = time_bin
                assigned = True
                break
        
        if not assigned:
            # 新しい粒子として登録
            new_particle_id = max(particles.keys(), default=-1) + 1
            particles[new_particle_id] = [(row['centroid_time'], row['centroid_x'], row['centroid_y'], row['cluster_id'])]
            particles_points[new_particle_id] = [row['cluster_points']]
            kf, state_mean, state_covariance = initialize_kalman_filter(current_position)
            kalman_filters[new_particle_id] = (kf, state_mean, state_covariance)
            active_particles[new_particle_id] = current_position
            last_update_time[new_particle_id] = time_bin
            print(f"New particle {new_particle_id} created at time {row['centroid_time']}")
    
    # トラッキングから外れる粒子の削除
    for particle_id in list(active_particles.keys()):
        if time_bin - last_update_time[particle_id] > max_time_without_update:
            del active_particles[particle_id]
            del last_update_time[particle_id]
            del kalman_filters[particle_id]

# リンク結果の確認
print("粒子リンク結果の確認：")
for particle_id, positions in particles.items():
    print(f"Particle {particle_id}: {positions}")

# 距離が大きくなりすぎた場合の粒子の追跡を止める処理（フィルタリング）
filtered_particles = {}
filtered_particles_points = {}
for particle_id, positions in particles.items():
    if len(positions) > 2:
        filtered_particles[particle_id] = positions
        filtered_particles_points[particle_id] = particles_points[particle_id]

# フィルタリング結果の確認
print("フィルタリング結果の確認：")
for particle_id, positions in filtered_particles.items():
    print(f"Filtered Particle {particle_id}: {positions}")

# 結果をCSVファイルに保存
output_file = 'particle_movements.csv'
particle_data = []

for particle_id, positions in filtered_particles.items():
    for pos in positions:
        particle_data.append([particle_id, pos[0], pos[1], pos[2], pos[3]])

particle_df = pd.DataFrame(particle_data, columns=['particle_id', 'time', 'x', 'y', 'cluster_id'])
particle_df.to_csv(output_file, index=False)
print(f"粒子のIDと座標データを {output_file} に保存しました")

# クラスタの点群データを保存
pkl_output_file = 'particle_clusters.pkl'
with open(pkl_output_file, 'wb') as f:
    pickle.dump(filtered_particles_points, f)
print(f"クラスタの点群データを {pkl_output_file} に保存しました")