import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import progressbar
import pickle

# CSVファイルのパス
file_path = '80_balls.csv'

# CSVファイルの読み込み
try:
    data = pd.read_csv(file_path, header=None, names=['x', 'y', 'polarity', 'time'])
    print("CSVファイルの読み込みに成功しました")
except Exception as e:
    print(f"CSVファイルの読み込みに失敗しました: {e}")
    exit()

# 極性が0のデータのみを使用
data_filtered = data[data['polarity'] == 0].copy()
print(f"フィルタリング後のデータ数: {len(data_filtered)}")

# 時間を1000マイクロ秒ごとに分割
time_bins = np.arange(data_filtered['time'].min(), data_filtered['time'].max() + 1000, 1000)
data_filtered['time_bin'] = pd.cut(data_filtered['time'], bins=time_bins, labels=False, include_lowest=True)

# クラスタリング結果を保存するリスト
results = []

# クラスタリングの実施
for time_bin in progressbar.progressbar(data_filtered['time_bin'].unique(), prefix='クラスタリング進行中: '):
    bin_data = data_filtered[data_filtered['time_bin'] == time_bin][['x', 'y', 'time']].values
    
    if len(bin_data) == 0:
        continue
    
    # DBSCANによるクラスタリング
    db = DBSCAN(eps=20, min_samples=3).fit(bin_data)
    labels = db.labels_

    # クラスタごとの重心を計算
    unique_labels = set(labels)
    for cluster_idx in unique_labels:
        if cluster_idx == -1:
            continue
        cluster_points = bin_data[labels == cluster_idx]
        if len(cluster_points) >= 3:
            centroid = cluster_points.mean(axis=0)
            results.append([time_bin, cluster_idx, centroid[0], centroid[1], centroid[2], cluster_points])

# 結果をデータフレームに変換
results_df = pd.DataFrame(results, columns=['time_bin', 'cluster_id', 'centroid_x', 'centroid_y', 'centroid_time', 'cluster_points'])

# クラスタの確認
print("クラスタリング結果の確認：")
print(results_df.head())

# 粒子ごとの座標変化を追跡するための辞書
particles = {}
particles_points = {}
max_distance = 50  # 粒子が次の位置に移動できる最大距離
max_time_without_update = 3  # 粒子がトラッキングされずに削除されるまでの最大時間

# 現在トラッキング中の粒子と、最近の更新時間を追跡する辞書
active_particles = {}
last_update_time = {}

# クラスタのリンク
for time_bin in progressbar.progressbar(results_df['time_bin'].unique(), prefix='粒子リンク進行中: '):
    current_clusters = results_df[results_df['time_bin'] == time_bin]
    
    for i, row in current_clusters.iterrows():
        current_position = np.array([row['centroid_x'], row['centroid_y'], row['centroid_time']])
        assigned = False
        
        # 現在トラッキング中の粒子をチェック
        for particle_id, last_position in active_particles.items():
            if np.linalg.norm(last_position[:2] - current_position[:2]) < max_distance:
                # 同一粒子としてリンク
                particles[particle_id].append((row['centroid_time'], row['centroid_x'], row['centroid_y']))
                particles_points[particle_id].append(row['cluster_points'])
                active_particles[particle_id] = current_position
                last_update_time[particle_id] = time_bin
                assigned = True
                break
        
        if not assigned:
            # 新しい粒子として登録
            new_particle_id = max(particles.keys(), default=-1) + 1
            particles[new_particle_id] = [(row['centroid_time'], row['centroid_x'], row['centroid_y'])]
            particles_points[new_particle_id] = [row['cluster_points']]
            active_particles[new_particle_id] = current_position
            last_update_time[new_particle_id] = time_bin
            print(f"New particle {new_particle_id} created at time {row['centroid_time']}")
    
    # トラッキングから外れる粒子の削除
    for particle_id in list(active_particles.keys()):
        if time_bin - last_update_time[particle_id] > max_time_without_update:
            del active_particles[particle_id]
            del last_update_time[particle_id]

# リンク結果の確認
print("粒子リンク結果の確認：")
for particle_id, positions in particles.items():
    print(f"Particle {particle_id}: {positions}")

# 距離が大きくなりすぎた場合の粒子の追跡を止める処理（フィルタリング）
filtered_particles = {}
filtered_particles_points = {}
for particle_id, positions in particles.items():
    if len(positions) > 1:
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
        particle_data.append([particle_id, pos[0], pos[1], pos[2]])

particle_df = pd.DataFrame(particle_data, columns=['particle_id', 'time', 'x', 'y'])
particle_df.to_csv(output_file, index=False)
print(f"粒子のIDと座標データを {output_file} に保存しました")

# クラスタの点群データを保存
pkl_output_file = 'particle_clusters.pkl'
with open(pkl_output_file, 'wb') as f:
    pickle.dump(filtered_particles_points, f)
print(f"クラスタの点群データを {pkl_output_file} に保存しました")