import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ファイルパスの指定
particle_output_file = 'particle_tracking_results_potter_suruga_test_short.pkl'
centroid_output_file = 'centroid_history_results_potter_suruga_test_short.pkl'

# データの読み込み
with open(particle_output_file, 'rb') as f:
    particle_data = pickle.load(f)

with open(centroid_output_file, 'rb') as f:
    centroid_history = pickle.load(f)

# 質量の閾値
mass_threshold = 500  # この値以上の質量を持つ粒子のみをプロット

# プロット設定
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 各粒子の重心の軌跡をプロット
for pid, history in centroid_history.items():
    if particle_data[pid]['mass'] >= mass_threshold:
        times, centroids = zip(*history)
        centroids = np.array(centroids)
        
        # 時間をミリ秒に変換
        times = np.array(times) * 1e-3

        # 重心の軌跡をラインプロット
        ax.plot(times, centroids[:, 0], centroids[:, 1], label=f'Particle {pid}')

# 軸ラベルの設定
ax.set_xlabel('Time (ms)')
ax.set_ylabel('X Coordinate')
ax.set_zlabel('Y Coordinate')

# プロット表示
plt.show()