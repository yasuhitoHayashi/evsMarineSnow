import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import minmax_scale
import progressbar
import pickle
import concurrent.futures
import time
import matplotlib.pyplot as plt

# CSVファイルのパス
file_path = 'suruga_test_short.csv'
#file_path = '195_falling_particles.csv'

# CSVファイルの読み込み
data = pd.read_csv(file_path, header=None, names=['x', 'y', 'polarity', 'time'])
data['time'] = data['time']* 1e-3
data  = data[(data['time'] >0) & (data['time'] < 1000)]

dataP = data[data['polarity'] == 1]
dataN = data[data['polarity'] == 0]

# 3次元プロットの作成
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataP['time'], dataP['x'], dataP['y'], s=2, c='blue', edgecolors='none', marker='.')
#ax.scatter(dataN['time'], dataN['x'], dataN['y'], s=2, c='red', edgecolors='none', marker='.')

# 上からのビュー設定
#ax.view_init(elev=90, azim=0)

# プロットの表示
#plt.show()
# アニメーションの更新関数
#def update(frame):
#    ax.view_init(elev=10, azim=frame)
#    return ax,

# アニメーションの作成
#ani = FuncAnimation(fig, update, frames=range(0, 360), interval=10, blit=False, repeat=False)

# 動画の保存
#ani.save('particle_rotation.mp4', writer='ffmpeg', dpi=100)

# プロットの表示
plt.show()