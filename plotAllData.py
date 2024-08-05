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

# Set axis labels
ax.set_ylabel('X Coordinate')
ax.set_zlabel('Y Coordinate')
ax.set_xlabel('Time (milliseconds)')

# Set axis limits to maintain aspect ratio
ax.set_xlim([0, 1000])
ax.set_ylim([0, 1280])
ax.set_zlim([0, 720])

plt.show()