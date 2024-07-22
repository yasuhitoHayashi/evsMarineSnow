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
data_filtered = data[(data['polarity'] == 0) & (data['y'] >= 100) & (data['y'] <= 400)].copy()
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
    db = DBSCAN(eps=15, min_samples=10).fit(bin_data)
    labels = db.labels_

    # クラスタごとの重心を計算
    unique_labels = set(labels)
    for cluster_idx in unique_labels:
        if cluster_idx == -1:
            continue
        cluster_points = bin_data[labels == cluster_idx]
        if len(cluster_points) >= 5:
            centroid = cluster_points.mean(axis=0)
            results.append([time_bin, cluster_idx, centroid[0], centroid[1], centroid[2], cluster_points])

# 結果をデータフレームに変換
results_df = pd.DataFrame(results, columns=['time_bin', 'cluster_id', 'centroid_x', 'centroid_y', 'centroid_time', 'cluster_points'])

# クラスタリング結果をPickleファイルに保存
pkl_output_file = 'clustering_results.pkl'
with open(pkl_output_file, 'wb') as f:
    pickle.dump(results_df, f)
print(f"クラスタリング結果を {pkl_output_file} に保存しました")


1'59-2'01