import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Particle tracking script with splitting.')
parser.add_argument('-i', '--input', required=True, help='Path to the input pickle file.')
args = parser.parse_args()

# 投影のための関数（視点ベクトルに平行に移動させてからXY平面に投影）
def project_parallel_to_view_vector(points, view_vector):
    view_vector = view_vector / np.linalg.norm(view_vector)  # 視点ベクトルを正規化
    t = -points[:, 2] / view_vector[2]  # Z軸成分を基準に平行移動量を計算
    projection = points + np.outer(t, view_vector)  # 視点ベクトル方向に平行移動
    projection[:, 2] = 0  # XY平面に投影
    return projection

# パーティクルデータをロード
file_path = args.input
with open(file_path, 'rb') as f:
    particle_data = pickle.load(f)

# 出力ファイル名を .pkl から .csv に変更
output_file = f'{file_path.split(".pkl")[0]}_.csv'

# CSVヘッダー
header = ['Particle_ID', 'Projected_Points_Size', 'Major_Axis', 'Minor_Axis',
          'withoutWarp', 'Major_Axis_woWarp', 'Minor_Axis_woWarp']

# 出力ファイルに書き込み開始
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # 各パーティクルを処理
    for particle_id, particle_info in particle_data.items():
        event_coords = np.array(particle_info['events'])
        
        # 視点ベクトルなしでXY平面に投影
        projected_points_woWarp = event_coords.copy()
        projected_points_woWarp[:, 2] = 0

        # PCAを使用して楕円フィット（XY平面）
        pca_woWarp = PCA(n_components=2)
        pca_woWarp.fit(projected_points_woWarp[:, :2])
        width_woWarp, height_woWarp = 4 * np.sqrt(pca_woWarp.explained_variance_)
        semi_major_woWarp = width_woWarp / 2
        semi_minor_woWarp = height_woWarp / 2
        area_woWarp = np.pi * semi_major_woWarp * semi_minor_woWarp

        # CSVに書き込み
        writer.writerow([particle_id, '-', '-', '-', area_woWarp, semi_major_woWarp, semi_minor_woWarp])

print(f"Results saved to {output_file}")
