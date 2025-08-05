import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# グリッドラインやその他の詳細な設定
plt.rcParams.update({
    'lines.linewidth': 2,
    'grid.linestyle': '--',
    'axes.grid': True,  # グリッドを表示
    'axes.facecolor': 'white',  # 背景を白に
    'axes.edgecolor': 'gray',  # 軸の色を灰色に
    'font.size': 11,  # 基本フォントサイズ
    'axes.labelsize': 14,  # 軸ラベルのフォントサイズ
    'xtick.labelsize': 11,  # x軸の目盛りラベルサイズ
    'ytick.labelsize': 11,  # y軸の目盛りラベルサイズ
    'figure.figsize': (12, 8),  # 図のサイズ
})

parser = argparse.ArgumentParser(description='Particle tracking script.')
parser.add_argument('-i', '--input', required=True, help='Path to the input pickle file or directory.')
args = parser.parse_args()

# プロット用にイベントを抽出
sampling_ratio = 0.01

# イベントをプロットするか否かのフラグ  
plot_events = True  # Falseにすることで、重心のみをプロット

def process_pickle_file(particle_output_file):
    output_directory = os.path.dirname(particle_output_file)
    
    # 出力フォルダ (fft_results) を作成
    fft_results_dir = os.path.join(output_directory, 'plotTrajectory_results')
    os.makedirs(fft_results_dir, exist_ok=True)  # 既に存在していてもエラーを出さない
    
    # ファイル名の処理: 前から3つ目の "_" より後の部分を使い、".pkl"を除去
    base_filename = os.path.basename(particle_output_file)
    base_filename = '_'.join(base_filename.split('_')[3:]).replace('.pkl', '')

    with open(particle_output_file, 'rb') as f:
        particle_data = pickle.load(f)

    # Create a 3D plot using Matplotlib
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 粒子の軌跡とイベントをプロット
    for particle_id, particle_info in particle_data.items():
        if 'centroid_history' in particle_info:
            centroid_history = np.array(particle_info['centroid_history'])  # (time, centroid_x, centroid_y)

            # 重心の履歴、1以上あるかどうか
            if len(centroid_history) > 1:
                # 重心の軌跡をプロット
                ax.plot(centroid_history[:, 0] * 1e-3, centroid_history[:, 1], centroid_history[:, 2],label=f'Particle {particle_id} Centroid Trajectory')

        # フラグがTrueの場合、イベントをプロット
        if plot_events:
            events = particle_info['events']
            event_coords = np.array(events)
            event_times = event_coords[:, 2] * 1e-3  # Convert event times to milliseconds

            # 描画のためのダウンサンプリング
            num_events = len(events)
            if sampling_ratio < 1.0:
                sample_size = int(num_events * sampling_ratio)
                if sample_size > 0:
                    sampled_indices = np.random.choice(num_events, sample_size, replace=False)
                    sampled_events = event_coords[sampled_indices]
                    sampled_event_times = event_times[sampled_indices]
                    
                    # Scatter plot
                    ax.scatter(sampled_event_times, sampled_events[:, 0], sampled_events[:, 1], alpha=0.3, marker='.')
            else:
                # ダウンサンプリングしない場合
                ax.scatter(event_times, event_coords[:, 0], event_coords[:, 1], alpha=0.3, marker='.')

    ax.set_xlabel('Time (milliseconds)')
    ax.set_ylabel('X Coordinate')
    ax.set_zlabel('Y Coordinate')

    ax.set_ylim([0, 1280])
    ax.set_zlim([720, 0])
    ax.view_init(elev=4, azim=-40)  # 必要に応じて数値を変更

    #output_image_file = os.path.join(fft_results_dir, f'{base_filename}_trajectory.png')
    plt.tight_layout()  # レイアウトを調整
    plt.show()
    #plt.savefig('plotTrajectory.png')
    #plt.close()

input_path = args.input

if os.path.isdir(input_path):
    # ディレクトリ内のすべてのファイルを処理
    for filename in os.listdir(input_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(input_path, filename)
            print(f"Processing file: {file_path}")
            process_pickle_file(file_path)
elif os.path.isfile(input_path) and input_path.endswith('.pkl'):
    # 単一ファイルを処理
    print(f"Processing file: {input_path}")
    process_pickle_file(input_path)
else:
    print(f"Error: {input_path} is not a valid pickle file or directory.")