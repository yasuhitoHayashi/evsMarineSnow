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
parser.add_argument('--event_threshold', type=int, default=100,
                    help='Minimum number of events required to plot a particle (default: 1000)')
parser.add_argument('--sampling_ratio', type=float, default=0.1,
                    help='Ratio of events to sample for plotting (default: 0.1)')
args = parser.parse_args()

# プロット用にイベントを抽出
sampling_ratio = args.sampling_ratio
event_threshold = args.event_threshold

# イベントをプロットするか否かのフラグ  
plot_events = False  # Falseにすることで、重心のみをプロット

def process_pickle_file(pkl_file, event_threshold):
    output_dir = os.path.dirname(pkl_file)
    results_dir = os.path.join(output_dir, 'plotTrajectory_results')
    os.makedirs(results_dir, exist_ok=True)

    base_name = os.path.basename(pkl_file)
    base_name = '_'.join(base_name.split('_')[3:]).replace('.pkl', '')

    with open(pkl_file, 'rb') as f:
        particle_data = pickle.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for pid, info in particle_data.items():
        events = info.get('events', [])
        if len(events) < event_threshold:
            continue  # Skip particles with fewer events than threshold

        centroids = np.array(info['centroid_history'])  # [n, 3]
        if centroids.shape[0] > 1:
            times_ms = centroids[:, 0] * 1e-3
            xs = centroids[:, 1]
            ys = centroids[:, 2]
            ax.plot(times_ms, xs, ys, label=f'Particle {pid}')

        if plot_events:
            ev = np.array(events)
            ev_times = ev[:, 3] * 1e-3
            ax.scatter(ev_times, ev[:, 0], ev[:, 1], alpha=0.3, marker='.')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('X Coordinate')
    ax.set_zlabel('Y Coordinate')
    ax.set_ylim(0, 1280)
    ax.set_zlim(0, 720)
    ax.view_init(elev=4, azim=-40)
    plt.tight_layout()
    plt.show()


input_path = args.input
if os.path.isdir(input_path):
    for fname in os.listdir(input_path):
        if fname.endswith('.pkl'):
            print(f'Processing {fname}')
            process_pickle_file(os.path.join(input_path, fname), args.event_threshold)
elif os.path.isfile(input_path) and input_path.endswith('.pkl'):
    print(f'Processing {input_path}')
    process_pickle_file(input_path, args.event_threshold)
else:
    print(f'Error: {input_path} is not a pickle file or directory.')
