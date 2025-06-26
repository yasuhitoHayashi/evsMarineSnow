import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import argparse
import matplotlib.pyplot as plt  # pltからカラーマップを取得


# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Particle tracking script with splitting.')
parser.add_argument('-i', '--input', required=True, help='Path to the input pickle file.')
args = parser.parse_args()


# ---- New general warp utilities ----
def estimate_velocity(centroid_history):
    """Estimate average (vx, vy) from centroid history."""
    history = np.array(centroid_history)
    if history.shape[0] < 2:
        return np.array([0.0, 0.0])
    dt = np.diff(history[:, 0])
    dx = np.diff(history[:, 1])
    dy = np.diff(history[:, 2])
    valid = dt != 0
    if not np.any(valid):
        return np.array([0.0, 0.0])
    vx = np.mean(dx[valid] / dt[valid])
    vy = np.mean(dy[valid] / dt[valid])
    return np.array([vx, vy])


def warp_events(events, velocity, t_ref):
    """Warp events (x, y, t) to reference time using estimated velocity."""
    warped = events.copy().astype(float)
    dt = t_ref - warped[:, 2]
    warped[:, 0] += velocity[0] * dt
    warped[:, 1] += velocity[1] * dt
    warped[:, 2] = t_ref
    return warped

# 時間ビンのサイズを設定 (マイクロ秒単位)
time_bin_size = 1000  # 1ミリ秒

# スムージング用のウィンドウサイズを設定 (例: 10ミリ秒)
smoothing_window_size = 10  # ビンの数で設定


# パーティクルデータをロード
file_path = args.input  # コマンドライン引数で指定されたファイルパスを使用

with open(file_path, 'rb') as f:
    particle_data = pickle.load(f)

# 1280x720の黒い背景画像を作成
# 高解像度の1280x720の黒い背景画像を作成 (dpi=300で解像度向上)
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=300)  # dpiを300に設定
ax.set_facecolor('black')

# 軸の範囲を設定 (1280x720ピクセルの範囲)
ax.set_xlim(0, 1280)
ax.set_ylim(0, 720)
ax.invert_yaxis()  # 画像のY軸を反転して、通常の画像と同じように上が0、下が720になるように設定

# 出力ファイル名を .pkl ファイルから .csv に変更
output_file = f'{file_path.split(".pkl")[0]}_noWarp.csv'

# CSVヘッダー
header = ['Particle_ID', 'Projected_Points_Size', 'Major_Axis', 'Minor_Axis','Projected_Points_Size_noWarp','Major_Axis_noWarp','Minor_Axis_noWarp', 'eventCount']
cmap = plt.get_cmap('hsv', len(particle_data))  # 粒子の数に基づいてカラーマップを作成

# 出力ファイルに書き込み開始
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # ヘッダーを書き込み

    # 各パーティクルを処理
    for idx, (particle_id, particle_info) in enumerate(particle_data.items()):
        # パーティクルのイベントとセントロイド履歴を抽出
        event_coords = np.array(particle_info['events'])
        centroid_history = np.array(particle_info['centroid_history'])

        # イベント時間をミリ秒に変換
        event_times = event_coords[:, 2] * 1e-3
        
        # 時間の範囲を取得
        min_time, max_time = np.min(event_times), np.max(event_times)
        
        # 時間ビンを作成
        time_bins = np.arange(min_time, max_time + time_bin_size * 1e-3, time_bin_size * 1e-3)
        
        # 各時間ビン内のイベント数をカウント
        event_counts, _ = np.histogram(event_times, bins=time_bins)
         
        # 移動平均を使用してスムージング
        smoothed_event_counts = np.convolve(event_counts, np.ones(smoothing_window_size) / smoothing_window_size, mode='same')
        
        # スムージングされたイベント数にポリフィットを適用
        try:
            poly_coeff = np.polyfit(time_bins[:-1], smoothed_event_counts, deg=5)  # 5次の多項式
            poly_fit = np.polyval(poly_coeff, time_bins[:-1])
        except Exception as e:
            print(f"Failed to fit polynomial for particle {particle_id}: {e}")
            continue

        # 多項式フィットのピークを特定
        peak_index = np.argmax(poly_fit)
        peak_time = time_bins[peak_index]

        # 多項式ピークを中心とする10ミリ秒のウィンドウをハイライト
        window_start = peak_time - 5  # ピークの5ミリ秒前から
        window_end = peak_time + 5    # ピークの5ミリ秒後まで

        # ハイライトされたウィンドウ内のイベントを抽出
        window_mask = (event_times >= window_start) & (event_times <= window_end)
        window_events = event_coords[window_mask]

        # 前後2ミリ秒のイベントを抽出
        pre_window_mask = (event_times >= window_start - 2) & (event_times < window_start)
        post_window_mask = (event_times > window_end) & (event_times <= window_end + 2)
        
        pre_window_events = event_coords[pre_window_mask]
        post_window_events = event_coords[post_window_mask]

        if pre_window_events.shape[0] < 2 or post_window_events.shape[0] < 2:
            print(f"Not enough events in pre/post window for particle {particle_id}")
            continue

        # 速度を推定し、イベントをピーク時刻へワープ
        velocity = estimate_velocity(centroid_history)
        warped_events = warp_events(window_events, velocity, peak_time * 1000)

        # ワープ後は時間軸を無視してXY平面に投影
        projected_points_original = warped_events.copy()
        projected_points_original[:, 2] = 0

        # 視点ベクトルなしでXY平面に投影
        projected_points_noWarp = window_events.copy()
        projected_points_noWarp[:, 2] = 0

        # PCAを使用して楕円フィット（XY平面）
        pca_noWarp = PCA(n_components=2)
        pca_noWarp.fit(projected_points_noWarp[:, :2])
        width_noWarp, height_noWarp = 4 * np.sqrt(pca_noWarp.explained_variance_)
        semi_major_noWarp = width_noWarp / 2
        semi_minor_noWarp = height_noWarp / 2
        area_noWarp = np.pi * semi_major_noWarp * semi_minor_noWarp
        # パーティクルのイベントをプロット

        # PCAを使用して楕円をフィット（視点ベクトルによる投影）
        pca_projected = PCA(n_components=2)
        pca_projected.fit(projected_points_original[:, :2])
        center_projected = pca_projected.mean_
        width_projected, height_projected = 4 * np.sqrt(pca_projected.explained_variance_)
        angle_projected = np.degrees(np.arctan2(*pca_projected.components_[0][::-1]))
        # 視点ベクトル投影の楕円面積を計算
        semi_major_axis_projected = width_projected / 2
        semi_minor_axis_projected = height_projected / 2
        area_projected = np.pi * semi_major_axis_projected * semi_minor_axis_projected

        # 結果をCSVファイルに保存
        writer.writerow([particle_id, area_projected, semi_major_axis_projected, semi_minor_axis_projected,area_noWarp,semi_major_noWarp,semi_minor_noWarp,event_counts[peak_index]])

# 画像をファイルに保存
#img_output_file = f'{file_path.split(".pkl")[0]}.png'
#plt.savefig(img_output_file, dpi=300)
#plt.close()

#print(f"Results saved to {output_file} and image saved to {img_output_file}")