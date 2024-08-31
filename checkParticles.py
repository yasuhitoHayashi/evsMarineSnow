import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

# 投影のための関数（視点ベクトルに平行に移動させてからXY平面に投影）
def project_parallel_to_view_vector(points, view_vector):
    # 視点ベクトルを正規化
    view_vector = view_vector / np.linalg.norm(view_vector)
    
    # 各点を視点ベクトルに平行に移動させるための係数を計算
    t = -points[:, 2] / view_vector[2]
    
    # 視点ベクトル方向に平行移動
    projection = points + np.outer(t, view_vector)
    
    # Z成分を0にする（XY平面に投影）
    projection[:, 2] = 0
    
    return projection

# パーティクルデータをロード
particle_output_file = 'particle_tracking_results.pkl'

with open(particle_output_file, 'rb') as f:
    particle_data = pickle.load(f)

# フィルタリングのための質量の閾値を設定
mass_threshold = 1000  # 必要に応じて調整

# 時間ビンのサイズを設定 (マイクロ秒単位)
time_bin_size = 1000  # 1ミリ秒

# スムージング用のウィンドウサイズを設定 (例: 10ミリ秒)
smoothing_window_size = 10  # ビンの数で設定

# 各パーティクルを処理
for particle_id, particle_info in particle_data.items():
    # パーティクルの質量が閾値以上かを確認
    if particle_info['mass'] >= mass_threshold:
        # パーティクルのイベントを抽出
        event_coords = np.array(particle_info['events'])
        
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

        # セントロイドを計算
        pre_centroid = pre_window_events[:, :3].mean(axis=0)
        post_centroid = post_window_events[:, :3].mean(axis=0)

        # 前後のセントロイド間の方向ベクトルを計算
        direction_vector = post_centroid - pre_centroid
        direction_vector /= np.linalg.norm(direction_vector)  # 正規化

        # 投影処理
        projected_points_original = project_parallel_to_view_vector(window_events, direction_vector)

        # 元のXY平面にそのまま投影（Z成分を無視）
        projected_points_xy = window_events[:, :2]

        # 可視化
        fig = plt.figure(figsize=(12, 9))

        # イベント数の時間変化とポリフィットのプロット
        ax1 = fig.add_subplot(221)
        ax1.plot(time_bins[:-1], smoothed_event_counts, label='Smoothed Event Count', linewidth=2)
        ax1.plot(time_bins[:-1], poly_fit, label='Polynomial Fit', linestyle='--')
        ax1.axvspan(window_start, window_end, color='red', alpha=0.3, label='Peak 10ms Window')
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Event Count')
        ax1.set_title(f'Event Count Over Time for Particle {particle_id}')
        ax1.legend()
        ax1.grid(True)

        # 3次元散布図のプロット
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.scatter(window_events[:, 0], window_events[:, 1], window_events[:, 2], color='b', label='Original Events')
        ax2.set_title('3D Event Distribution in Peak Window')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Time (ms)')

        # 投影点のプロットを追加
        ax2.scatter(projected_points_original[:, 0], projected_points_original[:, 1], projected_points_original[:, 2], color='r', label='Projected Events')
        ax2.legend()

        # 元のXY平面への投影プロット
        ax3 = fig.add_subplot(223)
        ax3.scatter(projected_points_xy[:, 0], projected_points_xy[:, 1], color='g', label='Original XY Projection')

        # PCAを使用して楕円をフィット（元のXY投影）
        pca_xy = PCA(n_components=2)
        pca_xy.fit(projected_points_xy)
        center_xy = pca_xy.mean_
        width_xy, height_xy = 4 * np.sqrt(pca_xy.explained_variance_)
        angle_xy = np.degrees(np.arctan2(*pca_xy.components_[0][::-1]))

        # 楕円の描画（元のXY投影）
        ellipse_xy = Ellipse(xy=center_xy, width=width_xy, height=height_xy, angle=angle_xy, edgecolor='blue', facecolor='none', linestyle='--', linewidth=2)
        ax3.add_patch(ellipse_xy)
        ax3.set_xlim([projected_points_xy[:, 0].min() - 10, projected_points_xy[:, 0].max() + 10])
        ax3.set_ylim([projected_points_xy[:, 1].min() - 10, projected_points_xy[:, 1].max() + 10])
        ax3.set_title('Original XY Projection')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.axis('equal')
        ax3.legend()

        # 元のXY投影の楕円面積を計算
        semi_major_axis_xy = width_xy / 2
        semi_minor_axis_xy = height_xy / 2
        area_xy = np.pi * semi_major_axis_xy * semi_minor_axis_xy

        # 面積をプロットに表示
        ax3.text(0.05, 0.95, f'Area: {area_xy:.2f} units²', transform=ax3.transAxes, fontsize=10, verticalalignment='top')

        # 視点ベクトルに基づく投影の2次元プロット
        ax4 = fig.add_subplot(224)
        ax4.scatter(projected_points_original[:, 0], projected_points_original[:, 1], color='r', label='Projected with View Vector')

        # PCAを使用して楕円をフィット（視点ベクトルによる投影）
        pca_projected = PCA(n_components=2)
        pca_projected.fit(projected_points_original[:, :2])
        center_projected = pca_projected.mean_
        width_projected, height_projected = 4 * np.sqrt(pca_projected.explained_variance_)
        angle_projected = np.degrees(np.arctan2(*pca_projected.components_[0][::-1]))

        # 楕円の描画（視点ベクトルによる投影）
        ellipse_projected = Ellipse(xy=center_projected, width=width_projected, height=height_projected, angle=angle_projected, edgecolor='blue', facecolor='none', linestyle='--', linewidth=2)
        ax4.add_patch(ellipse_projected)
        ax4.set_xlim([projected_points_original[:, 0].min() - 10, projected_points_original[:, 0].max() + 10])
        ax4.set_ylim([projected_points_original[:, 1].min() - 10, projected_points_original[:, 1].max() + 10])
        ax4.set_title('Projected XY Plane using View Vector')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.axis('equal')
        ax4.legend()

        # 視点ベクトル投影の楕円面積を計算
        semi_major_axis_projected = width_projected / 2
        semi_minor_axis_projected = height_projected / 2
        area_projected = np.pi * semi_major_axis_projected * semi_minor_axis_projected

        # 面積をプロットに表示
        ax4.text(0.05, 0.95, f'Area: {area_projected:.2f} units²', transform=ax4.transAxes, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()