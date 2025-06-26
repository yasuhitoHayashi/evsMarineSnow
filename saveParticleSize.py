import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import csv
import argparse
import warnings


# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='Particle tracking script with splitting.')
parser.add_argument('-i', '--input', required=True, help='Path to the input pickle file.')

# このスクリプトをモジュールとしてインポートした際に実行されるのを防ぐため、
# 引数の解析とメイン処理を関数にまとめる


# 投影のための関数（視点ベクトルに平行に移動させてからXY平面に投影）
def project_parallel_to_view_vector(points, view_vector):
    """Project events onto the XY plane along a view vector.

    If the view vector is ill-defined (zero magnitude or nearly zero z-component),
    a warning is issued and the points are returned with their z values set to 0
    without additional warping.
    """
    norm = np.linalg.norm(view_vector)
    if norm == 0 or abs(view_vector[2]) < 1e-8:
        warnings.warn(
            "View vector magnitude is zero or z component too small; skipping projection",
            UserWarning,
        )
        projection = points.copy()
        projection[:, 2] = 0
        return projection

    view_vector = view_vector / norm

    # 各点を視点ベクトルに平行に移動させるための係数を計算
    t = -points[:, 2] / view_vector[2]

    # 視点ベクトル方向に平行移動
    projection = points + np.outer(t, view_vector)

    # Z成分を0にする（XY平面に投影）
    projection[:, 2] = 0

    return projection

# 時間ビンのサイズを設定 (マイクロ秒単位)
time_bin_size = 1000  # 1ミリ秒

# スムージング用のウィンドウサイズを設定 (例: 10ミリ秒)
smoothing_window_size = 10  # ビンの数で設定


def process_file(file_path: str) -> None:
    """Process a single pickle file and save particle size information."""

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

        # セントロイドを計算
        pre_centroid = pre_window_events[:, :3].mean(axis=0)
        post_centroid = post_window_events[:, :3].mean(axis=0)

        # 前後のセントロイド間の方向ベクトルを計算
        direction_vector = post_centroid - pre_centroid
        direction_vector /= np.linalg.norm(direction_vector)  # 正規化

        # 投影処理
        projected_points_original = project_parallel_to_view_vector(window_events, direction_vector)

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
            width_projected, height_projected = 4 * np.sqrt(
                pca_projected.explained_variance_
            )
            angle_projected = np.degrees(
                np.arctan2(*pca_projected.components_[0][::-1])
            )

            # 視点ベクトル投影の楕円面積を計算
            semi_major_axis_projected = width_projected / 2
            semi_minor_axis_projected = height_projected / 2
            area_projected = (
                np.pi * semi_major_axis_projected * semi_minor_axis_projected
            )

            # 結果をCSVファイルに保存
            writer.writerow(
                [
                    particle_id,
                    area_projected,
                    semi_major_axis_projected,
                    semi_minor_axis_projected,
                    area_noWarp,
                    semi_major_noWarp,
                    semi_minor_noWarp,
                    event_counts[peak_index],
                ]
            )

    # 画像をファイルに保存
    # img_output_file = f'{file_path.split(".pkl")[0]}.png'
    # plt.savefig(img_output_file, dpi=300)
    # plt.close()

    # print(f"Results saved to {output_file} and image saved to {img_output_file}")


def main() -> None:
    args = parser.parse_args()
    process_file(args.input)


if __name__ == "__main__":
    main()
