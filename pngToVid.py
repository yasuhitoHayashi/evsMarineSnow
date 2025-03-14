import cv2
import os
import re
import numpy as np

# 画像ファイル名から日時を抽出する正規表現パターン
date_pattern = re.compile(r"(\d{8}_\d{6})")

# フォルダ内のすべてのpngファイルを取得
image_folder = '/Users/yasu-air/yhLab/KS24-16/cam2/probit'  # 画像フォルダのパスに変更
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# 画像を時間順にソート（ファイル名から日時を抽出してソート）
images = sorted(images, key=lambda x: date_pattern.search(x).group(1))

# 動画の出力先
video_output = 'cam2Probit.mp4'

# 動画フレームサイズを定義
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 動画ファイル作成 (コーデック指定、フレームレート2fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデック指定
video = cv2.VideoWriter(video_output, fourcc, 2, (width, height))  # フレームレート2fps

# フォントを定義 (フォントサイズや色を調整可能)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2  # フォントサイズ
font_color = (0, 0, 0)  # 白色
font_thickness = 3  # フォントの太さ
text_position = (width - 1000, 200)  # 右上にテキストを表示 (位置は調整可能)

# 各画像に日時情報を追加し、動画にフレームを追加
for image_name in images:
    # 画像のファイル名から日時情報を抽出
    match = date_pattern.search(image_name)
    if match:
        timestamp = match.group(1)  # '20240823_090057' のような日時情報
    
    # 画像を読み込む
    img_path = os.path.join(image_folder, image_name)
    img = cv2.imread(img_path)
    print(timestamp)
    # 画像に日時情報を追加 (OpenCVを使用)
    cv2.putText(img, timestamp, text_position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    
    # フレームとして動画に追加
    video.write(img)

# 動画作成完了
video.release()
cv2.destroyAllWindows()

print(f"Video saved as {video_output}")