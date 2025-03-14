import os
import glob
import subprocess

# ディレクトリのパスを指定
directory = "/Users/yasu-air/Dropbox/yhLab/KS24-16/cam1"  # 処理するディレクトリを指定してください

# `directory` 内のすべてのCSVファイルを取得
pkl_files = glob.glob(os.path.join(directory, "*.pkl"))

# すべてのpklファイルに対して `saveParticleSize.py` を実行
for pkl_file in pkl_files:
    try:
        print(f"Processing file: {pkl_file}")
        # saveParticleSize.py を実行
        subprocess.run(["python3", "saveParticleSize.py", "-i", pkl_file], check=False)
        print(f"Finished processing: {pkl_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {pkl_file}: {e}")
