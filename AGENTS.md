# Repository Guidelines

## プロジェクト構成（要点）
- C++: `real_time_tracking.cpp`（SDL2ビューア／追跡）、`gaussian_EKF_initial.cpp`、`particle_tracking.cpp`（pybind拡張）。
- Python: `trackParticlesC.py`（CSV→PKL追跡）、`plotTrajectory.py`（可視化）、その他ユーティリティ。
- 生成物: `build/` に CMake バイナリ、`particle_tracking.*.so` は拡張モジュール。データはリポジトリ直下（`*.csv`, `*.pkl`, `*.raw`）。

## セットアップと実行（最短経路）
- 依存関係（例: macOS/Homebrew）: `brew install sdl2 opencv eigen boost`、`python -m pip install -U pybind11`。
- C++ビルド: `mkdir -p build && cd build && cmake .. && make -j` → 実行: `./real_time_tracking`。
- Python拡張: `python setup.py build_ext --inplace`。
- 追跡（CSV→PKL）: `python trackParticlesC.py -i suruga_test_short.csv`。
- 可視化: `python plotTrajectory.py -i suruga_test_short.pkl --event_threshold 1000`。

## スタイルと命名（統一指針）
- Python: 4スペース、snake_case、型のある引数/戻り値は可能なら注釈を付与。
- C++: C++17、インデントはスペース、関数/変数はsnake_case、型はPascalCase。`#include`は標準/外部/ローカルの順に整理。
- ファイル: 大容量生成物はコミットしない。データ名は `recording_YYYY-MM-DD_HH-MM-SS.*` のように一貫性を持たせる。

## テスト（軽量チェック）
- 公式テストは未整備。以下のスモークテストを基準とする:
  - CSV→PKL生成後、`plotTrajectory.py` で出力を確認。
  - `real_time_tracking` 実行でウィンドウが表示され、エラーが出ないこと。
- テストを追加する場合は `tests/` に `test_*.py`（pytest）で配置。

## コミットとPR
- メッセージは簡潔な命令形（例: `feat(tracking): improve EKF gating`）。
- PRには変更理由、再現手順（コマンド例）、期待される出力（PKL/図）、依存関係の追加や変更点を記載。必要に応じてスクリーンショットを添付。

## データとセキュリティ
- `*.raw`/`*.pkl` は巨大化しやすい。外部ストレージの利用と `.gitignore` 管理を推奨。
- 絶対パスのハードコードは避け、CLI引数（例: `-i/--input`）で指定できるようにする。
