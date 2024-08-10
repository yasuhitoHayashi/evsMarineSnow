import numpy as np
import matplotlib.pyplot as plt

# 円のプロパティ
radius = 1.0
num_points = 100

# 円の点を生成
theta = np.linspace(0, 2 * np.pi, num_points)
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)
z_circle = np.zeros_like(x_circle)  # xy平面なのでz=0

# 初回の回転をするための角度
x_rotation_angle = np.pi / 6  # x軸方向の視点の角度（ラジアン）
y_rotation_angle = np.pi / 4  # y軸方向の視点の角度（ラジアン）

# 3D空間での初回の回転行列（x軸およびy軸を中心に回転）
rotation_matrix_x = np.array([
    [1, 0, 0],
    [0, np.cos(x_rotation_angle), -np.sin(x_rotation_angle)],
    [0, np.sin(x_rotation_angle), np.cos(x_rotation_angle)]
])

rotation_matrix_y = np.array([
    [np.cos(y_rotation_angle), 0, np.sin(y_rotation_angle)],
    [0, 1, 0],
    [-np.sin(y_rotation_angle), 0, np.cos(y_rotation_angle)]
])

# x軸およびy軸方向の回転を組み合わせた回転
initial_rotation_matrix = rotation_matrix_y @ rotation_matrix_x

# 初回の回転による斜めから見た円を計算
xyz_circle = np.vstack((x_circle, y_circle, z_circle))
xyz_oblique = initial_rotation_matrix @ xyz_circle

# 斜め視点の2D投影を計算（xy平面に投影）
x_oblique = xyz_oblique[0]
y_oblique = xyz_oblique[1]

# 視点ベクトルを保存（初回の回転の影響を記録）
view_vector = np.array([np.sin(y_rotation_angle), 
                        np.sin(x_rotation_angle) * np.cos(y_rotation_angle), 
                        np.cos(x_rotation_angle) * np.cos(y_rotation_angle)])

# 逆回転を視点ベクトルを使って計算
# 視点ベクトルから回転を逆にするための角度を推定
restoration_rotation_matrix_y = np.array([
    [np.cos(-y_rotation_angle), 0, np.sin(-y_rotation_angle)],
    [0, 1, 0],
    [-np.sin(-y_rotation_angle), 0, np.cos(-y_rotation_angle)]
])

restoration_rotation_matrix_x = np.array([
    [1, 0, 0],
    [0, np.cos(-x_rotation_angle), -np.sin(-x_rotation_angle)],
    [0, np.sin(-x_rotation_angle), np.cos(-x_rotation_angle)]
])

# 逆回転を適用して元の円を復元
restoration_rotation_matrix = restoration_rotation_matrix_x @ restoration_rotation_matrix_y
xyz_restored = restoration_rotation_matrix @ xyz_oblique

# 2Dプロットに復元した円を描画
x_restored = xyz_restored[0]
y_restored = xyz_restored[1]

# プロット
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# 元の円
axs[0].plot(x_circle, y_circle, label='Original Circle')
axs[0].set_title('Original Circle (2D)')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_xlim(-1.5, 1.5)
axs[0].set_ylim(-1.5, 1.5)
axs[0].set_aspect('equal', 'box')

# 斜めから見た円（2D投影）
axs[1].plot(x_oblique, y_oblique, label='Oblique View Circle')
axs[1].set_title('Oblique View Circle (2D Projection)')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_xlim(-1.5, 1.5)
axs[1].set_ylim(-1.5, 1.5)
axs[1].set_aspect('equal', 'box')

# 復元した円
axs[2].plot(x_restored, y_restored, label='Restored Circle')
axs[2].set_title('Restored Circle (2D)')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
axs[2].set_xlim(-1.5, 1.5)
axs[2].set_ylim(-1.5, 1.5)
axs[2].set_aspect('equal', 'box')

plt.tight_layout()
plt.show()
