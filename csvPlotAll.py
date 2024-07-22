import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSVファイルから粒子のデータを読み込む
particle_movements_file = 'particle_movements.csv'
particle_movements_df = pd.read_csv(particle_movements_file)

# 5点以上の粒子のみ3Dプロットを作成する関数
def plot_filtered_particles(particle_movements_df, min_points=5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 各粒子のデータポイントの数をカウント
    particle_counts = particle_movements_df['particle_id'].value_counts()
    
    # データポイントがmin_points以上の粒子のみプロット
    for particle_id in particle_counts[particle_counts >= min_points].index:
        particle_data = particle_movements_df[particle_movements_df['particle_id'] == particle_id]
        ax.plot(particle_data['x'], particle_data['y'], particle_data['time'], label=f'Particle {particle_id}', marker='o')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0,1280)
    ax.set_ylim(0,720)
    ax.set_zlabel('Time')
    ax.set_title(f'Particles with at least {min_points} points')
    ax.legend()
    plt.show()

# 5点以上の粒子をプロット
plot_filtered_particles(particle_movements_df, min_points=5)