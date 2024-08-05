import pandas as pd
import numpy as np
import pickle
from collections import deque
from tqdm import tqdm

# CSVファイルのパス
file_path = 'suruga_test_short.csv'
# file_path = '195_falling_particles.csv'

# CSVファイルの読み込み
data = pd.read_csv(file_path, header=None, names=['x', 'y', 'polarity', 'time'])

# 極性が1のデータのみを使用
data_filtered = data[data['polarity'] == 1].copy()
print(f"フィルタリング後のデータ数: {len(data_filtered)}")

# パーティクルを管理するクラス
class Particle:
    def __init__(self, particle_id, x, y, time):
        self.particle_id = particle_id
        self.events = deque([(x, y, time)])  # Events are stored with coordinates and time
        self.mass = 1
        self.centroid = np.array([x, y])
        self.centroid_history = [(time, self.centroid.copy())]
    
    def add_event(self, x, y, time):
        self.events.append((x, y, time))
        self.mass += 1
        # Remove old events outside the 10 ms window
        while self.events and self.events[0][2] < time - 10000:
            self.events.popleft()
        
        # Update centroid
        coords = np.array([event[:2] for event in self.events])
        self.centroid = coords.mean(axis=0)
        self.centroid_history.append((time, self.centroid.copy()))
    
    def is_active(self, current_time, m_threshold):
        # A particle is inactive if no events in 2 ms and mass < M
        if self.events and self.events[-1][2] < current_time - 2000:
            return self.mass > m_threshold
        return True

# パーティクル追跡
def track_particles(data, spatial_radius=24, time_window=2000, m_threshold=10000):
    particles = {}
    particle_id_counter = 0
    active_particles = []

    spatial_radius_squared = spatial_radius ** 2  # Precompute for efficiency

    for event in tqdm(data.itertuples(index=False, name='Event'), total=len(data), desc="Processing events"):
        x, y, time = event.x, event.y, event.time
        
        # Check for nearby active particles
        nearby_particle_id = None
        for particle in active_particles:
            # Calculate spatial distance squared for efficiency
            if (np.sum((particle.centroid - np.array([x, y])) ** 2) <= spatial_radius_squared and
                    particle.events[-1][2] >= time - time_window):
                nearby_particle_id = particle.particle_id
                break
        
        if nearby_particle_id is not None:
            # Update existing particle
            particles[nearby_particle_id].add_event(x, y, time)
        else:
            # Create a new particle
            particle_id_counter += 1
            new_particle = Particle(particle_id_counter, x, y, time)
            particles[particle_id_counter] = new_particle
            active_particles.append(new_particle)

        # Clean up inactive particles
        active_particles = [p for p in active_particles if p.is_active(time, m_threshold)]
    
    return particles

# パーティクル追跡の実行
particles = track_particles(data_filtered)
print(f"追跡されたパーティクル数: {len(particles)}")

# 結果を辞書として出力
particle_data = {pid: {'centroid': p.centroid.tolist(), 'mass': p.mass, 'events': list(p.events)} for pid, p in particles.items()}

# 重心履歴を保存する辞書
centroid_history = {pid: p.centroid_history for pid, p in particles.items()}

# クラスタリング結果をPickleファイルに保存
particle_output_file = 'particle_tracking_results.pkl'
centroid_output_file = 'centroid_history_results.pkl'

with open(particle_output_file, 'wb') as f:
    pickle.dump(particle_data, f)
print(f"パーティクル追跡結果を {particle_output_file} に保存しました")

with open(centroid_output_file, 'wb') as f:
    pickle.dump(centroid_history, f)
print(f"重心履歴を {centroid_output_file} に保存しました")