import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QListWidget, QListWidgetItem
from PyQt5.QtCore import QTimer

class TrackingApp(QMainWindow):
    def __init__(self, cluster_data):
        super().__init__()
        self.cluster_data = cluster_data
        self.time_bins = self.cluster_data['time_bin'].unique()
        self.current_time_bin_index = 0
        self.tracking_data = {}
        self.selected_id = None
        
        self.initUI()

        # タイマーの初期化
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_next)
        self.timer.start(10)  # 0.2秒ごとに次のフレームへ

    def initUI(self):
        self.setWindowTitle('Particle Tracking')
        
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout(self.central_widget)
        
        self.canvas_widget = QWidget(self.central_widget)
        self.canvas_layout = QVBoxLayout(self.canvas_widget)
        self.layout.addWidget(self.canvas_widget)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas_layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.controls_widget = QWidget(self.central_widget)
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.layout.addWidget(self.controls_widget)
        
        self.id_input_label = QLabel('Enter Tracking ID:', self.controls_widget)
        self.controls_layout.addWidget(self.id_input_label)
        
        self.id_input = QLineEdit(self.controls_widget)
        self.controls_layout.addWidget(self.id_input)
        
        self.id_list = QListWidget(self.controls_widget)
        self.id_list.itemClicked.connect(self.on_id_click)
        self.controls_layout.addWidget(self.id_list)
        
        self.next_button = QPushButton('Next', self.controls_widget)
        self.next_button.clicked.connect(self.on_next)
        self.controls_layout.addWidget(self.next_button)
        
        self.update_plot()

    def update_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.current_time_bin_index >= len(self.time_bins):
            print("No more time bins to display.")
            return
        
        current_time_bin = self.time_bins[self.current_time_bin_index]
        time_bin_data = self.cluster_data[self.cluster_data['time_bin'] == current_time_bin]
        
        if time_bin_data.empty:
            print(f"No data for time bin {current_time_bin}. Skipping.")
            self.current_time_bin_index += 1
            self.update_plot()
            return
        
        for _, row in time_bin_data.iterrows():
            points = np.array(row['cluster_points'])
            centroid = [row['centroid_x'], row['centroid_y']]
            
            ax.scatter(points[:, 0], points[:, 1], c='blue', s=10)
            ax.scatter(centroid[0], centroid[1], c='red', marker='x', s=50)
        
        ax.set_xlim(0, 1280)  # x軸の範囲を設定
        ax.set_ylim(720, 0)   # y軸の範囲を反転して設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Time Bin: {current_time_bin}')
        
        self.canvas.draw()

    def on_click(self, event):
        if self.selected_id is not None and event.xdata is not None and event.ydata is not None:
            self.tracking_data[(self.time_bins[self.current_time_bin_index], (event.xdata, event.ydata))] = self.selected_id
            print(f"Linked Tracking ID {self.selected_id} with centroid at ({event.xdata}, {event.ydata})")

    def on_id_click(self, item):
        self.selected_id = item.text()
        print(f"Selected Tracking ID {self.selected_id}")

    def on_next(self):
        self.current_time_bin_index += 1
        if self.current_time_bin_index < len(self.time_bins):
            self.update_plot()
        else:
            self.timer.stop()  # 終了したらタイマーを停止する
        
        unique_ids = list(set(self.tracking_data.values()))
        self.id_list.clear()
        for uid in unique_ids:
            self.id_list.addItem(uid)
        
        self.id_input.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # クラスタリング結果のPickleファイルを読み込み
    pkl_input_file = 'clustering_bat_results.pkl'
    with open(pkl_input_file, 'rb') as f:
        results_df = pickle.load(f)
    
    main_window = TrackingApp(results_df)
    main_window.show()
    sys.exit(app.exec_())