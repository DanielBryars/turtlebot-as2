#Build up an intuition for if the features I'm using are enough of a signal to find the goal.


import time
import numpy as np
import os
import sys
import tty
import termios
import time
from math import pi
from env.robot import *  # RobEnv, ros, etc
import select
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self, 
                 near_threshold=0.5, 
                 wall_threshold=1.2, 
                 max_range=3.5,
                 min_range=0.0):
        
        self.near_threshold = near_threshold
        self.wall_threshold = wall_threshold
        self.max_range = max_range
        self.min_range = min_range

    def extract_features(self, scan_data):
        scans = np.clip(scan_data, self.min_range, self.max_range)
        scans[np.isnan(scans)] = self.max_range

        n = len(scans)

        """
                FRONT [0]
                   ↑
     FRONT/LEFT [315]     FRONT/RIGHT [45]
          ↖                      ↗
LEFT [270] ←                     → RIGHT [90]
          ↙                      ↘
     BACK/LEFT [225]     BACK/RIGHT [135]
                   ↓
               BACK [180]
        """


        width_each_side = 5

        left_scan_range = scans[270-width_each_side:270+width_each_side]
        left_scan_average = np.mean(left_scan_range)

        front_scan_range = scans[np.r_[360-width_each_side:360, 0:width_each_side]]
        front_scan_average = np.mean(front_scan_range)
        
        right_scan_range = scans[90-width_each_side:90+width_each_side]
        right_scan_average = np.mean(right_scan_range)

        def is_wall_detected(avg_distance):
            return avg_distance < self.wall_threshold
                
        def is_wall_too_near(avg_distance):
            return avg_distance < self.near_threshold
        
        labels = []
        features = []

        labels.append('Left Wall Ahead')
        features.append(is_wall_detected(left_scan_average))
        labels.append('Left Wall Getting Close')
        features.append(is_wall_too_near(left_scan_average))

        labels.append('Front Wall Ahead')
        features.append(is_wall_detected(front_scan_average))
        labels.append('Front Wall Getting Close')
        features.append(is_wall_too_near(front_scan_average))

        labels.append('Right Wall Ahead')
        features.append(is_wall_detected(right_scan_average))
        labels.append('Right Wall Getting Close')
        features.append(is_wall_too_near(right_scan_average))

        #labels.append('Pillar Recognised')
        #features.append(self.detect_cylinder_like(scans))
        
        return labels, np.array(features, dtype=np.float32), left_scan_average, front_scan_average, right_scan_average

    def detect_cylinder_like(self, scan_data, range_threshold=0.5, cluster_width=5, variance_threshold=0.05):
        for i in range(len(scan_data) - cluster_width):
            window = scan_data[i:i+cluster_width]
            if np.all(window < range_threshold) and np.var(window) < variance_threshold:
                return 1
        return 0

print("Initiating ROS")

if not ros.ok():
   ros.init()

env = RobEnv()
fe = FeatureExtractor(
    near_threshold=0.5,
    wall_threshold=1.2,
    max_range=env.max_range,
    min_range=env.min_range
)
try:
    while True:
        env.spin_n(1)
        labels, features_detected, left_scan_average, front_scan_average, right_scan_average = fe.extract_features(env.scans)
        
        # Clear console
        os.system('clear')  # or 'cls' if on Windows
        
        # Header

        #for i, s in enumerate(env.scans):
        #    print(f"{i:3d} | {s:.2f}m")
        
        print("="*40)
        print(" Robot Feature Detector and reward detector")
        print("="*40)
        
        print(f"left:{left_scan_average:.2f}")
        print(f"front:{front_scan_average:.2f}")
        print(f"right:{right_scan_average:.2f}")
        print(features_detected)

        for label, value in zip(labels, features_detected):
            status = "🟠" if value else "➖"
            print(f"{label:30s} : {status}")
        
        print("="*40)

        print(f"atwall: {env.atwall()}")

        #plot_laser_sectors(env.scans)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped by user.")
