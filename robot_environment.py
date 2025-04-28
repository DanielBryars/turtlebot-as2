import torch
import numpy as np
from env.robot import *
import numpy as np
from math import pi
from time import sleep
#from tqdm import tqdm

from tqdm.notebook import tqdm
import sys
import termios
import tty
import select
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import time
import matplotlib.pyplot as plt
import numpy as np


#ACTIONS - make the code easier to read
FORWARDS = 1
LEFT = 0
RIGHT = 2

class vRobEnv(RobEnv):
    def __init__(self, **kw):
        self.ignoreReset = kw.pop('ignoreReset', False)
        super().__init__(ignoreReset=self.ignoreReset, **kw)

        self.nF = len(self.scans)
        print('state size(laser beams)=', self.nF)

    def reset(self):
        '''Override the original so we can skip the reset (used for monitoring applications)'''
        if (self.ignoreReset):
            print("Reset called BUT self.ignoreReset is True, so ignoring")
            return self.s_()
        else:
            return super().reset()

    def nearly_atwall(self):
        # check only 2*rng front scans for collision, given the robot does not move backward
        rng = int(len(self.scans)*self.nscansρ//2)  # nscansρ//2 left and nscansρ//2 right 
        return np.r_[self.scans[-rng:], self.scans[:rng]].min() <= (self.min_range + 0.1) 
        # return self.scans.min() <= self.min_range

    # overridding reward_, 
    # you may use goal_dist, Δgoal_dist, θgoal_dist, Δθgoal_dist or at_wall and at_goal
    def reward_(self, a):

        if not hasattr(self, 'Δgoal_dist'):
            return 0

        if a == FORWARDS and self.θgoal_dist < 0.2:  
            alignment_reward = +0.5  
        elif (a == LEFT or a == RIGHT) and self.θgoal_dist > 0.5:
            alignment_reward = +0.5  
        else:
            alignment_reward = 0

        #Don't like steps
        per_step_reward = -0.1 

        #Dont like getting too close to walls 
        anti_crash_into_wall_reward = -5 * self.nearly_atwall() 

        #Going towards the goal is good, away is bad
        goal_getting_closer_reward = -2 * self.Δgoal_dist

        #Going goal direction is good, away is bad
        goal_direction_better_reward = -0.5 * self.Δθgoal_dist

        # let's promote moving forward
        move_forward_reward = 0.2 * (a == FORWARDS) 

        #The goal is great
        goal_reached_reward = 10 * self.atgoal(self.goal_dist)

        reward = sum([
            per_step_reward, 
            anti_crash_into_wall_reward,
            goal_getting_closer_reward,
            alignment_reward,
            move_forward_reward,
            goal_reached_reward])
        
        if self.verbose and reward>-1:
            print(f"per_step_reward: {per_step_reward}, ")
            print(f"anti_crash_into_wall_reward: {anti_crash_into_wall_reward}")
            print(f"goal_getting_closer_reward: {goal_getting_closer_reward}")
            print(f"goal_direction_better_reward: {goal_direction_better_reward}")
            print(f"move_forward_reward: {move_forward_reward}")
            print(f"goal_reached_reward: {goal_reached_reward}")
            print('reward =', reward)#; print(f'action = {a}')
        
        return reward
    
    # overriding state representation, you may only use the laser self.scans
    # original, I don't think this works well because the state doesn't really "know" where it is.
    #def s_(self):
    #    max, min = self.max_range, self.min_range
    #    # returns a normalise and descritised componenets
    #    return  1*(((self.scans - min)/(max - min))>=.5)

    def s_(self):
        #State is if we're near a wall
        states = (self.scans <= 0.3).astype(int)
        assert states.shape[0] == 64 # self.nF
        return states


class HandcraftedFeatureExtractor:
    NUM_FEATURES = 6

    def __init__(self, 
                 near_threshold=0.5, 
                 wall_threshold=1.2, 
                 max_range=3.5,
                 min_range=0.0):

        #This implementation depends on 360 laser lines
        assert get_nscans_LiDAR() == 360        
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
        
        assert len(labels) == self.NUM_FEATURES

        return labels, np.array(features, dtype=np.float32), left_scan_average, front_scan_average, right_scan_average

    def detect_cylinder_like(self, scan_data, range_threshold=0.5, cluster_width=5, variance_threshold=0.05):
        for i in range(len(scan_data) - cluster_width):
            window = scan_data[i:i+cluster_width]
            if np.all(window < range_threshold) and np.var(window) < variance_threshold:
                return 1
        return 0
  
class vRobEnvCornerDetector(vRobEnv):
    '''This uses hand crafted features'''
    def __init__(self, **kw):
        super().__init__(**kw)
        self.nF = HandcraftedFeatureExtractor.NUM_FEATURES
        print('state size=', self.nF)
 
    def s_(self):

        #walls (front, left, right)
        #corners (front+left, front+right)
        #and pillar signal (from above)?

        fe = HandcraftedFeatureExtractor(
            near_threshold=0.5,
            wall_threshold=1.2,
            max_range=self.max_range,
            min_range=self.min_range)
        
        _, features_detected, _, _, _ = fe.extract_features(self.scans)
        
        # returns a normalise and descritised componenets
        return  features_detected
    