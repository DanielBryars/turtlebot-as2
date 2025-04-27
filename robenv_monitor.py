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

#ACTIONS - make the code easier to read
FORWARDS = 1
LEFT = 0
RIGHT = 2

class vRobEnv(RobEnv):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.nF = len(self.scans)
        print('state size(laser beams)=', self.nF)


    def reset(self):
        print("Reset called ignoring")

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

        reward = sum([
            -0.1, #Don't like steps

            -5 * self.nearly_atwall(), #Dont like getting too close to walls 

            -2 * self.Δgoal_dist, #Going towards the goal is good, away is bad
            #-0.5 * self.Δθgoal_dist / pi

            0.2 * (a == FORWARDS), # let's promote moving forward

            10 * self.atgoal(self.goal_dist) #Like goals
            ])
        
        if self.verbose and reward>-1: print('reward =', reward)#; print(f'action = {a}')
        return reward
    
    # overriding state representation, you may only use the laser self.scans
    def s_(self):
        max, min = self.max_range, self.min_range
        # returns a normalise and descritised componenets
        return  1*(((self.scans - min)/(max - min))>=.5)
        

print("Initiating ROS")

if not ros.ok():
   ros.init()

env = vRobEnv(resetworld = False)
try:
    while True:
        env.spin_n(1)
        
        # Clear console
        os.system('clear')  # or 'cls' if on Windows
                
        # Header

        #for i, s in enumerate(env.scans):
        #    print(f"{i:3d} | {s:.2f}m")
        
        print("="*40)
        print(" Robot ENV Monitor")
        print("="*40)
        
        print(f"hasattr(self, 'Δgoal_dist'):{hasattr(env, 'Δgoal_dist'):}")
        print(f"Reward LEFT:{env.reward(LEFT)}")
        print(f"Reward FORWARD:{env.reward(FORWARDS)}")
        print(f"Reward RIGHT:{env.reward(RIGHT)}")
        print(f"nearly atwall: {env.nearly_atwall()}")
        print(f"atwall: {env.atwall()}")
        print(env.s_())
        
        print("="*40)


        #plot_laser_sectors(env.scans)

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopped by user.")
