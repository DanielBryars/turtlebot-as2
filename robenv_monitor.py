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
from robot_environment import *
from robot_environment import *

#ACTIONS - make the code easier to read
FORWARDS = 1
LEFT = 0
RIGHT = 2

        

print("Initiating ROS")

if not ros.ok():
   ros.init()

env = vRobEnv(ignoreReset = True)
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
        
        #goal_dist, Δgoal_dist, θgoal_dist, Δθgoal_dist or at_wall and at_goal
        
        try:
            print(f"goal_dist:{env.goal_dist},Δgoal_dist:{env.Δgoal_dist},θgoal_dist:{env.θgoal_dist}, Δθgoal_dist:{env.Δθgoal_dist},at_wall:{env.at_wall},at_goal:{env.at_goal} ")        
        except:
            #goal_dist and so on are created lazily when an odom message arrives
            #so they might not be created yet
            pass

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
