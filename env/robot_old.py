'''
   Author: Abdulrahman Altahhan,  2025.
   version: 3.4

    This library of functionality in RL that aims for simplicity and general
    insight into how algorithms work, these libraries are written from scratch
    using standard Python libraries (numpy, matplotlib etc.).
    Please note that you will need permission from the author to use the code
    for research, commercially or otherwise.
'''

import rclpy as ros
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg  import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.msg import ModelStates

import numpy as np
from random import randint
from math import atan2, atan, pi
import matplotlib.pyplot as plt

from rl.rlln import *
# ====================================================================================================
def name(): return 'node'+str(randint(1,1000))
demoRobot = demoGame
# ====================================================================================================

class RobEnv(Node):
# initialisation--------------------------------------------------------------
    # frequency: how many often (in seconds) the spin_once is invoked, or the publisher is publishing to the /cmd_vel
    def __init__(self, name=name(),
                 freq=1/20, n=28,
                 speed=.5, θspeed=pi/6,
                 rewards=[0, -10, 0, 0],
                 verbose=False):
        super().__init__(name)
        
        self.freq = freq
        self.n = n
        
        self.speed = speed
        self.θspeed = round(θspeed,2)
        
        self.robot = Twist()
        self.rewards = rewards
        self.verbose = verbose

        # do not change----------------------------------------------------
        self.x = 0 # initial x position
        self.y = 0 # initial y position
        self.θ = 0 # initial θ angle
        self.scans = np.zeros(60) # change to how many beams you are using
        self.t = 0
        
        self.tol = .6  # meter from goal as per the requirement (tolerance)
        self.goals =  [[2.0, 2.0], [-2.0, -2.0]]
        # -----------------------------------------------------------------
        
        self.controller = self.create_publisher(Twist, '/cmd_vel', 0)
        self.timer = self.create_timer(self.freq, self.control)

        self.scanner = self.create_subscription(LaserScan,   '/scan',          self.scan, 0)
        self.odometr = self.create_subscription(Odometry,    '/odom',          self.odom, 0)
        self.position = self.create_subscription(ModelStates, '/model_states', self.pos, 0)
        
        self.max_range = 3.5
        self.min_range = .35
       

        # establish a reset client 
        self.reset_world = self.create_client(Empty, '/reset_world')
        while not self.reset_world.wait_for_service(timeout_sec=2.0):
            print('world client service...')


        # compatibility----------------------------------------------
        nturns = 15 # number of turns robot takes to complete a full circle
        resol = speed/2
        
        θresol = 2*pi/nturns
        dims = [4,4]
        self.xdim = dims[0]  # realted to the size of the environment
        self.ydim = dims[1]  # realted to the size of the environment
        
        self.resol = round(resol,2)
        self.θresol = round(θresol,2)
        
        self.cols  = int(self.xdim//self.resol) +1   # number of grid columns, related to linear speed
        self.rows  = int(self.ydim//self.resol) +1   # number of grid rows,    related to linear speed
        self.orts  = int(2*pi//self.θresol)     +1   # number of angles,       related to angular speed

        self.nC = self.rows*self.cols              # Grid size
        self.nS = self.rows*self.cols*self.orts # State space size
        self.nA = 3

        self.Vstar = None # for compatibility
        self.figsize0 = (12, 2) # for compatibility
        # --------------------------------------------------------------- 
        # self.rate = self.create_rate(30)
        self.reset()
        
        print('speed  = ', self.speed)
        print('θspeed = ', self.θspeed)
        print('freq   = ', self.freq)

# sensing--------------------------------------------------------------
    # odometry (position and orientation) readings
    def odom(self, odoms):
        self.x = round(odoms.pose.pose.position.x, 1)
        self.y = round(odoms.pose.pose.position.y, 1)
        self.θ = round(self.yaw(odoms.pose.pose.orientation),2) 
        self.odom = np.array([self.x, self.y, self.θ])
        # if self.verbose: print('odom = ',  self.odom )

    # laser scanners readings
    def scan(self, scans):
        self.scans = np.array(scans.ranges)
        self.scans = np.clip(self.scans, self.min_range, self.max_range)
        self.scans[self.scans == np.nan] = 0        
        # if self.verbose: print('scan = ', self.scans[:10].round(2))
        # if self.verbose: print('scan = ', np.r_[self.scans[-5:], self.scans[:5]].round(2))

    def pos(self, msg):
        robot_index = msg.name.index('tr')  # Replace with your model name
        position = msg.pose[robot_index].position
        # self.get_logger().info(f'Real Position: x={position.x}, y={position.y}, z={position.z}')
        print(f'Real Position: x={position.x}, y={position.y}, z={position.z}')

    def yaw(self, orient):
        """Convert a quaternion to an Euler yaw angle (in radians, [0, 2π]).
        The yaw is computed using the atan2 function for better numerical stability.
        Args:
            orient: A quaternion object with attributes x, y, z, w.
        Returns:
            float: The yaw angle in radians within the range [0, 2π].
        """
        x, y, z, w = orient.x, orient.y, orient.z, orient.w
        yaw = atan2(2.0*(x*y + w*z), w*w + x*x - y*y - z*z)
        return yaw if yaw>0 else yaw + 2*pi # in radians, [0, 2pi]

    # angular distance of robot to a specific goal.......................................
    def θdistgoal(self, goal):
        xgoal, ygoal = goal
        θgoal = atan2(abs(self.x - xgoal), abs(self.y - ygoal))
        θgoal = θgoal if θgoal >= 0 else θgoal+ 2 * pi # [0, 2pi]

        θdiff = abs(θgoal - self.θ)
        return round(min(θdiff, 2*pi - θdiff), 2)
    
    # Eucleadian distance of robot to a goal.............................................
    def distgoal(self, goal):
        xgoal, ygoal = goal
        return round(((self.x - xgoal)**2 + (self.y - ygoal)**2)**.5, 2)

    # # Eucleadian distance of robot to nearest goal......................................
    # def distgoal(self):
    #     dists = [np.inf, np.inf]        # distances of robot to the two goals
    #     for goal, (xgoal, ygoal) in enumerate(self.goals):
    #         dists[goal] = (self.x - xgoal)**2 + (self.y - ygoal)**2

    #     dist = min(dists)         # nearest goal distance
    #     goal = dists.index(dist)  # nearest goal index

    #     # if self.verbose: print('seeking goal ____________________', goal)

    #     return round(dist**.5, 2), goal

    # robot reached goal ...............................................................
    def atgoal(self):
        tol, x, y = self.tol,  self.x, self.y
        atgoal = False
        for xgoal, ygoal in self.goals:
            atgoal = xgoal + tol > x > xgoal - tol  and  \
                     ygoal + tol > y > ygoal - tol

            if atgoal:
                print('Goal has been reached woohoooooooooooooooooooooooooooooo!!')
                break

        return atgoal

    # robot hits a wall...................................................................
    def atwall(self, rng=5):
        # check only 2*rng front scans for collision, given the robot does not move backward
        return np.r_[self.scans[-rng:], self.scans[:rng]].min() < self.min_range 
        #return self.scans.min()<self.min_range

    # reward function to produce a suitable policy..........................................
    def reward_(self, a):
        # sa_type 0-reached a goal, 1-hits a wall 2-moved forward or 3-turn
        sa_type = [self.atgoal(), self.atwall(), a==1, a!=1].index(True)

        # obtain the distance and the angular distance to goals
        for g, goal in enumerate(self.goals):
            dists[g] = self.distgoal(goal)
            θdists[g] = self.θdistgoal(goal)
            θdist = round(abs(abs(self.θ - θgoal) - pi*g), 2) # subtract pi if it's goal 1 (behind)
            #  reward/penalise robot relative to its orientation towards a goal

        # reset without restarting an episode if the robot hits a wall
        if sa_type==1:
            self.reset()

        return self.reward(sa_type, dist, θdist), sa_type==0, sa_type==1

    # override this function for fruther adjust your own reward
    def reward(self, sa_type, dist, θdist):
        """
        set the reward based on the agent's state (and potentially distances to the nearest goal).
        Args:
            dist (float): The distance to the goal.
            θdist (float): The angular distance to the goal.
            sa_type (int): The state and action types:
                - 0: Reached the goal       - 2: Moved forward
                - 1: Hit a wall             - 3: Took a turn
        Returns:
            float: The adjusted reward.
        """
        return self.rewards[sa_type] # rewards is set by the constructor of the env isntance 

# State representation-------------------------------------------------
   # override with a suitable state representation
    def s_(self):
        
        self.xi = int((self.x+self.xdim/2)//self.resol)     # x index = col, assuming the grid middle is (0,0)
        self.yi = int((self.y+self.ydim/2)//self.resol)     # y index = row, assuming the grid middle is (0,0)
        
        # pi/2 to be superficially resilient to slight angle variation to keep θi unchanged
        self.θi = int((self.θ+pi/2)%(2*pi)//self.θresol)
        
        self.si = self.xi + self.yi*self.cols     # position state in the grid
        self.s = self.nC*(self.θi) + self.si      # position state with orientation
        # if self.verbose: print('grid cell= ', self.si, 'state = ', self.s)

        return self.s 

# Control--------------------------------------------------------------    
    def spin_n(self, n):
        for _ in range(n):
            ros.spin_once(self)

    def control(self):
        self.controller.publish(self.robot)

    # move then stop to get a defined action
    def step(self, a=1, speed=None, θspeed=None):
        if speed is None: speed = self.speed
        if θspeed is None: θspeed = self.θspeed

        self.t +=1
        # if self.verbose: print('step = ', self.t)
        
        if  a==-1: self.robot.linear.x  = -speed  # backwards
        elif a==1: self.robot.linear.x  =  speed  # forwards
        elif a==0: self.robot.angular.z = -θspeed # turn left
        elif a==2: self.robot.angular.z =  θspeed # turn right

        # Now move and stop so that we can have a well defined actions  
        self.spin_n(self.n) if a==1 else self.spin_n(self.n//2)
        self.stop()

        reward, done, wall = self.reward_(a)
        return self.s_(), reward, done, {}
        
    def stop(self):
        self.robot.linear.x = .0
        self.robot.angular.z = .0
        #  spin less so that we have smoother actions
        self.spin_n(self.n//8)

# reseting--------------------------------------------------------------
    def reset(self):
        if self.verbose: print('resetting world..........................................')
        # to ensure earlier queued actions are flushed, there are better ways to do this
        for _ in range(1): self.reset_world.call_async(Empty.Request())
        for _ in range(2): self.step(a=1, speed=0.001)  # move slightly forward to update the odometry to prevent repeating an episode unnecessary
        for _ in range(1): self.reset_world.call_async(Empty.Request())
        
        return self.s_()

    # for compatibility, do not delete
    def render(self, **kw):
        pass

# ====================================================================================================
