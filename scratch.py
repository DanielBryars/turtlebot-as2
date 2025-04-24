from env.robot import *
import numpy as np
from math import pi
from time import sleep
LEFT=0

#    frequencies = [10,20,30]
#    angular_velocity = 0.3


def measure_rotation(robot : RobEnv):
    
    thresholdRadians = 0.01 #about 0.57deg
    
    robot.freq = 10
    
    
    target_angle = 2*pi
    error = pi
    max_error = 2* pi 

    number_to_average = 5

    robot.reset()

    
    final_errors = np.zeros(number_to_average)
    step_counts = np.zeros(number_to_average)
    
    for i in range(number_to_average):
        error = abs(target_angle - robot.θ)
        step_count = 0
        while(error > thresholdRadians):
            step_count += 1
            print(f"step_count:{step_count}: error is: {np.degrees(error):.2f} degrees")
            robot.step(LEFT)
            error = abs(target_angle - robot.θ)
            if error > max_error:
                #anti spin
                raise RuntimeError(f"Loop failed to converge - we're getting further away from the target, the speed is probably too high, or the closed loop frequency is too low: {np.degrees(error):.2f}°")
        
        # Store results in arrays
        final_errors[i] = error
        step_counts[i] = step_count
    
    return final_errors, step_counts

#measure_turn_response(env, angular_velocity=0.3, duration=2.0, reps=5)