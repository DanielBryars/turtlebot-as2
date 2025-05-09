import sys
import tty
import termios
import time
from math import pi
from env.robot import *  # RobEnv, ros, etc
from robot_environment import *
import select

# ACTIONS
FORWARDS = 1
LEFT = 0
RIGHT = 2
RESET = -10

# Key to action mappings
key_action_map = {
    'w': FORWARDS,
    'a': LEFT,
    'd': RIGHT,
    'r': RESET,  # Reset environment
}

def get_key():
    """Capture a single keypress without printing it to the terminal."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Set raw mode and disable ECHO
        tty.setraw(fd)
        attrs = termios.tcgetattr(fd)
        attrs[3] = attrs[3] & ~(termios.ECHO)
        termios.tcsetattr(fd, termios.TCSADRAIN, attrs)

        key = sys.stdin.read(1)  # Block until a key is pressed
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return key

def print_intro():
    print(r"""

 _____ _   _ ____ _____ _     _____       
|_   _| | | |  _ \_   _| |   | ____|      
  | | | | | | |_) || | | |   |  _|        
  | | | |_| |  _ < | | | |___| |___       
  |_|_ \___/|_| \_\|_|_|_____|_____|_     
 / ___/ _ \| \ | |_   _|  _ \ / _ \| |    
| |  | | | |  \| | | | | |_) | | | | |    
| |__| |_| | |\  | | | |  _ <| |_| | |___ 
 \____\___/|_| \_| |_| |_| \_\\___/|_____|

 READY TO ROLL. MANUAL REMOTE CONTROL ONLINE.
 COMMANDS: (w = forward, a = left, d = right, r = reset, q = quit)
""")

def main():
    if not ros.ok():
        ros.init()

    θspeed = pi / 6
    speed = 10.0
    n = 10

    print(f"n = {n}")
    env = vRobEnv(speed=speed, θspeed=θspeed, n=n, verbose=True)
    env.reset()

    print_intro()

    nSteps = 0
    try:
        while True:
            key = get_key()

            #print(f"KeyDetected: '{key}'")

            os.system('clear')  # or 'cls' if on Windows
            print("q to quit")

            if key == 'q':
                print("\nExiting...")
                break

            if key in key_action_map:
                action = key_action_map[key]
                print(f"{key}-->{action}")

                if action == RESET:
                    env.reset()
                    print("\rEnvironment reset.")
                    
                else:
                    obs, reward, done, info = env.step(action)
                    nSteps += 1
                    

            print(f'\rAction: {action} | Reward: {reward:.2f} | Done: {done} | nSteps:{nSteps}')
            print()
            
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

if __name__ == '__main__':
    main()
