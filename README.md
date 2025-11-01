# TurtleBot3 Reinforcement Learning Navigation

A reinforcement learning project implementing autonomous navigation for TurtleBot3 in a simulated indoor environment. This project explores both linear and non-linear function approximation approaches to train an agent to reach goals while avoiding obstacles.

## Overview

This project demonstrates the application of reinforcement learning algorithms to robotic navigation. The TurtleBot3 agent learns to navigate through a maze-like environment using LIDAR sensor data to reach goal positions in the fewest steps possible.

## Architecture

The system consists of three main components:

- **Environment** (`robot.py`, `robot_environment.py`): Interfaces with Gazebo simulation and ROS2, provides state representation and reward signals
- **RL Agents** (`rl/`): Implements various reinforcement learning algorithms
- **Assessment Notebook** (`Assessment2.ipynb`): Main implementation and experimentation code

```
RL/
├── env/
│   ├── grid.py
│   ├── gridln.py
│   ├── gridnn.py
│   ├── mountainln.py
│   ├── robot.py           # Gazebo interface and environment
│   └── robot_old.py
├── rl/
│   ├── dp.py              # Dynamic programming
│   ├── rl.py              # Core RL logic
│   ├── rlln.py            # Linear approximation model
│   ├── rlnn.py            # Non-linear (neural network) model
│   └── rlselect.py        # Experimental trial runner
├── Assessment2.ipynb      # Main implementation notebook
├── robot_environment.py   # State representation and reward structure
├── remote_control.py      # Manual robot control (WASD)
├── robenv_monitor.py      # Real-time reward/state monitor
└── feature_monitor.py     # Hand-crafted feature testing
```

## Tech Stack

- **Robotics Platform**: ROS2 Humble on Ubuntu 22.04
- **Simulation**: Gazebo Classic
- **Robot**: TurtleBot3 Burger
- **ML Framework**: PyTorch
- **Deep Learning**: CUDA support (NVIDIA GPU)
- **Python Libraries**: NumPy, scikit-learn, tqdm, ipywidgets

## Features

### Model 1: Linear Function Approximation
- SARSA(λ) on-policy learning
- Binary state representation using LIDAR scans
- Hand-crafted features for corner and wall detection
- Configurable hyperparameters (ε, λ, γ, α)

### Model 2: Non-Linear Function Approximation
- Deep Q-Network (DQN) with experience replay
- Direct LIDAR scan input (64 or 360 beams)
- GPU acceleration support
- Target network for stable learning

### Development Tools
- **Remote Control**: Keyboard control (WASD) for testing
- **Environment Monitor**: Real-time state and reward visualization
- **Feature Monitor**: Debug hand-crafted features
- **Calibration Tools**: Speed and rotation calibration

## Installation

### Prerequisites

1. Ubuntu 22.04
2. ROS2 Humble
3. Gazebo Classic
4. TurtleBot3 packages
5. CUDA toolkit (optional, for GPU acceleration)

### Setup

1. Install ROS2 Humble and TurtleBot3:
```bash
# Follow official TurtleBot3 setup guide
# https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
```

4. Launch Gazebo simulation:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_assessment2.launch.py
```

## Usage

### Training the Agent

1. Open `Assessment2.ipynb` in Jupyter:
```bash
jupyter notebook Assessment2.ipynb
```

2. Initialize ROS connection:
```python
if not ros.ok():
    ros.init()
```

3. Configure hyperparameters and train:

**Linear Model (SARSA λ)**:
```python
hyperparameters = {
    'max_t': 2000,
    'ε': 0.2,
    'εmin': 0.05,
    'α': 0.05,
    'γ': 0.99,
    'λ': 0.6,
    'θspeed': pi / 2,
    'speed': 3.0,
    'n': 3
}

env = vRobEnv(speed=3.0, θspeed=pi/2, n=3)
agent = Sarsaλ(env=env, episodes=200, **hyperparameters)
agent.interact(resume=False, save_ep=True)
```

**Non-Linear Model (DQN)**:
```python
env = nnRobEnv()
agent = cudaDQN(
    env=env,
    episodes=300,
    α=5e-4,
    ε=0.9,
    dε=0.995,
    εmin=0.05,
    γ=0.99,
    h1=256,
    h2=256,
    nbuffer=10000,
    nbatch=32
)
agent.interact(resume=False, save_ep=True)
```

### Manual Control and Debugging

Test robot control:
```bash
python remote_control.py
```

Monitor environment state:
```bash
python robenv_monitor.py
```

### Simulation Speed

Accelerate simulation for faster training:
```python
from env.robot import accelerate_sim, set_nscans_LiDAR

accelerate_sim(speed=100)
set_nscans_LiDAR(nscans=64)  # or 360 for higher resolution
```

## Environment Details

### State Representation

**Linear Model**: Binary features indicating obstacle proximity
```python
# Returns 1 if obstacle within 0.3m, 0 otherwise
states = (self.scans <= 0.3).astype(int)
```

**Non-Linear Model**: Normalized LIDAR scans
```python
# Normalizes scan data to [0, 1]
normalised = ((self.scans - min_range) / (max_range - min_range))
```

### Reward Function

The reward function encourages goal-reaching behavior while penalizing collisions:

```python
reward = -0.1              # Small step penalty
       + -5 * at_wall      # Collision penalty
       + 10 * at_goal      # Goal reward
```

### Actions

- **0**: Turn left (π/2 radians)
- **1**: Move forward
- **2**: Turn right (π/2 radians)

## Results and Analysis

The project explored various configurations:

- **Q-Learning**: Showed erratic behavior
- **SARSA**: More conservative, stable learning
- **SARSA(λ)**: Best linear performance, but struggled to converge
- **DQN**: GPU-accelerated learning, variable results based on hidden layer size

### Challenges Encountered

1. Large hyperparameter search space
2. Sparse reward signal in complex maze
3. State representation design
4. Simulation speed limitations (before acceleration)
5. Balancing exploration vs exploitation

### Key Learnings

- Accelerating simulation dramatically improved iteration speed
- On-policy methods (SARSA) proved safer than off-policy (Q-Learning)
- Reward function design critically impacts convergence
- State feature engineering is challenging for LIDAR data

## Customization

### Modify Reward Function

Edit `robot_environment.py`:
```python
def reward_(self, a):
    # Custom reward logic
    return reward
```

### Change State Features

Modify state representation in environment class:
```python
def s_(self):
    # Custom state features
    return state_vector
```

### Adjust Network Architecture

For DQN, modify hidden layers:
```python
agent = cudaDQN(h1=128, h2=128, ...)  # Smaller network
agent = cudaDQN(h1=512, h2=512, ...)  # Larger network
```

## Troubleshooting

### Multicast Traffic (WSL2)

For DDS communication through Windows firewall:
```powershell
New-NetFirewallRule -Name 'WSL' -DisplayName 'WSL' -InterfaceAlias 'vEthernet (WSL (Hyper-V firewall))' -Direction Inbound -Action Allow
```

### Gazebo Not Responding

Restart Gazebo and ensure ROS is initialized:
```bash
killall gzserver gzclient
ros2 launch turtlebot3_gazebo turtlebot3_assessment2.launch.py
```

### CUDA Out of Memory

Reduce batch size or network size:
```python
agent = cudaDQN(nbatch=16, h1=64, h2=64, ...)
```

## References

- [ROS2 TurtleBot3 Documentation](https://emanual.robotis.com/docs/en/platform/turtlebot3/)
- [Reinforcement Learning Course by David Silver](https://www.youtube.com/watch?v=KHZVXao4qXs)
- Feature extraction from laser scan data based on curvature estimation - Núñez et al. (2015)
- Simulation Speed in ROS/Gazebo - CETI TU Dresden

## License

This is an academic project for educational purposes.

## Acknowledgments

Built as part of an MSc AI Reinforcement Learning assessment at University of Leeds.
