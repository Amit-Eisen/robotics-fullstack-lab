# Autonomous Car (ROS 2) — early prototype

This repo is my robotics “full stack” playground.

Right now it contains a small ROS 2 pipeline (simulation + controller) and an Optuna-based tuner:
- **Sim (Python / PyBullet)**: publishes `/odom` and accepts `/cmd_vel`
- **Control (C++ / ROS 2)**: Pure Pursuit + PID(+FF) velocity control, subscribes to `/odom` and `/path`
- **Tuning (Python)**: headless PyBullet rollouts + cost function to tune controller params

## What’s inside
- `ros2_ws/src/sim`: PyBullet sim node + launch
- `ros2_ws/src/control`: `control_node` (Pure Pursuit + PID) + launch + params
- `ros2_ws/src/tuning`: Optuna tuner, cost function, and results utilities

## Quick start (ROS 2)
Assuming you already have ROS 2 installed and sourced.

Build:
```bash
cd ros2_ws
colcon build
source install/setup.bash
```

Run sim + controller (two terminals):
```bash
ros2 launch sim sim_launch.py
```

```bash
ros2 launch control control.launch.py
```

Publish a simple goal (controller will generate a straight-line path):
```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 10.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

## Tuning (Optuna, headless)
The tuner runs headless PyBullet episodes (no ROS 2) for fast iteration.

```bash
python3 -m pip install -r requirements.txt
python3 ros2_ws/src/tuning/src/optuna_tuner.py --config ros2_ws/src/tuning/config/tuning_config.yaml
```

## Notes / next steps
- The current sim is PyBullet; I plan to migrate to **modern Gazebo (gz sim)** for richer sensors/worlds.
- Results under `ros2_ws/src/tuning/results/` are generated artifacts and are ignored by git.

