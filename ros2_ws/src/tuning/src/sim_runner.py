"""
Headless simulation runner for controller tuning.
Bypasses ROS2 for fast iteration during optimization.
"""

import sys
import math
import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Dict, Optional
from dataclasses import dataclass

from controllers import PIDController, PurePursuit, Pose2D, Waypoint
from cost_function import TrajectoryPoint, EpisodeResult


@dataclass
class SimConfig:
    use_gui: bool = False
    use_obstacles: bool = True
    use_heightmap: bool = False
    max_episode_time: float = 30.0
    dt: float = 1.0 / 240.0
    control_dt: float = 1.0 / 30.0
    target_velocity: float = 2.0
    goal_tolerance: float = 1.0


@dataclass
class ControllerParams:
    kp: float
    ki: float
    kd: float
    kff: float
    lookahead_min: float
    k_lookahead: float
    wheelbase: float = 0.3


class SimRunner:
    """Runs headless PyBullet simulation with controllers for tuning."""
    
    def __init__(self, config: Optional[SimConfig] = None):
        self.config = config or SimConfig()
        self.physics_client = None
        self.car_id = None
        self.terrain_id = None
        self.obstacle_ids = []
        self.wheel_joints = []
        self.steering_joints = []
    
    def _connect(self):
        """Connect to PyBullet physics server."""
        if self.physics_client is not None:
            try:
                if p.isConnected(self.physics_client):
                    return
            except:
                pass
            self.physics_client = None
        
        mode = p.GUI if self.config.use_gui else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.config.dt, physicsClientId=self.physics_client)
    
    def _disconnect(self):
        """Disconnect from PyBullet physics server."""
        if self.physics_client is not None:
            try:
                if p.isConnected(self.physics_client):
                    p.disconnect(self.physics_client)
            except:
                pass
            self.physics_client = None
    
    def _setup_environment(self):
        """Setup terrain and obstacles."""
        if self.config.use_heightmap:
            self.terrain_id = self._create_terrain()
        else:
            self.terrain_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        if self.config.use_obstacles:
            self._add_obstacles()
    
    def _create_terrain(self):
        """Create heightmap terrain with hills."""
        terrain_size = 50
        resolution = 128
        height_scale = 0.3
        
        heightfield = np.zeros((resolution, resolution), dtype=np.float32)
        
        for i in range(resolution):
            for j in range(resolution):
                x = (i / resolution - 0.5) * terrain_size
                y = (j / resolution - 0.5) * terrain_size
                
                dist_from_center = np.sqrt(x**2 + y**2)
                if dist_from_center < 8:
                    heightfield[i, j] = 0
                else:
                    heightfield[i, j] = (
                        np.sin(x * 0.3) * np.cos(y * 0.3) * height_scale +
                        np.sin(x * 0.1 + y * 0.1) * height_scale * 0.5
                    )
        
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[terrain_size / resolution, terrain_size / resolution, 1],
            heightfieldData=heightfield.flatten(),
            numHeightfieldRows=resolution,
            numHeightfieldColumns=resolution,
            physicsClientId=self.physics_client
        )
        
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        
        p.changeDynamics(terrain_id, -1, lateralFriction=1.0, physicsClientId=self.physics_client)
        return terrain_id
    
    def _add_obstacles(self):
        """Add obstacles to the environment."""
        obstacle_positions = [
            [5, 0, 0.5], [8, 3, 0.5], [8, -3, 0.5], [12, 0, 0.5],
            [15, 5, 0.5], [15, -5, 0.5], [-5, 2, 0.5], [-5, -2, 0.5],
        ]
        
        for pos in obstacle_positions:
            col_shape = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5],
                physicsClientId=self.physics_client
            )
            obstacle = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                basePosition=pos,
                physicsClientId=self.physics_client
            )
            self.obstacle_ids.append(obstacle)
        
        cylinder_positions = [
            [3, 4, 0.75], [3, -4, 0.75], [10, 2, 0.75], [10, -2, 0.75],
        ]
        
        for pos in cylinder_positions:
            col_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.3, height=1.5,
                physicsClientId=self.physics_client
            )
            obstacle = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                basePosition=pos,
                physicsClientId=self.physics_client
            )
            self.obstacle_ids.append(obstacle)
    
    def _setup_car(self, start_pos: List[float], start_yaw: float = 0.0):
        """Load and configure the racecar."""
        start_orn = p.getQuaternionFromEuler([0, 0, start_yaw])
        self.car_id = p.loadURDF(
            "racecar/racecar.urdf", start_pos, start_orn,
            physicsClientId=self.physics_client
        )
        
        self.wheel_joints = []
        self.steering_joints = []
        
        num_joints = p.getNumJoints(self.car_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.car_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            
            if 'wheel' in joint_name.lower():
                self.wheel_joints.append(i)
            if 'steer' in joint_name.lower() or 'hinge' in joint_name.lower():
                self.steering_joints.append(i)
        
        if not self.wheel_joints:
            self.wheel_joints = [2, 3, 5, 7]
        if not self.steering_joints:
            self.steering_joints = [4, 6]
        
        p.changeDynamics(
            self.car_id, -1, linearDamping=0.1, angularDamping=0.1,
            physicsClientId=self.physics_client
        )
        for joint in self.wheel_joints:
            if joint < num_joints:
                p.changeDynamics(
                    self.car_id, joint, lateralFriction=1.2,
                    physicsClientId=self.physics_client
                )
    
    def _get_car_state(self) -> Dict:
        """Get car position, orientation, and velocity."""
        pos, orn = p.getBasePositionAndOrientation(
            self.car_id, physicsClientId=self.physics_client
        )
        vel, ang_vel = p.getBaseVelocity(
            self.car_id, physicsClientId=self.physics_client
        )
        euler = p.getEulerFromQuaternion(orn)
        
        return {
            'position': pos,
            'euler': euler,
            'velocity': vel,
            'speed': np.linalg.norm(vel[:2])
        }
    
    def _apply_control(self, velocity_cmd: float, steering_cmd: float):
        """Apply velocity and steering commands to the car."""
        wheel_velocity = velocity_cmd * 10
        
        num_joints = p.getNumJoints(self.car_id, physicsClientId=self.physics_client)
        
        for joint in self.wheel_joints:
            if joint < num_joints:
                p.setJointMotorControl2(
                    self.car_id, joint, p.VELOCITY_CONTROL,
                    targetVelocity=wheel_velocity, force=20,
                    physicsClientId=self.physics_client
                )
        
        for joint in self.steering_joints:
            if joint < num_joints:
                p.setJointMotorControl2(
                    self.car_id, joint, p.POSITION_CONTROL,
                    targetPosition=steering_cmd, force=10,
                    physicsClientId=self.physics_client
                )
    
    def _check_collision(self) -> bool:
        """Check if car collided with any obstacle."""
        for obs_id in self.obstacle_ids:
            contacts = p.getContactPoints(
                self.car_id, obs_id, physicsClientId=self.physics_client
            )
            if contacts:
                return True
        return False
    
    def run_episode(self, params: ControllerParams, path: List[Waypoint]) -> EpisodeResult:
        """Run a single episode with given controller parameters and path."""
        self._disconnect()
        self._connect()
        
        self.obstacle_ids = []
        self.wheel_joints = []
        self.steering_joints = []
        
        self._setup_environment()
        
        if path:
            start_pos = [path[0].x, path[0].y, 0.5]
            if len(path) > 1:
                start_yaw = math.atan2(path[1].y - path[0].y, path[1].x - path[0].x)
            else:
                start_yaw = 0.0
        else:
            start_pos = [0, 0, 0.5]
            start_yaw = 0.0
        
        self._setup_car(start_pos, start_yaw)
        
        pid = PIDController(params.kp, params.ki, params.kd, params.kff)
        pid.set_limits(-15.0, 15.0)
        
        pursuit = PurePursuit(params.lookahead_min, params.k_lookahead, params.wheelbase)
        pursuit.set_path(path)
        
        trajectory = []
        sim_time = 0.0
        control_accumulator = 0.0
        collision = False
        completed = False
        
        current_steering = 0.0
        current_velocity = 0.0
        
        while sim_time < self.config.max_episode_time:
            p.stepSimulation(physicsClientId=self.physics_client)
            sim_time += self.config.dt
            control_accumulator += self.config.dt
            
            if self._check_collision():
                collision = True
                break
            
            if control_accumulator >= self.config.control_dt:
                control_accumulator = 0.0
                
                state = self._get_car_state()
                current_pose = Pose2D(
                    x=state['position'][0],
                    y=state['position'][1],
                    yaw=state['euler'][2]
                )
                current_speed = state['speed']
                
                steering, success = pursuit.compute_steering(current_pose, current_speed)
                if success:
                    current_steering = steering
                
                velocity_cmd = pid.compute(current_speed, self.config.control_dt, 
                                          self.config.target_velocity)
                current_velocity = velocity_cmd
                
                self._apply_control(current_velocity, current_steering)
                
                trajectory.append(TrajectoryPoint(
                    pose=current_pose,
                    velocity=current_speed,
                    steering=current_steering,
                    timestamp=sim_time
                ))
                
                # Only check goal after minimum time (avoids false completion on closed loops)
                min_time_before_goal_check = 2.0
                if sim_time > min_time_before_goal_check and pursuit.reached_goal(current_pose, self.config.goal_tolerance):
                    completed = True
                    break
        
        return EpisodeResult(
            trajectory=trajectory,
            path=path,
            collision=collision,
            completed=completed,
            total_time=sim_time
        )
    
    def cleanup(self):
        """Cleanup and disconnect."""
        self._disconnect()


def run_single_test(params: ControllerParams, path: List[Waypoint], 
                    config: Optional[SimConfig] = None) -> EpisodeResult:
    """Convenience function to run a single test episode."""
    runner = SimRunner(config)
    try:
        result = runner.run_episode(params, path)
    finally:
        runner.cleanup()
    return result
