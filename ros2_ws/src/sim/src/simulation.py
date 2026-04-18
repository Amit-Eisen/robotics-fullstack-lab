#!/usr/bin/env python3
"""PyBullet Simulation - Racecar with velocity control and terrain"""

import pybullet as p
import pybullet_data
import numpy as np
import cv2


class BasicSimulation:
    """PyBullet racecar simulation with velocity-based control"""
    
    def __init__(self, use_heightmap=True):
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        if use_heightmap:
            self.terrain_id = self._create_terrain()
        else:
            self.terrain_id = p.loadURDF("plane.urdf")
        
        self._add_obstacles()
        
        self.car_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.5])
        
        self._setup_car_joints()
        self._setup_friction()
        
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 60
        self.ext_cam_distance = 5
        self.ext_cam_height = 3
        
        self.tgtVel = 0.0
        self.target_steering = 0.0
        self.max_velocity = 10.0
        self.max_steering = 0.5
        
        self.obstacles = []
    
    def _create_terrain(self):
        """Create heightmap terrain with hills"""
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
            numHeightfieldColumns=resolution
        )
        
        terrain_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0]
        )
        
        p.changeDynamics(terrain_id, -1, lateralFriction=1.0)
        p.changeVisualShape(terrain_id, -1, rgbaColor=[0.3, 0.5, 0.3, 1])
        
        return terrain_id
    
    def _add_obstacles(self):
        """Add obstacles to the environment"""
        self.obstacle_ids = []
        
        obstacle_positions = [
            [5, 0, 0.5],
            [8, 3, 0.5],
            [8, -3, 0.5],
            [12, 0, 0.5],
            [15, 5, 0.5],
            [15, -5, 0.5],
            [-5, 2, 0.5],
            [-5, -2, 0.5],
        ]
        
        for pos in obstacle_positions:
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], 
                                           rgbaColor=[0.8, 0.2, 0.2, 1])
            
            obstacle = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=pos
            )
            self.obstacle_ids.append(obstacle)
        
        cylinder_positions = [
            [3, 4, 0.75],
            [3, -4, 0.75],
            [10, 2, 0.75],
            [10, -2, 0.75],
        ]
        
        for pos in cylinder_positions:
            col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.3, height=1.5)
            vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=1.5,
                                           rgbaColor=[0.2, 0.2, 0.8, 1])
            
            obstacle = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=pos
            )
            self.obstacle_ids.append(obstacle)
    
    def _setup_car_joints(self):
        """Find wheel and steering joints in racecar URDF"""
        self.wheel_joints = []
        self.steering_joints = []
        
        num_joints = p.getNumJoints(self.car_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.car_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            if 'wheel' in joint_name.lower():
                self.wheel_joints.append(i)
            if 'steer' in joint_name.lower() or 'hinge' in joint_name.lower():
                self.steering_joints.append(i)
        
        if not self.wheel_joints:
            self.wheel_joints = [2, 3, 5, 7]
        if not self.steering_joints:
            self.steering_joints = [4, 6]
    
    def _setup_friction(self):
        """Set friction for realistic physics"""
        p.changeDynamics(self.terrain_id, -1, lateralFriction=1.0)
        p.changeDynamics(self.car_id, -1, linearDamping=0.1, angularDamping=0.1)
        
        for joint in self.wheel_joints:
            p.changeDynamics(self.car_id, joint, lateralFriction=1.2)
    
    def apply_control(self):
        """Apply velocity control to wheels and steering"""
        vel = np.clip(self.tgtVel, -self.max_velocity, self.max_velocity)
        steer = np.clip(self.target_steering, -self.max_steering, self.max_steering)
        
        wheel_velocity = vel * 10
        
        for joint in self.wheel_joints:
            p.setJointMotorControl2(
                self.car_id,
                joint,
                p.VELOCITY_CONTROL,
                targetVelocity=wheel_velocity,
                force=20
            )
        
        for joint in self.steering_joints:
            p.setJointMotorControl2(
                self.car_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=steer,
                force=10
            )
    
    def step(self):
        """Step simulation forward"""
        self.apply_control()
        p.stepSimulation()
    
    def get_car_state(self):
        """Get car position, orientation, and velocity"""
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        vel, ang_vel = p.getBaseVelocity(self.car_id)
        euler = p.getEulerFromQuaternion(orn)
        
        return {
            'position': pos,
            'orientation': orn,
            'euler': euler,
            'linear_velocity': vel,
            'angular_velocity': ang_vel,
            'speed': np.linalg.norm(vel[:2])
        }
    
    def get_camera_image(self):
        """Get camera image from car's perspective"""
        state = self.get_car_state()
        pos = state['position']
        yaw = state['euler'][2]
        
        cam_x = pos[0] + 0.5 * np.cos(yaw)
        cam_y = pos[1] + 0.5 * np.sin(yaw)
        cam_z = pos[2] + 0.3
        
        target_x = cam_x + 10 * np.cos(yaw)
        target_y = cam_y + 10 * np.sin(yaw)
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[target_x, target_y, cam_z],
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100.0
        )
        
        _, _, rgb_img, depth_img, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(self.camera_height, self.camera_width, 4)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        
        return bgr_array, depth_img
    
    def get_external_camera_image(self):
        """Get chase camera image behind the car"""
        state = self.get_car_state()
        pos = state['position']
        yaw = state['euler'][2]
        
        cam_x = pos[0] - self.ext_cam_distance * np.cos(yaw)
        cam_y = pos[1] - self.ext_cam_distance * np.sin(yaw)
        cam_z = pos[2] + self.ext_cam_height
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[cam_x, cam_y, cam_z],
            cameraTargetPosition=[pos[0], pos[1], pos[2] + 0.5],
            cameraUpVector=[0, 0, 1]
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=70,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100.0
        )
        
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(self.camera_height, self.camera_width, 4)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        
        return bgr_array
    
    def get_lidar_scan(self):
        """Simulate 2D LiDAR scan - returns list of distances"""
        state = self.get_car_state()
        pos = state['position']
        yaw = state['euler'][2]
        
        lidar_pos = [pos[0], pos[1], pos[2] + 0.3]
        num_rays = 72
        max_range = 15.0
        
        distances = []
        for i in range(num_rays):
            angle = yaw + (i / num_rays) * 2 * np.pi
            ray_to = [
                lidar_pos[0] + max_range * np.cos(angle),
                lidar_pos[1] + max_range * np.sin(angle),
                lidar_pos[2]
            ]
            
            result = p.rayTest(lidar_pos, ray_to)
            hit_fraction = result[0][2]
            distances.append(max_range * hit_fraction)
        
        return distances
    
    def reset_car(self, position=[0, 0, 0.5], orientation=[0, 0, 0, 1]):
        """Reset car to specified position"""
        p.resetBasePositionAndOrientation(self.car_id, position, orientation)
        p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0])
        self.tgtVel = 0.0
        self.target_steering = 0.0
    
    def disconnect(self):
        """Disconnect from PyBullet"""
        if p.isConnected():
            p.disconnect()
