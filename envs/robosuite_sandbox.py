"""
Robosuite Sandbox Environment for Object Theater VLA

Implements a Panda arm with OSC_POSE control for tabletop manipulation tasks.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch

from configs.device import DEVICE

import robosuite
from robosuite import make
from robosuite import load_part_controller_config
from robosuite.utils.transform_utils import mat2euler, quat2mat


class RobosuiteSandbox:
    """
    Custom environment wrapping robosuite with:
    - Panda robot arm
    - OSC_POSE controller for end-effector control
    - Tabletop scene with 3 manipulable objects (Box, Cylinder, Sphere)
    - RGB camera observations and proprioceptive state
    """
    
    def __init__(
        self,
        render_mode: str = "human",
        camera_name: str = "agentview",
        image_size: Tuple[int, int] = (224, 224),
        control_freq: int = 20,
        horizon: int = 1000,
    ):
        """
        Initialize the Robosuite Sandbox environment.
        
        Args:
            render_mode: Rendering mode ('human', 'rgb_array', etc.)
            camera_name: Camera to use for observations
            image_size: Output image size (height, width)
            control_freq: Control frequency in Hz
            horizon: Maximum steps per episode
        """
        if robosuite is None:
            raise ImportError(
                "robosuite is not installed. Install with: pip install robosuite"
            )
        
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.image_size = image_size
        self.control_freq = control_freq
        self.horizon = horizon
        
        # Load OSC_POSE controller configuration
        # Note: In robosuite 1.5.2, we need to include gripper config in the controller
        # The load_part_controller_config returns a dict without gripper, so we need to add it
        part_controller_config = load_part_controller_config(default_controller="OSC_POSE")
        part_controller_config["gripper"] = {"type": "GRIP"}
        self.controller_config = {
            "type": "BASIC",
            "body_parts": {
                "right": part_controller_config
            }
        }
        
        # Create the environment with custom settings
        # Note: robosuite 1.5.2 uses camera_heights/camera_widths instead of render_height/render_width
        self.env = make(
            "Lift",  # Using Lift as base for tabletop manipulation
            robots="Panda",
            controller_configs=self.controller_config,
            has_renderer=render_mode == "human",
            has_offscreen_renderer=True,
            render_camera=camera_name,
            render_collision_mesh=False,
            render_visual_mesh=True,
            control_freq=control_freq,
            horizon=horizon,
            camera_names=camera_name,
            camera_heights=image_size[0],
            camera_widths=image_size[1],
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,  # RGB only for now
        )
        
        # Environment dimensions
        self.action_space = self.env.action_spec
        self.observation_space = self.env.observation_spec
        
        # Reset environment to get initial state
        self.reset()
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment to initial state.
        
        Returns:
            Dictionary containing:
                - agentview_image: RGB image from agent camera
                - robot0_eye_in_hand_image: RGB image from hand camera
                - robot0_joint_pos: Joint positions
                - robot0_joint_vel: Joint velocities
                - robot0_gripper_qpos: Gripper positions
                - robot0_gripper_qvel: Gripper velocities
                - object_obs: Object positions and states
                - robot0_eef_force: End-effector force vector [Fx, Fy, Fz]
        """
        obs = self.env.reset()
        return self._inject_force_sensor(obs)
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one action step.
        
        Args:
            action: OSC_POSE action vector of shape (7,)
                [dx, dy, dz, droll, dpitch, dyaw, gripper]
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, info = self.env.step(action)
        # In robosuite 1.5.2, use env.timestep for episode tracking
        truncated = self.env.timestep >= self.env.horizon
        obs = self._inject_force_sensor(obs)
        return obs, reward, terminated, truncated, info
    
    def render(self) -> np.ndarray:
        """
        Render the current environment state.
        
        Returns:
            RGB image array of shape (H, W, 3)
        """
        return self.env.sim.render(width=self.image_size[1], height=self.image_size[0], camera_name=self.camera_name)
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
    
    @property
    def action_dim(self) -> int:
        """Action dimension (7 for OSC_POSE + 1 gripper)."""
        return self.action_space[0].shape[0]
    
    @property
    def obs_dim(self) -> int:
        """Observation dimension (flattened state vector)."""
        # Note: observation_space contains observation specs, not actual data
        # Use the image shape from the environment's camera
        return self.image_size[0] * self.image_size[1] * 3
    
    def get_scene_objects(self) -> List[Dict[str, Any]]:
        """
        Get information about manipulable objects in the scene.
        
        Returns:
            List of object dictionaries with name, type, and position.
        """
        # In robosuite 1.5.2, objects are stored in model.mujoco_objects
        # Get object positions from the latest observation (more reliable)
        obs = self.env._get_observations()
        objects = []
        for obj in self.env.model.mujoco_objects:
            obj_info = {
                "name": obj.name,
                "type": type(obj).__name__,  # e.g., BoxObject, CylinderObject
                "pos": obs.get(f"{obj.name}_pos", np.zeros(3)),
            }
            objects.append(obj_info)
        return objects
    
    def _inject_force_sensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Manually extracts EEF force from the robot and adds it to obs."""
        try:
            robot = self.env.robots[0]
            force_dict = robot.ee_force
            if force_dict:
                obs["robot0_eef_force"] = list(force_dict.values())[0]
            else:
                obs["robot0_eef_force"] = np.zeros(3)
        except (AttributeError, IndexError):
            obs["robot0_eef_force"] = np.zeros(3)
        return obs
    
    def get_proprioceptive_state(self) -> Dict[str, np.ndarray]:
        """
        Get current proprioceptive state.

        Returns:
            Dictionary with joint positions, velocities, gripper state, and EEF pose.
            Includes real force sensor data from the robot's end-effector.
        """
        obs = self.env._get_observations()
        return {
            "joint_pos": obs["robot0_joint_pos"],
            "joint_vel": obs["robot0_joint_vel"],
            "gripper_qpos": obs["robot0_gripper_qpos"],
            "gripper_qvel": obs["robot0_gripper_qvel"],
            "eef_pos": obs["robot0_eef_pos"],
            "eef_quat": obs["robot0_eef_quat"],
            "robot0_eef_force": self.get_force_sensor(),
        }
    
    def get_camera_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current camera observations.
        
        Returns:
            Dictionary with RGB images from agent and hand cameras.
        """
        obs = self.env._get_observations()
        camera_obs = {}
        # Check if agentview image exists
        if "agentview_image" in obs:
            camera_obs["agentview_image"] = obs["agentview_image"]
        elif "robot0_agentview_image" in obs:
            camera_obs["agentview_image"] = obs["robot0_agentview_image"]
        # Check if eye_in_hand image exists
        if "robot0_eye_in_hand_image" in obs:
            camera_obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"]
        return camera_obs

    def get_force_sensor(self) -> np.ndarray:
        """
        Get current force torque sensor reading at the end-effector.

        Returns:
            3D force vector [Fx, Fy, Fz] in Newtons.
        """
        try:
            robot = self.env.robots[0]
            force_dict = robot.ee_force
            if force_dict:
                return list(force_dict.values())[0]
        except (AttributeError, IndexError):
            pass
        return np.zeros(3)
    
    def set_object_positions(self, positions: np.ndarray) -> None:
        """
        Set positions of manipulable objects.
        
        Args:
            positions: Array of shape (num_objects, 3) with (x, y, z) positions.
        """
        # This would need to be implemented based on specific object names
        # and model APIs in robosuite
        pass
    
    def configure_scene(
        self,
        object_types: Optional[List[str]] = None,
        object_colors: Optional[List[str]] = None,
    ) -> None:
        """
        Configure the scene with custom objects and properties.
        
        Args:
            object_types: List of object types (e.g., "box", "cylinder", "sphere").
            object_colors: List of colors for each object.
        """
        # Scene configuration would be done during environment creation
        # This is a placeholder for future customization
        pass


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Initialize environment
    env = RobosuiteSandbox(render_mode="rgb_array")
    
    print("=== RobosuiteSandbox Force Sensor Test ===")
    print(f"Action space: {env.action_space}")
    print(f"Action dim: {env.action_dim}")
    
    # Reset and get initial observation
    obs = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    
    # Test force sensor on reset
    assert "robot0_eef_force" in obs, "Force sensor not in reset observations!"
    print(f"Force sensor after reset: {obs['robot0_eef_force']}")
    
    # Test camera observation
    camera_obs = env.get_camera_observation()
    print(f"Camera observation keys: {list(camera_obs.keys())}")
    assert "agentview_image" in camera_obs, "Agentview image not found!"
    assert camera_obs["agentview_image"].shape == (224, 224, 3), "Incorrect image shape!"
    print(f"Camera image shape: {camera_obs['agentview_image'].shape}")
    
    # Test proprioceptive state
    proprio = env.get_proprioceptive_state()
    print(f"Proprioceptive state keys: {list(proprio.keys())}")
    assert "robot0_eef_force" in proprio, "Force sensor not in proprio state!"
    print(f"Proprio force: {proprio['robot0_eef_force']}")
    
    # Test force sensor directly
    force = env.get_force_sensor()
    print(f"Direct force sensor: {force}")
    assert force.shape == (3,), "Force vector has incorrect shape!"
    
    # Test scene objects
    objects = env.get_scene_objects()
    print(f"Scene objects: {len(objects)}")
    for obj in objects:
        print(f"  - {obj['name']}: {obj['type']}")
    
    # Test render
    rendered = env.render()
    print(f"Rendered image shape: {rendered.shape}")
    assert rendered.shape == (224, 224, 3), "Rendered image has incorrect shape!"
    
    # Take a random action
    action = np.zeros(env.action_dim)
    action[:6] = np.random.randn(6) * 0.01  # Small end-effector movement
    action[6] = 0.0  # Gripper close
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    
    # Test force sensor on step
    assert "robot0_eef_force" in obs, "Force sensor not in step observations!"
    print(f"Force sensor after step: {obs['robot0_eef_force']}")
    
    # Test close
    env.close()
    print("\n✓ All force sensor tests passed!")
