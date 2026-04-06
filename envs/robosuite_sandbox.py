"""
Robosuite Sandbox Environment for Object Theater VLA

Implements a Panda arm with OSC_POSE control for tabletop manipulation tasks.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch

from configs.device import DEVICE

try:
    import robosuite
    from robosuite import make
    from robosuite.controllers import load_controller_config
    from robosuite.utils.transformations import quaternion_to_euler
except ImportError:
    robosuite = None


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
        self.controller_config = load_controller_config(default_controller="OSC_POSE")
        
        # Create the environment with custom settings
        self.env = make(
            "Lift",  # Using Lift as base for tabletop manipulation
            robots="Panda",
            controller_configs=self.controller_config,
            has_renderer=render_mode == "human",
            has_offscreen_renderer=True,
            render_camera=camera_name,
            render_height=image_size[0],
            render_width=image_size[1],
            control_freq=control_freq,
            horizon=horizon,
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
        """
        obs = self.env.reset()
        return obs
    
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
        truncated = self.env._step_count >= self.horizon
        return obs, reward, terminated, truncated, info
    
    def render(self) -> np.ndarray:
        """
        Render the current environment state.
        
        Returns:
            RGB image array of shape (H, W, 3)
        """
        return self.env.render(camera_name=self.camera_name, height=self.image_size[0], width=self.image_size[1])
    
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
        return sum(self.observation_space["robot0_agentview_image"].shape)
    
    def get_scene_objects(self) -> List[Dict[str, Any]]:
        """
        Get information about manipulable objects in the scene.
        
        Returns:
            List of object dictionaries with name, type, and position.
        """
        # The environment has Box, Cylinder, and Sphere objects by default
        objects = []
        for obj_name in self.env.model.names:
            if "object" in obj_name.lower():
                obj_info = {
                    "name": obj_name,
                    "type": self.env.model.body_type(obj_name),
                    "pos": self.env.model.body_pos(obj_name),
                }
                objects.append(obj_info)
        return objects
    
    def get_proprioceptive_state(self) -> Dict[str, np.ndarray]:
        """
        Get current proprioceptive state.

        Returns:
            Dictionary with joint positions, velocities, gripper state, and EEF force.
        """
        obs = self.env._get_observations()
        return {
            "joint_pos": obs["robot0_joint_pos"],
            "joint_vel": obs["robot0_joint_vel"],
            "gripper_qpos": obs["robot0_gripper_qpos"],
            "gripper_qvel": obs["robot0_gripper_qvel"],
            "eef_pos": obs["robot0_eef_pos"],
            "eef_quat": obs["robot0_eef_quat"],
            # Access force sensor through the robot's proprioceptive state
            "robot0_eef_force": obs.get("robot0_eef_force", np.zeros(3)),
        }
    
    def get_camera_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current camera observations.
        
        Returns:
            Dictionary with RGB images from agent and hand cameras.
        """
        obs = self.env._get_observations()
        return {
            "agentview_image": obs["robot0_agentview_image"],
            "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
        }

    def get_force_sensor(self) -> np.ndarray:
        """
        Get current force torque sensor reading at the end-effector.

        Returns:
            3D force vector [Fx, Fy, Fz] in Newtons.
        """
        obs = self.env._get_observations()
        return obs.get("robot0_eef_force", np.zeros(3))
    
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
    # Initialize environment
    env = RobosuiteSandbox(render_mode="rgb_array")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset and get initial observation
    obs = env.reset()
    print(f"Initial observation keys: {obs.keys()}")
    
    # Take a random action
    action = np.zeros(env.action_dim)
    action[:6] = np.random.randn(6) * 0.01  # Small end-effector movement
    action[6] = 0.0  # Gripper close
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step result - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    
    # Close environment
    env.close()
