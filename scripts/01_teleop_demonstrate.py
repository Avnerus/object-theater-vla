"""
Teleoperation Demonstration Script for Object Theater VLA

Records user-driven Panda arm trajectories with:
- Image frames from RGB cameras
- OSC_POSE action sequences
- User-provided text labels
- Saves to HDF5 format
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import h5py
import time
import cv2
import threading
import queue
from dataclasses import dataclass
import json
import os

from configs.device import DEVICE
from configs.config import EnvironmentConfig, default_config
from envs.robosuite_sandbox import RobosuiteSandbox

# Try to import robosuite devices
try:
    from robosuite.devices import Keyboard, SpaceMouse
    ROBOSUITE_DEVICES_AVAILABLE = True
except ImportError:
    ROBOSUITE_DEVICES_AVAILABLE = False


@dataclass
class DemonstrationRecord:
    """Single demonstration record."""
    observations: List[Dict[str, np.ndarray]]
    actions: List[np.ndarray]
    text_label: str
    timestamp: float


class TeleopDemonstrator:
    """
    Records teleoperation demonstrations for imitation learning.
    
    Uses Keyboard or SpaceMouse to drive the Panda arm while recording:
    - RGB camera observations
    - End-effector actions (OSC_POSE)
    - User-provided text labels
    """
    
    def __init__(
        self,
        env_config: EnvironmentConfig = None,
        device_type: str = "keyboard",
        output_dir: str = "data/demonstrations",
    ):
        """
        Initialize the teleop demonstrator.
        
        Args:
            env_config: Environment configuration
            device_type: Control device ('keyboard' or 'spacemouse')
            output_dir: Directory to save demonstration data
        """
        if env_config is None:
            env_config = default_config.env
        
        self.env_config = env_config
        self.device_type = device_type
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment
        self.env = RobosuiteSandbox(
            render_mode="rgb_array",
            camera_name=env_config.camera_name,
            image_size=env_config.image_size,
            control_freq=env_config.control_freq,
            horizon=env_config.horizon,
        )
        
        # Initialize control device
        self._init_control_device()
        
        # Data storage
        self.current_demonstration: Optional[DemonstrationRecord] = None
        self.demonstrations: List[DemonstrationRecord] = []
        
        # Control state
        self.is_recording = False
        self.text_label = ""
        self.action_queue: queue.Queue = queue.Queue()
        
        # Start control loop thread
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
    
    def _init_control_device(self) -> None:
        """Initialize the teleoperation control device."""
        if not ROBOSUITE_DEVICES_AVAILABLE:
            raise ImportError(
                "Robosuite devices not available. Install with: pip install robosuite"
            )
        
        if self.device_type == "keyboard":
            self.device = Keyboard()
        elif self.device_type == "spacemouse":
            self.device = SpaceMouse()
        else:
            raise ValueError(f"Unknown device type: {self.device_type}")
        
        self.device.start_control()
    
    def _control_loop(self) -> None:
        """Background thread for handling control input."""
        while True:
            if self.is_recording and self.current_demonstration is not None:
                # Get action from control device
                action = self._get_device_action()
                if action is not None:
                    self.action_queue.put(action)
            
            time.sleep(1.0 / self.env_config.control_freq)
    
    def _get_device_action(self) -> Optional[np.ndarray]:
        """
        Get action from control device.
        
        Returns:
            OSC_POSE action vector or None
        """
        if self.device_type == "keyboard":
            return self._get_keyboard_action()
        elif self.device_type == "spacemouse":
            return self._get_spacemouse_action()
        return None
    
    def _get_keyboard_action(self) -> Optional[np.ndarray]:
        """Get action from keyboard input."""
        # Keyboard action mapping would depend on robosuite implementation
        # This is a placeholder
        return None
    
    def _get_spacemouse_action(self) -> Optional[np.ndarray]:
        """Get action from SpaceMouse input."""
        # SpaceMouse action mapping would depend on robosuite implementation
        # This is a placeholder
        return None
    
    def start_recording(self, text_label: str) -> None:
        """
        Start recording a new demonstration.
        
        Args:
            text_label: Description of the task being demonstrated
        """
        if self.is_recording:
            print("Already recording!")
            return
        
        self.is_recording = True
        self.text_label = text_label
        self.current_demonstration = DemonstrationRecord(
            observations=[],
            actions=[],
            text_label=text_label,
            timestamp=time.time(),
        )
        print(f"Started recording: '{text_label}'")
    
    def stop_recording(self) -> None:
        """Stop current demonstration recording."""
        if not self.is_recording:
            print("Not recording!")
            return
        
        self.is_recording = False
        if self.current_demonstration is not None:
            self.demonstrations.append(self.current_demonstration)
            print(f"Stopped recording. Total demonstrations: {len(self.demonstrations)}")
            self.current_demonstration = None
    
    def record_step(self) -> None:
        """Record one step of demonstration data."""
        if not self.is_recording:
            return
        
        # Get observation
        obs = self.env.get_camera_observation()
        proprio = self.env.get_proprioceptive_state()
        
        # Combine observations
        combined_obs = {
            **{k: torch.from_numpy(v).to(DEVICE) for k, v in obs.items()},
            **{k: torch.from_numpy(v).to(DEVICE) for k, v in proprio.items()},
        }
        
        # Get action from queue
        try:
            action = self.action_queue.get_nowait()
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        except queue.Empty:
            action_np = np.zeros(self.env.action_dim)
        
        # Store data
        self.current_demonstration.observations.append(combined_obs)
        self.current_demonstration.actions.append(action_np)
    
    def save_demonstrations(self, filename: Optional[str] = None) -> str:
        """
        Save all recorded demonstrations to HDF5 file.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"demonstrations_{timestamp}.h5"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, "w") as f:
            # Save metadata
            meta = f.create_group("metadata")
            meta.attrs["num_demonstrations"] = len(self.demonstrations)
            meta.attrs["image_size"] = list(self.env_config.image_size)
            meta.attrs["action_dim"] = self.env.action_dim
            
            # Save each demonstration
            for i, demo in enumerate(self.demonstrations):
                demo_group = f.create_group(f"demonstration_{i}")
                
                # Save text label
                demo_group.attrs["text_label"] = demo.text_label
                demo_group.attrs["timestamp"] = demo.timestamp
                demo_group.attrs["num_steps"] = len(demo.actions)
                
                # Save observations
                obs_group = demo_group.create_group("observations")
                for step, obs in enumerate(demo.observations):
                    step_group = obs_group.create_group(f"step_{step}")
                    for key, value in obs.items():
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        step_group.create_dataset(key, data=value)
                
                # Save actions
                actions_array = np.array(demo.actions)
                demo_group.create_dataset("actions", data=actions_array)
        
        print(f"Saved {len(self.demonstrations)} demonstrations to {filepath}")
        return filepath
    
    def close(self) -> None:
        """Clean up resources."""
        self.env.close()
        if hasattr(self, "device"):
            self.device.stop_control()


# Main entry point
def main():
    """Main function to run teleop demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Teleop demonstration recorder")
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Control device type",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/demonstrations",
        help="Output directory for demonstration data",
    )
    args = parser.parse_args()
    
    if not ROBOSUITE_DEVICES_AVAILABLE:
        print("Error: robosuite devices not available.")
        print("Install with: pip install robosuite")
        return
    
    # Initialize demonstrator
    demonstrator = TeleopDemonstrator(
        device_type=args.device,
        output_dir=args.output_dir,
    )
    
    # Demonstration loop
    print("\n" + "=" * 60)
    print("Teleop Demonstration Recorder")
    print("=" * 60)
    print("\nControls:")
    print("  's' - Start recording")
    print("  'e' - End recording")
    print("  'q' - Quit")
    print("  'l' - List demonstrations")
    print("  'w' - Write demonstrations to file")
    print()
    
    while True:
        try:
            cmd = input("Command: ").strip().lower()
            
            if cmd == "s":
                label = input("Enter task description: ").strip()
                demonstrator.start_recording(label)
                
                # Record steps in background
                while demonstrator.is_recording:
                    demonstrator.record_step()
                    time.sleep(1.0 / demonstrator.env_config.control_freq)
            
            elif cmd == "e":
                demonstrator.stop_recording()
            
            elif cmd == "l":
                print(f"\nRecorded demonstrations: {len(demonstrator.demonstrations)}")
                for i, demo in enumerate(demonstrator.demonstrations):
                    print(f"  {i}: '{demo.text_label}' ({len(demo.actions)} steps)")
                print()
            
            elif cmd == "w":
                filepath = demonstrator.save_demonstrations()
                print(f"Saved to: {filepath}")
            
            elif cmd == "q":
                break
            
            else:
                print(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
    
    demonstrator.close()
    print("Done!")


if __name__ == "__main__":
    main()
